"""
Monkey-patch vLLM's Gemma4 model to support loading NVFP4-quantized MoE
checkpoints produced by llm-compressor's per-expert linearization.

Two problems in vLLM's gemma4.py prevent loading these checkpoints:

1. _weight_iterator() doesn't remap per-expert unfused weight names
   (.experts.experts.{id}.{proj}) to the .moe.experts. prefix that
   vLLM's model tree expects.

2. Gemma4Model.load_weights() expert loop builds incorrect parameter
   names for quantized suffixes (weight_packed, weight_scale, etc.)
   and hardcodes ".weight" in the weight_loader call.

This module patches both methods at import time. Import it before
vLLM loads a Gemma4 MoE checkpoint.

Auto-imported via .pth file (installed by serve.py). Guarded so it's
a no-op when vLLM isn't installed.
"""

try:
    from vllm.model_executor.models.gemma4 import (
        Gemma4ForCausalLM,
        Gemma4Model,
    )
except ImportError:
    # vLLM not installed or not importable — nothing to patch
    pass
else:
    from collections.abc import Iterable

    import torch

    from vllm.model_executor.model_loader.weight_utils import (
        default_weight_loader,
        maybe_remap_kv_scale_name,
    )
    from vllm.model_executor.models.utils import is_pp_missing_parameter

    # -------------------------------------------------------------------
    # Fix 1: _weight_iterator in Gemma4ForCausalLM.load_weights
    # -------------------------------------------------------------------

    _original_causal_lm_load_weights = Gemma4ForCausalLM.load_weights

    def _patched_causal_lm_load_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> set[str]:
        """Wrap the weight iterator to remap per-expert unfused names."""

        def _remap_iterator(inner):
            for name, weight in inner:
                if ".experts.experts." in name:
                    name = name.replace(".experts.experts.", ".moe.experts.", 1)
                yield name, weight

        return _original_causal_lm_load_weights(self, _remap_iterator(weights))

    Gemma4ForCausalLM.load_weights = _patched_causal_lm_load_weights

    # -------------------------------------------------------------------
    # Fix 2: Expert weight loading in Gemma4Model.load_weights
    # -------------------------------------------------------------------

    _original_model_load_weights = Gemma4Model.load_weights

    def _patched_model_load_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> set[str]:
        stacked_params_mapping = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        num_experts = getattr(self.config, "num_experts", None) or 0
        expert_params_mapping = [
            (
                "experts.w13_weight"
                if proj_name in ["gate_proj", "up_proj"]
                else "experts.w2_weight",
                f"experts.{expert_id}.{proj_name}",
                expert_id,
                shard_id,
            )
            for expert_id in range(num_experts)
            for shard_id, proj_name in [
                ("w1", "gate_proj"),
                ("w2", "down_proj"),
                ("w3", "up_proj"),
            ]
        ]

        params_dict = dict(self.named_parameters())
        params_dict.update(dict(self.named_buffers()))
        loaded_params: set[str] = set()

        for name, loaded_weight in weights:
            if self.quant_config is not None and (
                scale_name := self.quant_config.get_cache_scale(name)
            ):
                param = params_dict[scale_name]
                weight_loader = getattr(
                    param, "weight_loader", default_weight_loader
                )
                loaded_weight = loaded_weight[0]
                weight_loader(param, loaded_weight)
                loaded_params.add(scale_name)
                continue

            if name.endswith((".k_scale", ".v_scale", ".q_scale", ".prob_scale")):
                remapped_name = maybe_remap_kv_scale_name(name, params_dict)
                if remapped_name is not None and remapped_name in params_dict:
                    param = params_dict[remapped_name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
                    loaded_params.add(remapped_name)
                    continue

            for param_name, shard_name, shard_id in stacked_params_mapping:
                if shard_name not in name:
                    continue
                stacked_name = name.replace(shard_name, param_name)
                if stacked_name not in params_dict:
                    continue
                if is_pp_missing_parameter(stacked_name, self):
                    continue
                param = params_dict[stacked_name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(stacked_name)
                break
            else:
                for (
                    param_name,
                    weight_name,
                    expert_id,
                    shard_id,
                ) in expert_params_mapping:
                    if weight_name not in name:
                        continue

                    # Find what comes after the matched weight_name.
                    # Unquantized: ".weight" or nothing.
                    # NVFP4: ".weight_packed", ".weight_scale",
                    #        ".weight_global_scale", ".input_global_scale"
                    idx = name.index(weight_name)
                    after_match = name[idx + len(weight_name):]

                    # Build the correct FusedMoE parameter name.
                    # param_name is "experts.w13_weight" → base "experts.w13_"
                    base_param = param_name.rsplit("weight", 1)[0]

                    if after_match:
                        # ".weight_packed" → "weight_packed"
                        param_suffix = after_match[1:]
                    else:
                        param_suffix = "weight"

                    moe_name = name[:idx] + base_param + param_suffix

                    if moe_name not in params_dict:
                        continue
                    if is_pp_missing_parameter(moe_name, self):
                        continue
                    param = params_dict[moe_name]
                    weight_loader = param.weight_loader

                    if after_match:
                        # Quantized: pass mapped name so FusedMoE dispatches
                        # correctly (it inspects substrings like "input_scale")
                        weight_loader(
                            param,
                            loaded_weight,
                            moe_name,
                            shard_id=shard_id,
                            expert_id=expert_id,
                        )
                    else:
                        # Unquantized: original behavior
                        weight_loader(
                            param,
                            loaded_weight,
                            weight_name + ".weight",
                            shard_id=shard_id,
                            expert_id=expert_id,
                        )

                    loaded_params.add(moe_name)
                    break
                else:
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    name = maybe_remap_kv_scale_name(name, params_dict)
                    if name is None:
                        continue
                    if is_pp_missing_parameter(name, self):
                        continue
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
                    loaded_params.add(name)

        return loaded_params

    Gemma4Model.load_weights = _patched_model_load_weights
