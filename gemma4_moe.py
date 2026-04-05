"""
Gemma 4 MoE linearization module for llm-compressor.

Gemma4TextExperts stores expert weights as 3D nn.Parameter tensors:
    gate_up_proj: [num_experts, 2 * moe_intermediate_size, hidden_size]
    down_proj:    [num_experts, hidden_size, moe_intermediate_size]

These are invisible to targets="Linear" quantization. This module unfuses
them into individual Gemma4TextMLP modules with nn.Linear weights so that
each expert can be quantized independently.

Registers against "Gemma4TextExperts" in the MoECalibrationModule registry.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from llmcompressor.modeling.moe_context import MoECalibrationModule
from llmcompressor.utils.dev import skip_weights_initialize

if TYPE_CHECKING:
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextExperts


@MoECalibrationModule.register("Gemma4TextExperts")
class CalibrationGemma4TextExperts(MoECalibrationModule):
    """
    Calibration version of Gemma4TextExperts that unfuses 3D expert parameter
    tensors into individual MLP modules with nn.Linear weights so they can be
    individually quantized.

    is_permanent = True because the unfused structure must persist for
    quantization to target the individual nn.Linear expert weights.
    """

    is_permanent = True

    def __init__(
        self,
        original: Gemma4TextExperts,
        config,
        calibrate_all_experts: bool = True,
    ):
        super().__init__()
        text_config = getattr(config, "text_config", config)

        self.calibrate_all_experts = calibrate_all_experts
        self.num_experts = text_config.num_experts
        self.top_k = text_config.top_k_experts
        self.hidden_dim = text_config.hidden_size

        self.act_fn = original.act_fn
        self.experts = SequentialGemma4Experts(text_config, original)

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        Same interface as Gemma4TextExperts.forward — called by the decoder
        layer as self.experts(hidden_states_2d, top_k_index, top_k_weights).
        """
        final_hidden_states = torch.zeros_like(hidden_states)

        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(
                top_k_index, num_classes=self.num_experts
            )
            expert_mask = expert_mask.permute(2, 1, 0)

        for expert_idx, expert_layer in enumerate(self.experts):
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])

            if self.calibrate_all_experts:
                # Send all tokens through this expert for calibration coverage
                expert_out = expert_layer(hidden_states)[token_idx]
            else:
                expert_out = expert_layer(hidden_states[token_idx])

            if len(token_idx) > 0:
                current_hidden_states = (
                    expert_out * top_k_weights[token_idx, top_k_pos, None]
                )
                final_hidden_states.index_add_(
                    0,
                    token_idx,
                    current_hidden_states.to(hidden_states.dtype),
                )

        return final_hidden_states

    def restore(self, original: torch.nn.Module) -> torch.nn.Module:
        return self


class Gemma4ExpertMLP(nn.Module):
    """
    Single expert MLP with nn.Linear weights (quantization-visible).

    Uses the same gate/up/down pattern as Gemma4TextMLP but without
    the layer_idx / double-wide logic that's irrelevant for MoE experts.
    """

    def __init__(self, hidden_size: int, intermediate_size: int, act_fn):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = act_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class SequentialGemma4Experts(nn.ModuleList):
    """
    Unfuses 3D expert parameter tensors into individual Gemma4ExpertMLP modules
    so that each expert's weights are nn.Linear and can be targeted by
    quantization with targets="Linear".
    """

    def __init__(self, config, original: Gemma4TextExperts):
        self.num_experts = config.num_experts
        hidden_size = config.hidden_size
        intermediate_size = config.moe_intermediate_size

        with skip_weights_initialize():
            super().__init__(
                [
                    Gemma4ExpertMLP(hidden_size, intermediate_size, original.act_fn)
                    for _ in range(self.num_experts)
                ]
            )

        gate_up_data = original.gate_up_proj.data  # [num_experts, 2*inter, hidden]
        down_data = original.down_proj.data  # [num_experts, hidden, inter]

        for i in range(self.num_experts):
            gate_up = gate_up_data[i]  # [2*intermediate, hidden]
            down = down_data[i]  # [hidden, intermediate]

            # gate_up_proj stores [gate; up] stacked along dim 0
            # nn.Linear weight is [out_features, in_features]
            self[i].gate_proj.weight.data = (
                gate_up[:intermediate_size, :].clone().contiguous()
            )
            self[i].up_proj.weight.data = (
                gate_up[intermediate_size:, :].clone().contiguous()
            )
            self[i].down_proj.weight.data = down.clone().contiguous()

        # Free the original 3D tensors to avoid doubling GPU memory.
        # Without this, both the original (~45.7 GB) and cloned weights coexist.
        del gate_up_data, down_data
        original.gate_up_proj = None
        original.down_proj = None
        torch.cuda.empty_cache()
