"""
Quantize a model to NVFP4 (W4A4) using llm-compressor.

By default attention q/k/v/o projections are kept at FP8 (channel-wise
weights, dynamic per-token activations) and an FP8 KV cache scale is
calibrated, matching the mixed-precision layout used by NVIDIA/unsloth
NVFP4 checkpoints. FP8 attention is ~3.5x closer to the original weights
than NVFP4 for a small size cost (~13% on a dense 12B, ~1% on a MoE).

Usage:
    python quantize.py [--model MODEL_ID] [--output OUTPUT_DIR]
                       [--samples N] [--max-len N] [--weight-only]
                       [--no-fp8-attn] [--gptq-mlp {auto,on,off}]
                       [--ignore PATTERN ...] [--dtype TYPE]
                       [--trust-remote-code] [--dataset DATASET]
                       [--split SPLIT]

Defaults:
    model      = Qwen/Qwen2.5-0.5B-Instruct
    output     = <model-basename>-NVFP4
    samples    = 256
    max-len    = 512
    weight-only = False (W4A4; set flag for W4A16)
    fp8-attn   = True (FP8 attention + FP8 KV cache; the KV cache scale
                 needs calibration, so it is skipped with --weight-only)
    gptq-mlp   = auto (GPTQ + imatrix_mse observer + static actorder on
                 dense MLP gate/up/down projections; calibration-only, same
                 on-disk format, ~20% lower KL vs BF16 than plain minmax.
                 auto enables it for dense models and skips it for MoE,
                 --weight-only, and --no-fp8-attn runs)
    ignore     = lm_head
    dtype      = auto
    dataset    = HuggingFaceH4/ultrachat_200k
    split      = auto (train_sft for ultrachat, train otherwise)
"""

import argparse
import json
from pathlib import Path

from compressed_tensors.quantization import (
    QuantizationArgs,
    preset_name_to_scheme,
)
from compressed_tensors.utils import match_name
from compressed_tensors.utils.safetensors_load import (
    get_safetensors_header,
    get_weight_mappings,
)
from datasets import load_dataset
from llmcompressor import oneshot
from llmcompressor.modifiers.gptq import GPTQModifier
from llmcompressor.modifiers.quantization import QuantizationModifier

# llm-compressor 0.12.0 (current stable) only collects imatrix importance
# statistics when an IMatrixGatherer is prepended to the recipe; without it,
# observer=imatrix_mse silently falls back to a uniform MSE grid search.
# Nightlies (0.12.1a+) remove the class and make the observer self-collecting,
# so treat its absence as "no gatherer needed" rather than an error.
try:
    from llmcompressor.modifiers.transform.imatrix import IMatrixGatherer
except ImportError:
    IMatrixGatherer = None
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    PreTrainedTokenizerBase,
)

# Attention projections kept at FP8 by --fp8-attn. Name-based, so fused-QKV
# and MLA architectures won't match; main() verifies the pattern hits at
# least one module before quantizing.
FP8_ATTN_TARGET = r"re:.*self_attn\.(q|k|v|o)_proj$"
# vLLM fuses q/k/v into a single qkv_proj module and resolves its scheme by
# layer name before class name, so the saved config must name the fused module
# too — otherwise the broad Linear/NVFP4 target wins and vLLM tries to load
# the FP8 shards as NVFP4. Matches nothing at quantization time, where the
# projections are still unfused.
FP8_ATTN_FUSED_TARGET = r"re:.*self_attn\.qkv_proj$"

# Dense MLP projections given the GPTQ + imatrix_mse + actorder treatment by
# --gptq-mlp. MoE expert projections (".mlp.experts.N.gate_proj") do not
# match, deliberately: per-expert calibration coverage is too thin for
# activation-statistics observers to be trustworthy there.
GPTQ_MLP_TARGET = r"re:.*\.mlp\.(gate|up|down)_proj$"


def _collect_tensor_meta(output_dir: Path) -> dict:
    """Map every saved tensor name to its header entry ({dtype, shape, ...})."""
    weight_map = get_weight_mappings(str(output_dir))
    meta = {}
    for shard_path in set(weight_map.values()):
        header = get_safetensors_header(shard_path)
        header.pop("__metadata__", None)
        meta.update(header)
    return meta


def fix_ignore_list(output_dir: Path) -> None:
    """
    llm-compressor records the quantization ignore list using the live
    transformers module hierarchy. Some architectures (e.g. Gemma 4) rename
    modules between that hierarchy and the on-disk checkpoint via a checkpoint
    conversion mapping, so the recorded names don't match what vLLM (which
    loads directly from checkpoint tensor names) expects. Add the on-disk name
    of every unquantized 2D float weight to the ignore list. The recorded
    transformers-side names are kept alongside: transformers matches ignore
    against its live module names when reloading the checkpoint, so both
    spellings are needed.
    """
    config_path = output_dir / "config.json"
    config = json.loads(config_path.read_text())
    qconfig = config.get("quantization_config")
    if qconfig is None:
        return

    tensors = _collect_tensor_meta(output_dir)
    quantized_prefixes = {
        k[: -len(".weight_packed")] for k in tensors if k.endswith(".weight_packed")
    }

    original = set(qconfig.get("ignore", []))
    ignore = set(original)
    for key, meta in tensors.items():
        if not key.endswith(".weight"):
            continue
        prefix = key[: -len(".weight")]
        if prefix in quantized_prefixes:
            continue
        if len(meta["shape"]) != 2 or meta["dtype"] not in (
            "F64",
            "F32",
            "F16",
            "BF16",
        ):
            continue
        ignore.add(prefix)

    if ignore == original:
        return

    qconfig["ignore"] = sorted(ignore)
    config_path.write_text(json.dumps(config, indent=2))
    print(
        f"Added on-disk layer names to quantization ignore list ({len(ignore)} entries)."
    )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument(
        "--output",
        default=None,
        help="Output directory (default: <model-basename>-NVFP4)",
    )
    p.add_argument(
        "--samples",
        type=int,
        default=256,
        help="Number of calibration samples (more = better accuracy)",
    )
    p.add_argument(
        "--max-len",
        type=int,
        default=512,
        help="Max token length per calibration sample",
    )
    p.add_argument(
        "--weight-only",
        action="store_true",
        help="W4A16 (weights only, no calibration data needed)",
    )
    p.add_argument(
        "--fp8-attn",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep attention q/k/v/o projections at FP8 (channel-wise weights, "
        "dynamic per-token activations) instead of NVFP4, and calibrate an "
        "FP8 KV cache scale. The KV cache scale needs calibration data, so "
        "it is skipped with --weight-only. Disable with --no-fp8-attn for "
        "uniform NVFP4.",
    )
    p.add_argument(
        "--gptq-mlp",
        choices=["auto", "on", "off"],
        default="auto",
        help="Quantize dense MLP gate/up/down projections with GPTQ error "
        "compensation (imatrix_mse observer + static activation ordering) "
        "instead of plain minmax. Calibration-only change: same NVFP4 "
        "on-disk format and serving cost, but ~20%% lower KL vs the BF16 "
        "original (matches unsloth's NVFP4 checkpoints; plain minmax and "
        "scale-only imatrix both measure no better than each other). "
        "'auto' (default) enables it for dense models and skips it for MoE "
        "(per-expert calibration coverage is too thin), --weight-only, and "
        "--no-fp8-attn runs. 'on' requires calibration data, --fp8-attn, "
        "and a non-MoE model.",
    )
    p.add_argument(
        "--ignore",
        nargs="+",
        default=["lm_head"],
        help="Layer names/regex patterns to exclude from quantization "
        "(default: lm_head). Use re: prefix for regex patterns.",
    )
    p.add_argument(
        "--dtype",
        default="auto",
        help="Model dtype: auto, bfloat16, float16 (default: auto)",
    )
    p.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code when loading model/tokenizer",
    )
    p.add_argument(
        "--dataset",
        default="HuggingFaceH4/ultrachat_200k",
        help="HuggingFace dataset for calibration (default: HuggingFaceH4/ultrachat_200k)",
    )
    p.add_argument(
        "--split",
        default=None,
        help="Dataset split for calibration (default: train_sft for "
        "ultrachat_200k, train otherwise)",
    )
    p.add_argument(
        "--cpu-offload",
        action="store_true",
        help="Load model to CPU/system RAM; llm-compressor dispatches "
        "layers to GPU during calibration. Use for large MoE models "
        "that don't fit alongside expert-unpacking overhead.",
    )
    return p.parse_args()


def main():
    args = parse_args()

    model_id = args.model
    basename = model_id.rstrip("/").split("/")[-1]
    suffix = "-NVFP4-W4A16" if args.weight_only else "-NVFP4"
    output_dir = args.output or (basename + suffix)

    scheme = "NVFP4A16" if args.weight_only else "NVFP4"
    # Regex targets outrank the class-name target when a module matches both
    # groups, so attention projections land in the FP8 group and every other
    # Linear falls through to NVFP4.
    config_groups = {"group_0": preset_name_to_scheme(scheme, ["Linear"])}
    kv_cache_scheme = None
    if args.fp8_attn:
        attn_scheme = preset_name_to_scheme(
            "FP8_DYNAMIC", [FP8_ATTN_TARGET, FP8_ATTN_FUSED_TARGET]
        )
        if args.weight_only:
            # Honor the A16 contract: FP8 weights only (W8A16), activations
            # stay at model dtype. The KV cache scale is skipped because its
            # static observer needs calibration data.
            attn_scheme.input_activations = None
        else:
            kv_cache_scheme = QuantizationArgs(
                num_bits=8,
                type="float",
                strategy="tensor",
                symmetric=True,
                dynamic=False,
                observer="static_minmax",
            )
        config_groups["group_1"] = attn_scheme
    gptq_mlp = args.gptq_mlp
    if gptq_mlp == "on":
        if args.weight_only:
            raise SystemExit(
                "--gptq-mlp on needs calibration data to collect activation "
                "statistics; it cannot be combined with --weight-only."
            )
        if not args.fp8_attn:
            raise SystemExit(
                "--gptq-mlp on moves the MLP projections into a GPTQ modifier "
                "and drops the catch-all NVFP4 Linear group, so it needs "
                "--fp8-attn to cover the attention projections."
            )
    elif gptq_mlp == "auto" and (args.weight_only or not args.fp8_attn):
        gptq_mlp = "off"

    mode = "W4A16 (weight-only)" if args.weight_only else "W4A4 (weights + activations)"
    if args.fp8_attn:
        mode += " + FP8 attention"
    if kv_cache_scheme is not None:
        mode += " + FP8 KV cache"
    if gptq_mlp == "on":
        mode += " + GPTQ MLP"

    print(f"Model:       {model_id}")
    print(f"Mode:        {mode}")
    print(f"Output dir:  {output_dir}")
    print(f"Dtype:       {args.dtype}")
    print(f"Ignore:      {args.ignore}")
    if not args.weight_only:
        print(f"Calibration: {args.samples} samples, max {args.max_len} tokens each")
        print(f"Dataset:     {args.dataset}")

    print("\nLoading model...")
    load_kwargs = dict(dtype=args.dtype)
    if not args.cpu_offload:
        load_kwargs["device_map"] = "auto"
    if args.trust_remote_code:
        load_kwargs["trust_remote_code"] = True
    model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=args.trust_remote_code
    )

    if args.fp8_attn and not any(
        match_name(name, FP8_ATTN_TARGET) for name, _ in model.named_modules()
    ):
        raise SystemExit(
            f"--fp8-attn matched no modules ({FP8_ATTN_TARGET!r}); this "
            "architecture likely fuses QKV or uses MLA naming. Rerun with "
            "--no-fp8-attn or adjust FP8_ATTN_TARGET."
        )

    # --gptq-mlp needs the live module tree: auto skips MoE models, and the
    # recipe with GPTQ drops the catch-all Linear group, which would leave
    # MoE expert projections unquantized entirely.
    if gptq_mlp != "off":
        module_names = [name for name, _ in model.named_modules()]
        has_dense_mlp = any(match_name(n, GPTQ_MLP_TARGET) for n in module_names)
        is_moe = any(n.endswith(".experts") or ".experts." in n for n in module_names)
        if gptq_mlp == "auto":
            if is_moe:
                print(
                    "GPTQ MLP:    auto -> skipped (MoE experts detected; "
                    "per-expert calibration coverage is too thin)"
                )
                gptq_mlp = "off"
            elif not has_dense_mlp:
                print(
                    "GPTQ MLP:    auto -> skipped (no modules match "
                    f"{GPTQ_MLP_TARGET!r})"
                )
                gptq_mlp = "off"
            else:
                print("GPTQ MLP:    auto -> enabled (dense MLP projections)")
                gptq_mlp = "on"
        else:
            if is_moe:
                raise SystemExit(
                    "--gptq-mlp on drops the catch-all NVFP4 Linear group, "
                    "which would leave this model's MoE expert projections "
                    "unquantized. Use --gptq-mlp auto or off for MoE models."
                )
            if not has_dense_mlp:
                raise SystemExit(
                    f"--gptq-mlp on matched no modules ({GPTQ_MLP_TARGET!r}); "
                    "this architecture names its MLP projections differently. "
                    "Rerun with --gptq-mlp off or adjust GPTQ_MLP_TARGET."
                )

    if gptq_mlp == "on":
        # GPTQ error compensation is what actually delivers the KL gain — a
        # plain QuantizationModifier with observer=imatrix_mse only refines
        # group scales and measures no better than minmax (verified by paired
        # per-token KL vs BF16 on gemma-4-12B, 2026-07-29). GPTQ owns the MLP
        # projections; the catch-all Linear group is dropped so the two
        # modifiers never claim the same module (matching the group layout of
        # NVIDIA/unsloth NVFP4 checkpoints).
        del config_groups["group_0"]
        mlp_scheme = preset_name_to_scheme(scheme, [GPTQ_MLP_TARGET])
        mlp_scheme.weights.observer = "imatrix_mse"
        mlp_scheme.weights.actorder = "static"
        recipe = [
            QuantizationModifier(
                config_groups=config_groups,
                ignore=args.ignore,
                kv_cache_scheme=kv_cache_scheme,
            ),
            GPTQModifier(
                config_groups={"group_2": mlp_scheme},
                ignore=args.ignore,
            ),
        ]
        if IMatrixGatherer is not None:
            # llm-compressor 0.12.0 only collects imatrix importance stats
            # when the gatherer runs first; nightlies self-collect (the class
            # is gone there, hence the guarded import).
            recipe.insert(
                0, IMatrixGatherer(targets=[GPTQ_MLP_TARGET], ignore=args.ignore)
            )
    else:
        recipe = QuantizationModifier(
            config_groups=config_groups,
            ignore=args.ignore,
            kv_cache_scheme=kv_cache_scheme,
        )

    # compressed-tensors initializes KV cache scales from the top-level model
    # config, but multimodal configs (e.g. Gemma 4) nest the attention geometry
    # inside text_config. Mirror the attributes it reads onto the top-level
    # config, and drop them after quantization so they don't leak into the
    # saved config.json.
    kv_shim_attrs = []
    text_config = model.config.get_text_config()
    if kv_cache_scheme is not None and text_config is not model.config:
        for attr in ("num_attention_heads", "num_key_value_heads", "head_dim"):
            if not hasattr(model.config, attr) and hasattr(text_config, attr):
                setattr(model.config, attr, getattr(text_config, attr))
                kv_shim_attrs.append(attr)

    if args.weight_only:
        print("Running weight-only quantization (no calibration data needed)...")
        oneshot(model=model, recipe=recipe)
    else:
        print(f"Loading calibration dataset ({args.dataset})...")

        # ultrachat names its split "train_sft"; most datasets use "train"
        split = args.split or (
            "train_sft" if args.dataset == "HuggingFaceH4/ultrachat_200k" else "train"
        )

        ds = load_dataset(args.dataset, split=f"{split}[:{args.samples}]")
        ds = ds.shuffle(seed=42)

        def preprocess(example):
            # Support datasets with "messages" (chat) or "text" (raw text) columns
            if "messages" in example:
                text = tokenizer.apply_chat_template(
                    example["messages"], tokenize=False
                )
            elif "text" in example:
                text = example["text"]
            elif "article" in example:
                text = example["article"]
            else:
                # Fall back to first string column
                text = next(
                    v for v in example.values() if isinstance(v, str) and len(v) > 0
                )
            return {"text": text}

        def tokenize(sample):
            return tokenizer(
                sample["text"],
                padding=False,
                max_length=args.max_len,
                truncation=True,
                add_special_tokens=False,
            )

        ds = ds.map(preprocess)
        ds = ds.map(tokenize, remove_columns=ds.column_names)

        print("Applying NVFP4 quantization with calibration data...")
        oneshot(
            model=model,
            dataset=ds,
            recipe=recipe,
            max_seq_length=args.max_len,
            num_calibration_samples=args.samples,
        )

    for attr in kv_shim_attrs:
        delattr(model.config, attr)

    print(f"\nSaving quantized model to ./{output_dir} ...")
    model.save_pretrained(output_dir, save_compressed=True)
    tokenizer.save_pretrained(output_dir)

    # Multimodal models (e.g. Gemma 4) need preprocessor_config.json for the
    # vision/audio feature extractor. Save the processor if there is one
    # (for text-only models AutoProcessor just returns the tokenizer again).
    try:
        processor = AutoProcessor.from_pretrained(
            model_id, trust_remote_code=args.trust_remote_code
        )
    except Exception:
        processor = None
    if processor is not None and not isinstance(processor, PreTrainedTokenizerBase):
        processor.save_pretrained(output_dir)
        if hasattr(processor, "image_processor"):
            processor.image_processor.save_pretrained(output_dir)
        print("Saved processor config (multimodal model)")

    fix_ignore_list(Path(output_dir))

    size_mb = (
        sum(f.stat().st_size for f in Path(output_dir).rglob("*") if f.is_file())
        / 1024**2
    )
    print(f"Done. Output size: {size_mb:.0f} MB  (saved to ./{output_dir})")


if __name__ == "__main__":
    main()
