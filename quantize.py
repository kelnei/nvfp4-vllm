"""
Quantize a model to NVFP4 (W4A4) using llm-compressor.

Usage:
    python quantize.py [--model MODEL_ID] [--output OUTPUT_DIR]
                       [--samples N] [--max-len N] [--weight-only]
                       [--ignore PATTERN ...] [--dtype TYPE]
                       [--trust-remote-code] [--dataset DATASET]

Defaults:
    model      = Qwen/Qwen2.5-0.5B-Instruct
    output     = <model-basename>-NVFP4
    samples    = 256
    max-len    = 512
    weight-only = False (W4A4; set flag for W4A16)
    ignore     = lm_head
    dtype      = auto
    dataset    = HuggingFaceH4/ultrachat_200k
"""

import argparse
from pathlib import Path

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--output", default=None,
                   help="Output directory (default: <model-basename>-NVFP4)")
    p.add_argument("--samples", type=int, default=256,
                   help="Number of calibration samples (more = better accuracy)")
    p.add_argument("--max-len", type=int, default=512,
                   help="Max token length per calibration sample")
    p.add_argument("--weight-only", action="store_true",
                   help="W4A16 (weights only, no calibration data needed)")
    p.add_argument("--ignore", nargs="+", default=["lm_head"],
                   help="Layer names/regex patterns to exclude from quantization "
                        "(default: lm_head). Use re: prefix for regex patterns.")
    p.add_argument("--dtype", default="auto",
                   help="Model dtype: auto, bfloat16, float16 (default: auto)")
    p.add_argument("--trust-remote-code", action="store_true",
                   help="Trust remote code when loading model/tokenizer")
    p.add_argument("--dataset", default="HuggingFaceH4/ultrachat_200k",
                   help="HuggingFace dataset for calibration (default: HuggingFaceH4/ultrachat_200k)")
    p.add_argument("--cpu-offload", action="store_true",
                   help="Load model to CPU/system RAM; llm-compressor dispatches "
                        "layers to GPU during calibration. Use for large MoE models "
                        "that don't fit alongside expert-unpacking overhead.")
    return p.parse_args()


def main():
    args = parse_args()

    model_id = args.model
    basename = model_id.rstrip("/").split("/")[-1]
    suffix = "-NVFP4-W4A16" if args.weight_only else "-NVFP4"
    output_dir = args.output or (basename + suffix)

    print(f"Model:       {model_id}")
    print(f"Mode:        {'W4A16 (weight-only)' if args.weight_only else 'W4A4 (weights + activations)'}")
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

    scheme = "NVFP4A16" if args.weight_only else "NVFP4"
    recipe = QuantizationModifier(targets="Linear", scheme=scheme, ignore=args.ignore)

    if args.weight_only:
        print("Running weight-only quantization (no calibration data needed)...")
        oneshot(model=model, recipe=recipe)
    else:
        print(f"Loading calibration dataset ({args.dataset})...")

        # Determine the split name — ultrachat uses "train_sft", most others use "train"
        if args.dataset == "HuggingFaceH4/ultrachat_200k":
            split = f"train_sft[:{args.samples}]"
        else:
            split = f"train[:{args.samples}]"

        ds = load_dataset(args.dataset, split=split)
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

    print(f"\nSaving quantized model to ./{output_dir} ...")
    model.save_pretrained(output_dir, save_compressed=True)
    tokenizer.save_pretrained(output_dir)

    # Multimodal models (e.g. Gemma 4) need preprocessor_config.json for the
    # vision/audio feature extractor. Save the processor if available.
    try:
        processor = AutoProcessor.from_pretrained(
            model_id, trust_remote_code=args.trust_remote_code
        )
        processor.save_pretrained(output_dir)
        if hasattr(processor, "image_processor"):
            processor.image_processor.save_pretrained(output_dir)
    except Exception:
        pass

    size_mb = sum(
        f.stat().st_size for f in Path(output_dir).rglob("*") if f.is_file()
    ) / 1024**2
    print(f"Done. Output size: {size_mb:.0f} MB  (saved to ./{output_dir})")


if __name__ == "__main__":
    main()
