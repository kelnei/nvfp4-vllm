"""
Quantize a model to NVFP4 (W4A4) using llm-compressor.

Usage:
    python quantize.py [--model MODEL_ID] [--output OUTPUT_DIR]
                       [--samples N] [--max-len N] [--weight-only]

Defaults:
    model      = Qwen/Qwen2.5-0.5B-Instruct
    output     = <model-basename>-NVFP4
    samples    = 256
    max-len    = 512
    weight-only = False (W4A4; set flag for W4A16)
"""

import argparse
from pathlib import Path

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
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
    if not args.weight_only:
        print(f"Calibration: {args.samples} samples, max {args.max_len} tokens each")

    print("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype="auto", device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    scheme = "NVFP4A16" if args.weight_only else "NVFP4"
    recipe = QuantizationModifier(targets="Linear", scheme=scheme, ignore=["lm_head"])

    if args.weight_only:
        print("Running weight-only quantization (no calibration data needed)...")
        oneshot(model=model, recipe=recipe)
    else:
        print("Loading calibration dataset (HuggingFaceH4/ultrachat_200k)...")
        ds = load_dataset(
            "HuggingFaceH4/ultrachat_200k",
            split=f"train_sft[:{args.samples}]",
        )
        ds = ds.shuffle(seed=42)

        def preprocess(example):
            return {
                "text": tokenizer.apply_chat_template(
                    example["messages"], tokenize=False
                )
            }

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

    size_mb = sum(
        f.stat().st_size for f in Path(output_dir).rglob("*") if f.is_file()
    ) / 1024**2
    print(f"Done. Output size: {size_mb:.0f} MB  (saved to ./{output_dir})")


if __name__ == "__main__":
    main()
