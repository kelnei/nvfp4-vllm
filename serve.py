"""
Start a vLLM OpenAI-compatible server for a quantized model.

Usage:
    python serve.py [--model PATH] [--port PORT] [--max-model-len N]

Defaults:
    model         = ./Qwen2.5-0.5B-Instruct-NVFP4
    port          = 8000
    max-model-len = 32768
"""

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="./Qwen2.5-0.5B-Instruct-NVFP4",
                   help="Path to quantized model directory")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--max-model-len", type=int, default=32768)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90,
                   help="Fraction of VRAM vLLM may use (model + KV cache). "
                        "Default 0.90 pre-allocates ~86 GB on a 96 GB card. "
                        "Use 0.3 or lower for small models during development.")
    p.add_argument("--quantization", default=None,
                   help="Force quantization backend (e.g. 'modelopt' for NVIDIA "
                        "pre-quantized checkpoints). Auto-detected for "
                        "compressed-tensors models.")
    return p.parse_args()


def main():
    args = parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: model path '{args.model}' does not exist.", file=sys.stderr)
        print("Run quantize.py first, or pass --model <path>.", file=sys.stderr)
        sys.exit(1)

    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", str(model_path),
        "--dtype", "auto",
        "--port", str(args.port),
        "--max-model-len", str(args.max_model_len),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
    ]

    if args.quantization:
        cmd += ["--quantization", args.quantization]

    print(f"Model:   {model_path}")
    print(f"Port:    {args.port}")
    print(f"GPU mem: {args.gpu_memory_utilization:.0%} of VRAM reserved for model + KV cache")
    print(f"URL:     http://localhost:{args.port}/v1")
    print()
    print("Waiting for NVFP4 kernels to load and CUDA graphs to compile (~60s first run)...")
    print("Look for: 'Using NvFp4LinearBackend.VLLM_CUTLASS for NVFP4 GEMM'")
    print("Press Ctrl+C to stop.\n")

    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\nServer stopped.")


if __name__ == "__main__":
    main()
