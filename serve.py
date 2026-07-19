"""
Start a vLLM OpenAI-compatible server for a quantized model.

Usage:
    python serve.py [--model PATH_OR_HF_ID] [--port PORT] [--max-model-len N] [options]

Defaults:
    model         = ./Qwen2.5-0.5B-Instruct-NVFP4
    port          = 8000
    max-model-len = 32768

--model accepts a local directory or a HuggingFace model ID
(e.g. nvidia/Llama-3.3-70B-Instruct-FP4).

Any flag this script does not define is passed through to vLLM unchanged,
so all `vllm serve` options are available, e.g.:
    python serve.py --model ./my-model --swap-space 8 --disable-log-requests
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()

    # Model and server basics
    p.add_argument("--model", default="./Qwen2.5-0.5B-Instruct-NVFP4",
                   help="Path to quantized model directory, or a HuggingFace model ID")
    p.add_argument("--host", default="0.0.0.0",
                   help="Bind address (default: 0.0.0.0)")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--served-model-name", default=None,
                   help="Model name exposed in the API (defaults to model path)")
    p.add_argument("--dtype", default="auto",
                   help="Model dtype: auto, bfloat16, float16, float32")
    p.add_argument("--seed", type=int, default=None,
                   help="Random seed for reproducibility")
    p.add_argument("--trust-remote-code", action="store_true",
                   help="Trust remote code when loading model/tokenizer")

    # Memory and parallelism
    p.add_argument("--max-model-len", type=int, default=32768)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90,
                   help="Fraction of VRAM vLLM may use (model + KV cache). "
                        "Default 0.90 pre-allocates ~86 GB on a 96 GB card. "
                        "Use 0.3 or lower for small models during development.")
    p.add_argument("--tensor-parallel-size", "-tp", type=int, default=1,
                   help="Number of GPUs for tensor parallelism")
    p.add_argument("--pipeline-parallel-size", "-pp", type=int, default=1,
                   help="Number of GPUs for pipeline parallelism")
    p.add_argument("--max-num-seqs", type=int, default=None,
                   help="Max concurrent sequences (batch size)")

    # Quantization and KV cache
    p.add_argument("--quantization", default=None,
                   help="Force quantization backend (e.g. 'modelopt' for NVIDIA "
                        "pre-quantized checkpoints). Auto-detected for "
                        "compressed-tensors models.")
    p.add_argument("--kv-cache-dtype", default="auto",
                   help="KV cache dtype: auto, fp8, fp8_e5m2, fp8_e4m3")
    p.add_argument("--linear-backend", default="cutlass",
                   help="Linear-layer GEMM backend. Default 'cutlass' uses vLLM's "
                        "built-in kernels; 'auto' lets vLLM pick, but it may choose "
                        "a FlashInfer backend that requires JIT compilation (nvcc). "
                        "Other options: marlin, flashinfer_cutlass, ...")

    # Performance
    p.add_argument("--enforce-eager", action="store_true",
                   help="Disable CUDA graph compilation (useful for debugging)")
    p.add_argument("--enable-prefix-caching", action="store_true",
                   help="Enable KV cache reuse across requests with shared prefixes")

    # Speculative decoding
    p.add_argument("--speculative-config", default=None,
                   help="JSON string or file path for speculative decoding config")

    # Tool calling
    p.add_argument("--tool-call-parser", default=None,
                   help="Tool/function call parser (e.g. hermes, llama3_json, mistral)")
    p.add_argument("--enable-auto-tool-choice", action="store_true",
                   help="Let the model decide when to use tools (requires --tool-call-parser)")

    # Anything not recognized above is forwarded to vLLM as-is
    return p.parse_known_args()


def looks_like_local_path(model: str) -> bool:
    """Distinguish a local path from a HuggingFace model ID."""
    return model.startswith((".", "/", "~")) or Path(model).exists()


def cuda_toolkit_available() -> bool:
    """FlashInfer JIT-compiles kernels at startup and needs nvcc to do it."""
    return bool(
        shutil.which("nvcc")
        or os.environ.get("CUDA_HOME")
        or os.environ.get("CUDA_PATH")
        or Path("/usr/local/cuda").exists()
    )


def main():
    args, extra_vllm_args = parse_args()

    if looks_like_local_path(args.model):
        if not Path(args.model).exists():
            print(f"Error: model path '{args.model}' does not exist.", file=sys.stderr)
            print("Run quantize.py first, or pass --model <path-or-HF-id>.", file=sys.stderr)
            sys.exit(1)

    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", args.model,
        "--host", args.host,
        "--port", str(args.port),
        "--dtype", args.dtype,
        "--max-model-len", str(args.max_model_len),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--kv-cache-dtype", args.kv_cache_dtype,
        "--tensor-parallel-size", str(args.tensor_parallel_size),
        "--pipeline-parallel-size", str(args.pipeline_parallel_size),
    ]

    if args.served_model_name:
        cmd += ["--served-model-name", args.served_model_name]
    if args.seed is not None:
        cmd += ["--seed", str(args.seed)]
    if args.trust_remote_code:
        cmd += ["--trust-remote-code"]
    if args.max_num_seqs is not None:
        cmd += ["--max-num-seqs", str(args.max_num_seqs)]
    if args.quantization:
        cmd += ["--quantization", args.quantization]
    if args.linear_backend and args.linear_backend != "auto":
        cmd += ["--linear-backend", args.linear_backend]
    if args.enforce_eager:
        cmd += ["--enforce-eager"]
    if args.enable_prefix_caching:
        cmd += ["--enable-prefix-caching"]
    if args.speculative_config:
        cmd += ["--speculative-config", args.speculative_config]
    if args.tool_call_parser:
        cmd += ["--tool-call-parser", args.tool_call_parser]
    if args.enable_auto_tool_choice:
        cmd += ["--enable-auto-tool-choice"]
    cmd += extra_vllm_args

    print(f"Model:   {args.model}")
    print(f"Port:    {args.port}")
    print(f"GPU mem: {args.gpu_memory_utilization:.0%} of VRAM reserved for model + KV cache")
    print(f"URL:     http://localhost:{args.port}/v1")
    print()
    print("Waiting for NVFP4 kernels to load and CUDA graphs to compile (~60s first run)...")
    print("Look for: 'Using CutlassNvFp4LinearKernel for NVFP4 GEMM' (or another")
    print("optimized *NvFp4LinearKernel — a warning means it fell back to emulation).")
    print("Press Ctrl+C to stop.\n")

    env = os.environ.copy()
    # vLLM enables FlashInfer's top-k/top-p sampler by default, but that
    # kernel is JIT-compiled at startup and crashes without the CUDA toolkit.
    if not cuda_toolkit_available():
        env.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
        print("Note: CUDA toolkit (nvcc) not found — disabling FlashInfer sampler.\n")

    try:
        subprocess.run(cmd, env=env)
    except KeyboardInterrupt:
        print("\nServer stopped.")


if __name__ == "__main__":
    main()
