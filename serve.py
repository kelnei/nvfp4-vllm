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
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()

    # Model and server basics
    p.add_argument(
        "--model",
        default="./Qwen2.5-0.5B-Instruct-NVFP4",
        help="Path to quantized model directory, or a HuggingFace model ID",
    )
    p.add_argument("--host", default="0.0.0.0", help="Bind address (default: 0.0.0.0)")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument(
        "--served-model-name",
        default=None,
        help="Model name exposed in the API (defaults to model path)",
    )
    p.add_argument(
        "--dtype", default="auto", help="Model dtype: auto, bfloat16, float16, float32"
    )
    p.add_argument(
        "--seed", type=int, default=None, help="Random seed for reproducibility"
    )
    p.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code when loading model/tokenizer",
    )

    # Memory and parallelism
    p.add_argument("--max-model-len", type=int, default=32768)
    p.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.90,
        help="Fraction of VRAM vLLM may use (model + KV cache). "
        "Default 0.90 pre-allocates ~86 GB on a 96 GB card. "
        "Use 0.3 or lower for small models during development.",
    )
    p.add_argument(
        "--tensor-parallel-size",
        "-tp",
        type=int,
        default=1,
        help="Number of GPUs for tensor parallelism",
    )
    p.add_argument(
        "--pipeline-parallel-size",
        "-pp",
        type=int,
        default=1,
        help="Number of GPUs for pipeline parallelism",
    )
    p.add_argument(
        "--max-num-seqs",
        type=int,
        default=None,
        help="Max concurrent sequences (batch size)",
    )

    # Quantization and KV cache
    p.add_argument(
        "--quantization",
        default=None,
        help="Force quantization backend (e.g. 'modelopt' for NVIDIA "
        "pre-quantized checkpoints). Auto-detected for "
        "compressed-tensors models.",
    )
    p.add_argument(
        "--kv-cache-dtype",
        default="auto",
        help="KV cache dtype: auto, fp8, fp8_e5m2, fp8_e4m3. With 'auto', "
        "vLLM selects fp8 on its own when the checkpoint declares an FP8 "
        "kv_cache_scheme (calibrated k/v scales); pass an explicit value "
        "to override.",
    )
    p.add_argument(
        "--linear-backend",
        default=None,
        help="Force the linear-layer GEMM backend (e.g. cutlass, marlin, "
        "flashinfer_cutlass). Default: vLLM auto-selects, except "
        "when the CUDA toolkit is missing — then 'cutlass' is used "
        "since FlashInfer backends JIT-compile with nvcc.",
    )
    p.add_argument(
        "--moe-backend",
        default=None,
        help="Force the fused-MoE kernel backend (e.g. cutlass, marlin, "
        "flashinfer_cutlass). Default: vLLM auto-selects, except "
        "when the CUDA toolkit is missing — then 'cutlass' is used "
        "since FlashInfer backends JIT-compile with nvcc.",
    )
    p.add_argument(
        "--attention-backend",
        default=None,
        help="Force the attention backend (e.g. FLASHINFER, TRITON_ATTN, "
        "FLASH_ATTN). Default: vLLM auto-selects, except when the CUDA "
        "toolkit is missing — then 'TRITON_ATTN' is used since the "
        "FLASHINFER backend JIT-compiles its prefill/decode modules "
        "with nvcc at startup.",
    )

    # Performance
    p.add_argument(
        "--enforce-eager",
        action="store_true",
        help="Disable CUDA graph compilation (useful for debugging)",
    )
    p.add_argument(
        "--enable-prefix-caching",
        action="store_true",
        help="Enable KV cache reuse across requests with shared prefixes",
    )

    # Speculative decoding
    p.add_argument(
        "--speculative-config",
        default=None,
        help="JSON string or file path for speculative decoding config",
    )

    # Tool calling
    p.add_argument(
        "--tool-call-parser",
        default=None,
        help="Tool/function call parser (e.g. hermes, llama3_json, mistral)",
    )
    p.add_argument(
        "--enable-auto-tool-choice",
        action="store_true",
        help="Let the model decide when to use tools (requires --tool-call-parser)",
    )

    # Anything not recognized above is forwarded to vLLM as-is
    return p.parse_known_args()


def cuda_toolkit_available() -> bool:
    """FlashInfer JIT-compiles kernels at startup and needs nvcc to do it.

    Checking for the compiler itself matters: runtime-only CUDA installs
    often have /usr/local/cuda (or CUDA_HOME set) without bin/nvcc.
    """
    if shutil.which("nvcc"):
        return True
    candidates = (
        os.environ.get("CUDA_HOME"),
        os.environ.get("CUDA_PATH"),
        "/usr/local/cuda",
    )
    return any(
        home and os.access(Path(home).expanduser() / "bin" / "nvcc", os.X_OK)
        for home in candidates
    )


def main():
    args, extra_vllm_args = parse_args()

    # Anything not path-like is assumed to be a HuggingFace model ID.
    if args.model.startswith((".", "/", "~")):
        model_path = Path(args.model).expanduser()
        if not model_path.exists():
            print(f"Error: model path '{args.model}' does not exist.", file=sys.stderr)
            print(
                "Run quantize.py first, or pass --model <path-or-HF-id>.",
                file=sys.stderr,
            )
            sys.exit(1)
        args.model = str(model_path)

    env = os.environ.copy()
    linear_backend = args.linear_backend
    moe_backend = args.moe_backend
    attention_backend = args.attention_backend
    have_nvcc = cuda_toolkit_available()
    if not have_nvcc:
        # FlashInfer's NVFP4 GEMM, fused-MoE, top-k/top-p sampling, and
        # (since vLLM 0.27) attention kernels are JIT-compiled at startup
        # and crash without the CUDA toolkit, so steer vLLM to its
        # built-in kernels instead.
        linear_backend = linear_backend or "cutlass"
        moe_backend = moe_backend or "cutlass"
        attention_backend = attention_backend or "TRITON_ATTN"
        env.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
        if args.speculative_config:
            # vLLM never inherits the attention backend for the draft
            # model — it re-autoselects (preferring FlashInfer, which
            # JIT-compiles on the first request) unless the speculative
            # config names one itself.
            spec = args.speculative_config.strip()
            if spec.startswith("{"):
                spec_dict = json.loads(spec)
            else:
                spec_dict = json.loads(Path(spec).expanduser().read_text())
            spec_dict.setdefault("attention_backend", attention_backend)
            args.speculative_config = json.dumps(spec_dict)

    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        args.model,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--dtype",
        args.dtype,
        "--max-model-len",
        str(args.max_model_len),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--kv-cache-dtype",
        args.kv_cache_dtype,
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
        "--pipeline-parallel-size",
        str(args.pipeline_parallel_size),
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
    if linear_backend and linear_backend != "auto":
        cmd += ["--linear-backend", linear_backend]
    if moe_backend and moe_backend != "auto":
        cmd += ["--moe-backend", moe_backend]
    if attention_backend and attention_backend != "auto":
        cmd += ["--attention-backend", attention_backend]
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
    print(
        f"GPU mem: {args.gpu_memory_utilization:.0%} of VRAM reserved for model + KV cache"
    )
    print(f"URL:     http://localhost:{args.port}/v1")
    print()
    print(
        "Waiting for NVFP4 kernels to load and CUDA graphs to compile (~60s first run)..."
    )
    print("Look for: 'Using CutlassNvFp4LinearKernel for NVFP4 GEMM' (or another")
    print("optimized *NvFp4LinearKernel — a warning means it fell back to emulation).")
    print("Press Ctrl+C to stop.\n")

    if not have_nvcc:
        print(
            "Note: CUDA toolkit (nvcc) not found — using built-in CUTLASS "
            "linear and MoE kernels, the Triton attention backend, and "
            "disabling the FlashInfer sampler.\n"
        )

    try:
        subprocess.run(cmd, env=env)
    except KeyboardInterrupt:
        print("\nServer stopped.")


if __name__ == "__main__":
    main()
