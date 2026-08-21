# nvfp4-vllm

Scripts for quantizing language models to NVFP4 and serving them with vLLM.

**Requirements:** NVIDIA Blackwell GPU (SM 12.0+), Python 3.12, [uv](https://github.com/astral-sh/uv)

## Setup

```bash
# Install system dependencies (one-time)
sudo apt-get install -y python3.12-dev gcc

# Create virtual environment and install dependencies
uv venv .venv --python 3.12
source .venv/bin/activate
uv sync
```

## Quickstart

```bash
# 1. Quantize a model to NVFP4
python quantize.py --model Qwen/Qwen2.5-0.5B-Instruct

# 2. Serve it (terminal 1)
python serve.py

# 3. Chat with it (terminal 2)
python chat.py
```

## Scripts

| Script | Description |
|--------|-------------|
| [`quantize.py`](quantize.py) | Quantize any HuggingFace model to NVFP4 (W4A4 or W4A16) |
| [`verify.py`](verify.py) | Check a quantized checkpoint for silent corruption (bit flips) |
| [`serve.py`](serve.py) | Start an OpenAI-compatible vLLM server |
| [`chat.py`](chat.py) | Interactive terminal chat client |

## Options

### quantize.py

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `Qwen/Qwen2.5-0.5B-Instruct` | HuggingFace model ID or local path |
| `--output` | `<basename>-NVFP4` | Output directory |
| `--samples` | `512` | Calibration samples (more = better accuracy) |
| `--max-len` | `1024` | Max tokens per calibration sample |
| `--weight-only` | off | W4A16 mode (no calibration data needed) |
| `--fp8-attn` | on | Keep attention q/k/v/o projections at FP8 and calibrate an FP8 KV cache scale (KV scale skipped with `--weight-only`); disable with `--no-fp8-attn` for uniform NVFP4 |
| `--fp8-deltanet` | off | On hybrid linear-attention models, keep the Gated DeltaNet qkv/z/out projections at FP8 instead of NVFP4 (~+1.5GB on a 27B for most of the DeltaNet quantization KL back); no effect elsewhere |
| `--fp8-lm-head` | off | Quantize the output head to FP8 and drop it from `--ignore`. Halves the largest tensor left in an NVFP4 checkpoint (2.4 GiB on Qwen3.8-27B), which single-stream decode reads once per `lm_head` pass — three times per step under k=2 MTP. Measured [+14% decode throughput](GUIDE.md#fp8-output-head) at k=2 for +0.0005 weight-only KL |
| `--gptq-mlp` | `auto` | Quantize dense MLP gate/up/down projections with GPTQ error compensation (imatrix_mse observer + static actorder): same on-disk format and serving cost, ~20% lower KL vs the BF16 original. `auto` enables it for dense models and skips MoE, `--weight-only`, and `--no-fp8-attn` runs; `on`/`off` force it |
| `--gptq-mlp-projections` | `gate,up,down` | Which dense MLP projections `--gptq-mlp` hands to GPTQ; the rest get plain minmax NVFP4 in a group of their own. Keep the default — `gate,up` measured as a tie/loss on Qwen3.8-27B, see [GPTQ and down_proj](GUIDE.md#gptq-and-down_proj) |
| `--gptq-mlp-damp` | `0.01` | GPTQ dampening fraction, single value or `PROJ=FRAC` pairs (`down=0.1`). Heavier damping pulls GPTQ back toward plain rounding; see [GPTQ and down_proj](GUIDE.md#gptq-and-down_proj) |
| `--fp8-mlp` | `off` | Keep the most quantization-sensitive dense MLP layers at FP8 instead of NVFP4, trading size for fidelity. `top:N` picks the N layers whose NVFP4-over-FP8 KL against the unquantized model is largest, an explicit list like `1,2,3,10-15` names them outright, and `scan` prints the [ranking](GUIDE.md#mixed-precision-mlp) and exits |
| `--sensitivity-dataset` | `ultrachat` | Data the `--fp8-mlp` ranking is measured on. Deliberately not the same default as `--dataset`: ranking layers on the wide mixture [picks measurably worse layers](GUIDE.md#why-the-scan-uses-a-different-corpus-than-calibration) |
| `--sensitivity-samples` | `64` | Calibration samples the `--fp8-mlp` ranking measures KL over |
| `--sensitivity-report` | none | Write the full `--fp8-mlp` ranking to a path as JSON |
| `--int8-ple` | `off` | Quantize the Gemma-4 E-series per-layer embedding table to weight-only INT8, taking ~30% off an E2B checkpoint in VRAM as well as on disk. Off by default because it is [a real trade](GUIDE.md#per-layer-embeddings-gemma-4-e-series): free on English chat, ~8% KL on multilingual/tool/code prompts under W4A4 |
| `--ignore` | `lm_head` | Layer names/regex patterns to exclude (use `re:` prefix for regex) |
| `--pipeline` | `auto` | Calibration pipeline. `basic` runs plain full-model forwards — needed for architectures the sequential tracer cannot split into per-layer subgraphs (e.g. gemma-4 E-series shared-KV lookups) |
| `--dtype` | `auto` | Model dtype: auto, bfloat16, float16 |
| `--trust-remote-code` | off | Trust remote code when loading model/tokenizer |
| `--dataset` | `mix` | Calibration data: a [named mixture](GUIDE.md#calibration-data) (`mix`, `ultrachat`) or any HuggingFace dataset ID |
| `--vision-samples` | `auto` | Share of `--samples` that carry an image. `auto` = 12.5% when the checkpoint has an image processor, 0 otherwise |
| `--split` | auto | Dataset split (`train_sft` for ultrachat, `train` otherwise); ignored for multi-source mixtures |
| `--cpu-offload` | off | Load model to system RAM; llm-compressor dispatches layers to GPU during calibration (use for large MoE models) |

### verify.py

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | required | Quantized checkpoint directory to verify |
| `--orig` | none | Source checkpoint (path or HF ID) the quantization ran from; enables the byte-identity and weight-deviation checks |
| `--quick` | off | Skip the weight-deviation scan (the slow check) |

Checks that pass-through tensors are byte-identical to the source, that
global/KV scales are finite, that block scales have no extreme outliers,
and that dequantized weights sit within the expected quantization error of
the source weights. Catches sign- and exponent-bit flips; a flip smaller
than ~2 scale-units is inside normal quantization noise (and equally
harmless). Exits non-zero on findings.

### serve.py

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `./Qwen2.5-0.5B-Instruct-NVFP4` | Path to quantized model, or a HuggingFace model ID |
| `--host` | `0.0.0.0` | Bind address |
| `--port` | `8000` | Server port |
| `--served-model-name` | model path | Model name exposed in the API |
| `--dtype` | `auto` | Model dtype: auto, bfloat16, float16, float32 |
| `--seed` | none | Random seed for reproducibility |
| `--trust-remote-code` | off | Trust remote code when loading model/tokenizer |
| `--max-model-len` | `32768` | Maximum context length |
| `--gpu-memory-utilization` | `0.90` | Fraction of VRAM reserved for model + KV cache |
| `--tensor-parallel-size` | `1` | Number of GPUs for tensor parallelism |
| `--pipeline-parallel-size` | `1` | Number of GPUs for pipeline parallelism |
| `--max-num-seqs` | vLLM default | Max concurrent sequences (batch size) |
| `--quantization` | auto | Force backend (use `modelopt` for NVIDIA pre-quantized checkpoints) |
| `--kv-cache-dtype` | `auto` | KV cache dtype: auto, fp8, fp8_e5m2, fp8_e4m3 |
| `--linear-backend` | auto (`cutlass` if nvcc missing) | Force the GEMM kernel backend (e.g. `cutlass`, `marlin`, `flashinfer_cutlass`) |
| `--attention-backend` | auto (`TRITON_ATTN` if nvcc missing) | Force the attention backend (e.g. `FLASHINFER`, `TRITON_ATTN`); FlashInfer JIT-compiles with nvcc at startup |
| `--enforce-eager` | off | Disable CUDA graph compilation (useful for debugging) |
| `--enable-prefix-caching` | off | Enable KV cache reuse across requests with shared prefixes |
| `--speculative-config` | none | JSON string or file path for speculative decoding config |
| `--tool-call-parser` | none | Tool/function call parser (e.g. hermes, llama3_json, mistral) |
| `--enable-auto-tool-choice` | off | Let the model decide when to use tools |

Any flag not listed above is passed through to vLLM unchanged, so every
`vllm serve` option is available (e.g. `--swap-space 8`).

### chat.py

| Flag | Default | Description |
|------|---------|-------------|
| `--url` | `http://localhost:8000/v1` | vLLM server URL |
| `--model` | auto-detect | Model name (auto-detected from server if omitted) |
| `--system` | `You are a helpful assistant.` | System prompt |
| `--temperature` | `0.7` | Sampling temperature |
| `--max-tokens` | `512` | Max tokens per response |

Chat commands: `/clear` resets history, `/system` prints the system prompt
(`/system <text>` sets a new one), `/quit` exits.

## Notes

- vLLM pre-allocates KV cache up to `--gpu-memory-utilization` of VRAM. On a 96 GB card
  with a small model, pass `--gpu-memory-utilization 0.3` to avoid reserving unused memory.
- Confirm NVFP4 kernels are active by checking vLLM logs for:
  `Using CutlassNvFp4LinearKernel for NVFP4 GEMM` (a warning about the
  emulation backend means the optimized kernels did not load)
- Pre-quantized NVIDIA checkpoints are available on HuggingFace (e.g.
  `nvidia/Llama-3.3-70B-Instruct-FP4`) and can be served directly with
  `python serve.py --model nvidia/Llama-3.3-70B-Instruct-FP4 --quantization modelopt`

See [GUIDE.md](GUIDE.md) for a full walkthrough.

## License

MIT — see [LICENSE](LICENSE)
