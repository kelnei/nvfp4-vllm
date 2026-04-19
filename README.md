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
| [`serve.py`](serve.py) | Start an OpenAI-compatible vLLM server |
| [`chat.py`](chat.py) | Interactive terminal chat client |

## Options

### quantize.py

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `Qwen/Qwen2.5-0.5B-Instruct` | HuggingFace model ID or local path |
| `--output` | `<basename>-NVFP4` | Output directory |
| `--samples` | `256` | Calibration samples (more = better accuracy) |
| `--max-len` | `512` | Max tokens per calibration sample |
| `--weight-only` | off | W4A16 mode (no calibration data needed) |
| `--ignore` | `lm_head` | Layer names/regex patterns to exclude (use `re:` prefix for regex) |
| `--dtype` | `auto` | Model dtype: auto, bfloat16, float16 |
| `--trust-remote-code` | off | Trust remote code when loading model/tokenizer |
| `--dataset` | `HuggingFaceH4/ultrachat_200k` | HuggingFace dataset for calibration |

### serve.py

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `./Qwen2.5-0.5B-Instruct-NVFP4` | Path to quantized model |
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
| `--enforce-eager` | off | Disable CUDA graph compilation (useful for debugging) |
| `--enable-prefix-caching` | off | Enable KV cache reuse across requests with shared prefixes |
| `--speculative-config` | none | JSON string or file path for speculative decoding config |
| `--tool-call-parser` | none | Tool/function call parser (e.g. hermes, llama3_json, mistral) |
| `--enable-auto-tool-choice` | off | Let the model decide when to use tools |

### chat.py

| Flag | Default | Description |
|------|---------|-------------|
| `--url` | `http://localhost:8000/v1` | vLLM server URL |
| `--model` | auto-detect | Model name (auto-detected from server if omitted) |
| `--system` | `You are a helpful assistant.` | System prompt |
| `--temperature` | `0.7` | Sampling temperature |
| `--max-tokens` | `512` | Max tokens per response |

Chat commands: `/clear` resets history, `/system` prints the system prompt, `/quit` exits.

## Notes

- vLLM pre-allocates KV cache up to `--gpu-memory-utilization` of VRAM. On a 96 GB card
  with a small model, pass `--gpu-memory-utilization 0.3` to avoid reserving unused memory.
- Confirm NVFP4 kernels are active by checking vLLM logs for:
  `Using NvFp4LinearBackend.VLLM_CUTLASS for NVFP4 GEMM`
- Pre-quantized NVIDIA checkpoints are available on HuggingFace (e.g.
  `nvidia/Llama-3.3-70B-Instruct-FP4`) and can be served directly with
  `python serve.py --model nvidia/Llama-3.3-70B-Instruct-FP4 --quantization modelopt`

See [GUIDE.md](GUIDE.md) for a full walkthrough.

## License

MIT — see [LICENSE](LICENSE)
