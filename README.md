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
uv pip install vllm llmcompressor
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

### serve.py

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `./Qwen2.5-0.5B-Instruct-NVFP4` | Path to quantized model |
| `--port` | `8000` | Server port |
| `--max-model-len` | `32768` | Maximum context length |
| `--gpu-memory-utilization` | `0.90` | Fraction of VRAM reserved for model + KV cache |
| `--quantization` | auto | Force backend (use `modelopt` for NVIDIA pre-quantized checkpoints) |

### chat.py

| Flag | Default | Description |
|------|---------|-------------|
| `--url` | `http://localhost:8000/v1` | vLLM server URL |
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
