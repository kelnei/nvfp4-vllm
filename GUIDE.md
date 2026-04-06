# NVFP4 Quantization with vLLM

End-to-end walkthrough: quantize a model to NVFP4 and serve it with vLLM.

**Hardware used:** NVIDIA RTX PRO 6000 Blackwell Workstation (SM 12.0, 96 GB VRAM)
**Confirmed working:** vLLM 0.19.0, torch 2.10.0+cu128, llmcompressor (git main)

---

## What is NVFP4?

NVFP4 is NVIDIA's 4-bit floating-point format (E2M1 encoding) with two levels of scaling:
- Every **16 values** share an **FP8 (E4M3)** scale factor
- One **FP32** scale factor covers the whole tensor

Effective storage is ~4.5 bits/value — about **3.5× smaller than FP16**.

| Mode | Description | Calibration data? |
|------|-------------|-------------------|
| W4A4 (`NVFP4`) | Weights *and* activations quantized | Yes |
| W4A16 (`NVFP4A16`) | Weights only quantized | No |

W4A4 is faster at inference (both matrix sides are FP4); W4A16 is simpler to produce
but gives less throughput improvement.

**Hardware requirement:** Blackwell GPUs (SM 12.0+). On older architectures vLLM falls
back to weight-only dequantization, losing most of the speedup.

---

## 1. Environment Setup

### System prerequisites

Triton (bundled with torch/vLLM) compiles CUDA kernel utilities at runtime and requires
Python development headers and GCC:

```bash
sudo apt-get install -y python3.12-dev gcc
```

### Python environment

Dependencies are managed via `pyproject.toml`. The released llmcompressor (0.10.0.1)
pins `transformers<=4.57.6` and crashes with `transformers>=5.x` due to a moved import
(`TORCH_INIT_FUNCTIONS`). The fix exists on llm-compressor's `main` branch but is not
yet released, so `pyproject.toml` installs llmcompressor from git main. The `[tool.uv]`
section uses `override-dependencies` to force `transformers>=5.5.0` past llmcompressor's
declared constraint.

```bash
# Create a Python 3.12 virtual environment and install all dependencies
uv venv .venv --python 3.12
source .venv/bin/activate
uv sync

# Verify
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "import vllm; print(vllm.__version__)"
```

Expected output:
```
2.10.0+cu128 True
0.19.0
```

---

## 2. Quantize a Model

The script [`quantize.py`](quantize.py) handles both W4A4 and W4A16 modes.

### Quick demo (W4A4, 0.5B model)

```bash
source .venv/bin/activate

python quantize.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --samples 256 \
  --max-len 512
```

This downloads `Qwen2.5-0.5B-Instruct`, runs calibration against 256 samples from
`HuggingFaceH4/ultrachat_200k`, and saves the result to `./Qwen2.5-0.5B-Instruct-NVFP4/`.

### Weight-only (no calibration data)

```bash
python quantize.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --weight-only
```

Output: `./Qwen2.5-0.5B-Instruct-NVFP4-W4A16/`

### Larger models / better accuracy

```bash
# More calibration samples = better accuracy (512 is a good default for 7B+)
python quantize.py \
  --model meta-llama/Llama-3.2-3B-Instruct \
  --samples 512 \
  --max-len 2048
```

### Multimodal models (Gemma 4, etc.)

Models with vision/audio components need those layers excluded from quantization.
Use `--ignore` with regex patterns:

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Dense model (31B)
python quantize.py \
  --model google/gemma-4-31B-it \
  --samples 512 \
  --max-len 2048 \
  --ignore lm_head "re:.*vision_tower.*" "re:.*audio_tower.*" "re:.*embed_vision.*" "re:.*embed_audio.*"

# MoE model (26B-A4B)
python quantize.py \
  --model google/gemma-4-26B-A4B-it \
  --samples 512 \
  --max-len 2048 \
  --ignore lm_head "re:.*vision_tower.*" "re:.*audio_tower.*" "re:.*embed_vision.*" "re:.*embed_audio.*" "re:.*router.*"
```

**Gemma 4 MoE note:** `serve.py` automatically installs a runtime patch
(`gemma4_vllm_patch.py`) that fixes vLLM's Gemma4 MoE weight loading for NVFP4
checkpoints. The patch remaps per-expert weight names produced by llm-compressor's
MoE linearization to the fused format vLLM expects. No manual steps needed.

Expected output (0.5B model completes in under a minute):
```
Done. Output size: 467 MB  (saved to ./Qwen2.5-0.5B-Instruct-NVFP4)
```
The original FP16 model is ~950 MB — roughly 2× smaller for W4A4.

### Script flags

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `Qwen/Qwen2.5-0.5B-Instruct` | HuggingFace model ID or local path |
| `--output` | `<basename>-NVFP4` | Output directory |
| `--samples` | `256` | Calibration samples (more = better accuracy) |
| `--max-len` | `512` | Max tokens per calibration sample |
| `--weight-only` | off | Use W4A16 instead of W4A4 |
| `--ignore` | `lm_head` | Layer names/regex patterns to exclude (use `re:` prefix for regex) |
| `--dtype` | `auto` | Model dtype: auto, bfloat16, float16 |
| `--trust-remote-code` | off | Trust remote code when loading model/tokenizer |
| `--dataset` | `HuggingFaceH4/ultrachat_200k` | HuggingFace dataset for calibration |

---

## 3. Serve and Chat

### Start the server

```bash
source .venv/bin/activate
python serve.py

# Limit VRAM usage (useful for small models or when sharing the GPU):
python serve.py --gpu-memory-utilization 0.3
```

By default vLLM reserves 90% of VRAM (~86 GB on this card) for the model and KV
cache combined. For a 0.5B model the model itself is only ~0.5 GB — the rest is KV
cache blocks pre-allocated to avoid fragmentation at runtime. Use
`--gpu-memory-utilization` to cap it.

### Chat interactively

In a second terminal:

```bash
source .venv/bin/activate
python chat.py
# or with options:
python chat.py --system "You are a concise assistant." --temperature 0.5
```

Commands inside chat: `/clear` resets history, `/system` prints the system prompt,
`/quit` exits.

---

## 4. Serve with vLLM (manual)

llm-compressor saves models in **compressed-tensors** format, which vLLM detects
automatically — no `--quantization` flag needed.

To confirm NVFP4 kernels are actually active (not silently falling back), look for
this line in the vLLM startup logs:
```
Using NvFp4LinearBackend.VLLM_CUTLASS for NVFP4 GEMM
```

### CLI

```bash
source .venv/bin/activate

vllm serve ./Qwen2.5-0.5B-Instruct-NVFP4 \
  --dtype auto \
  --max-model-len 8192
```

This starts an OpenAI-compatible server on `http://localhost:8000`.

### Quick smoke test

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen2.5-0.5B-Instruct-NVFP4",
    "messages": [{"role": "user", "content": "Hello! What are you?"}]
  }'
```

### Python API (no server required)

```python
from vllm import LLM, SamplingParams

llm = LLM(model="./Qwen2.5-0.5B-Instruct-NVFP4", dtype="auto")

outputs = llm.generate(
    ["What is NVFP4 quantization?"],
    SamplingParams(temperature=0.7, max_tokens=200),
)
print(outputs[0].outputs[0].text)
```

**Note:** `"EngineCore died unexpectedly"` printed during Python process exit is normal
cleanup — not an error. vLLM's engine runs in a subprocess and this message appears
when the parent process ends without an explicit shutdown.

### Skip quantization: use a pre-built NVIDIA checkpoint

NVIDIA publishes ready-to-use NVFP4 checkpoints on HuggingFace. These use ModelOpt
format and need `--quantization modelopt`:

```bash
# Examples
vllm serve nvidia/Llama-3.1-8B-Instruct-NVFP4   --quantization modelopt
vllm serve nvidia/Llama-3.3-70B-Instruct-FP4    --quantization modelopt
vllm serve nvidia/DeepSeek-R1-NVFP4             --quantization modelopt
```

---

## 5. Accuracy vs. Speed Trade-offs

| Approach | Accuracy | Throughput | Notes |
|----------|----------|------------|-------|
| W4A4 (`NVFP4`) | High | Best | Recommended default |
| W4A4 + GPTQ | Highest | Best | Slower to produce; use for 7B–14B |
| W4A16 (`NVFP4A16`) | Medium | Moderate | No calibration needed |
| `nvfp4_mlp_only` | Near-lossless | Good | Keep attention in BF16 (ModelOpt only) |

For models under ~7B, accuracy loss is more pronounced. Use GPTQ-based quantization
or `nvfp4_mlp_only` if quality matters.

---

## 6. Known Gotchas

### SM 12.0 (RTX Pro 6000 / RTX 5080 / RTX 6000 Pro)
Desktop Blackwell cards are SM 12.0 vs the data-center B200 at SM 10.0. vLLM has
historically had kernel-selection issues on SM 12.0 — verify in the vLLM startup logs
that NVFP4 kernels are loaded, not a fallback. Open issues in vllm-project/vllm:
[#30707](https://github.com/vllm-project/vllm/issues/30707),
[#31085](https://github.com/vllm-project/vllm/issues/31085).

**Confirmed working on RTX PRO 6000 Blackwell (SM 12.0) with vLLM 0.19.0** — CUTLASS
NVFP4 kernels load correctly with the `compressed-tensors` quantization path.

### Out of memory during quantization
If you hit OOM on large models:
```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python quantize.py --model <large-model> ...
```

### FlashInfer conflicts
If vLLM crashes on startup, try:
```bash
pip uninstall flashinfer-python
```

### Silent fallback kills performance
A misconfigured model can run in dequantization mode (loads weights as FP4,
immediately expands to BF16 for compute) — correct VRAM usage but no speedup.
Check logs for `NVFP4` kernel mentions to confirm.

---

## 7. References

- [NVIDIA: Introducing NVFP4](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/)
- [vLLM + llm-compressor NVFP4 docs](https://docs.vllm.ai/projects/llm-compressor/en/latest/examples/quantization_w4a4_fp4/)
- [llm-compressor GitHub](https://github.com/vllm-project/llm-compressor)
- [NVIDIA Model Optimizer GitHub](https://github.com/NVIDIA/Model-Optimizer)
- [vLLM ModelOpt integration](https://docs.vllm.ai/en/stable/features/quantization/modelopt/)
