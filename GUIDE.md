# NVFP4 Quantization with vLLM

End-to-end walkthrough: quantize a model to NVFP4 and serve it with vLLM.

**Hardware used:** NVIDIA RTX PRO 6000 Blackwell Workstation (SM 12.0, 96 GB VRAM)
**Confirmed working:** vLLM 0.25.1, torch 2.11.0+cu130, llmcompressor 0.12.0

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

Dependencies are managed via `pyproject.toml` with exact pins for the core stack
(vLLM 0.25.1, llmcompressor 0.12.0, transformers 5.10.1). One quirk: vLLM 0.25.1
pins `compressed-tensors==0.17.0` while llmcompressor 0.12.0 pins `==0.17.1`, so the
`[tool.uv]` section uses `override-dependencies` to force the newer patch release —
without it the two packages cannot resolve together.

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
2.11.0+cu130 True
0.25.1
```

---

## 2. Quantize a Model

The script [`quantize.py`](quantize.py) handles both W4A4 and W4A16 modes.
For multimodal models, use [`inspect_layers.py`](inspect_layers.py) first to
find the right `--ignore` layers — see [Multimodal models](#multimodal-models-gemma-4-etc)
below.

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

Models with vision/audio components need those layers excluded from quantization,
and MoE variants additionally need their router layers excluded. Don't guess the
layer names — different Gemma 4 variants have used different module names for
their embedders (`vision_tower` vs `vision_embedder` vs `embed_vision`), and a
pattern that doesn't match means those layers get silently quantized instead of
skipped. Use [`inspect_layers.py`](inspect_layers.py) first — it builds the
model on a meta device (architecture only, no weights downloaded; only the
config is fetched) and works out what to ignore from the module structure:
standalone Linear layers outside the decoder stack, encoder-tower stacks, and
MoE routers (in-stack Linears whose `out_features == num_experts`):

```bash
python inspect_layers.py --model google/gemma-4-12B-it --trust-remote-code
```

```
328 Linear layers in the main decoder stack under 'model.language_model' (quantization targets), e.g.:
  model.language_model.layers.0.self_attn.q_proj
  ...

4 standalone Linear layers (--ignore candidates):
  lm_head
  model.embed_audio.embedding_projection
  model.embed_vision.multimodal_embedder.embedding_projection
  model.embed_vision.patch_dense

Suggested --ignore args:
lm_head model.embed_audio.embedding_projection model.embed_vision.multimodal_embedder.embedding_projection model.embed_vision.patch_dense
```

Feed the suggestion into `--ignore` — either paste the last line, or use
`--args-only`, which prints *only* the args on stdout (notes go to stderr) so
it is safe in shell substitution:

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python quantize.py \
  --model google/gemma-4-12B-it \
  --samples 512 \
  --max-len 2048 \
  --trust-remote-code \
  --ignore $(python inspect_layers.py --model google/gemma-4-12B-it --trust-remote-code --args-only)
```

This covers MoE variants too — for `google/gemma-4-26B-A4B-it` the tool emits
`re:model\.vision_tower\.encoder\..*` for the vision tower's own encoder stack
and `re:.*\.router\.proj$` for the 30 router layers, alongside the standalone
layers. Review its output (it prints a warning when it finds encoder towers)
before starting a long quantization run.

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
| `--split` | auto | Dataset split (`train_sft` for ultrachat, `train` otherwise) |
| `--cpu-offload` | off | Load model to system RAM; llm-compressor dispatches layers to GPU during calibration (use for large MoE models) |

---

## 3. Serve and Chat

### Start the server

```bash
source .venv/bin/activate
python serve.py

# Limit VRAM usage (useful for small models or when sharing the GPU):
python serve.py --gpu-memory-utilization 0.3

# Multi-GPU with tensor parallelism:
python serve.py --tensor-parallel-size 2

# FP8 KV cache for lower memory usage:
python serve.py --kv-cache-dtype fp8

# Enable tool/function calling (e.g. for Hermes-style models):
python serve.py --tool-call-parser hermes --enable-auto-tool-choice

# Speculative decoding:
python serve.py --speculative-config '{"draft_model": "org/small-model", "num_speculative_tokens": 5}'
```

By default vLLM reserves 90% of VRAM (~86 GB on this card) for the model and KV
cache combined. For a 0.5B model the model itself is only ~0.5 GB — the rest is KV
cache blocks pre-allocated to avoid fragmentation at runtime. Use
`--gpu-memory-utilization` to cap it.

### serve.py flags

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
| `--tensor-parallel-size`, `-tp` | `1` | Number of GPUs for tensor parallelism |
| `--pipeline-parallel-size`, `-pp` | `1` | Number of GPUs for pipeline parallelism |
| `--max-num-seqs` | vLLM default | Max concurrent sequences (batch size) |
| `--quantization` | auto | Force backend (e.g. `modelopt` for NVIDIA checkpoints) |
| `--kv-cache-dtype` | `auto` | KV cache dtype: auto, fp8, fp8_e5m2, fp8_e4m3 |
| `--linear-backend` | auto (`cutlass` if nvcc missing) | Force the GEMM kernel backend (e.g. `cutlass`, `marlin`, `flashinfer_cutlass`) |
| `--enforce-eager` | off | Disable CUDA graph compilation (useful for debugging) |
| `--enable-prefix-caching` | off | Enable KV cache reuse across requests with shared prefixes |
| `--speculative-config` | none | JSON string or file path for speculative decoding config |
| `--tool-call-parser` | none | Tool/function call parser (e.g. hermes, llama3_json, mistral) |
| `--enable-auto-tool-choice` | off | Let the model decide when to use tools |

Any flag not listed above is passed through to vLLM unchanged, so every
`vllm serve` option is available (e.g. `--swap-space 8`).

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
Using CutlassNvFp4LinearKernel for NVFP4 GEMM
```
A warning about the "emulation backend" means no optimized kernel loaded — the
model will run but without the FP4 speedup.

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

**Confirmed working on RTX PRO 6000 Blackwell (SM 12.0) with vLLM 0.25.1** — CUTLASS
NVFP4 kernels load correctly with the `compressed-tensors` quantization path.

### Out of memory during quantization
If you hit OOM on large models:
```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python quantize.py --model <large-model> ...
```

### FlashInfer JIT needs nvcc
With `--linear-backend auto`, vLLM may auto-select a FlashInfer NVFP4 kernel that
JIT-compiles CUDA code at startup. Without the CUDA toolkit installed, the engine
dies with:
```
RuntimeError: Could not find nvcc and default cuda_home='/usr/local/cuda' doesn't exist
```
The same applies to vLLM's default top-k/top-p sampler, which also comes from
FlashInfer. `serve.py` handles both with one check: when nvcc is not found it
defaults to `--linear-backend cutlass` (vLLM's built-in kernels, no JIT) and
sets `VLLM_USE_FLASHINFER_SAMPLER=0`; with the CUDA toolkit installed, vLLM
auto-selects freely, FlashInfer backends included.

### Silent fallback kills performance
A misconfigured model can run in dequantization mode (loads weights as FP4,
immediately expands to BF16 for compute) — correct VRAM usage but no speedup.
Check logs for `NVFP4` kernel mentions to confirm.

### llm-compressor's recorded ignore list can mismatch vLLM's module names
Symptom at `vllm serve` time:
```
ValueError: There is no module or parameter named '<module>.weight' in
<Model>ForConditionalGeneration. The available parameters belonging to
<module> (ColumnParallelLinear) are: {'weight_scale', 'weight_packed', ...}
```
This means the layer was correctly *excluded* from quantization (its saved
weight is plain, unquantized) but `config.json`'s `quantization_config.ignore`
lists it under the wrong name, so vLLM builds it as a quantized layer anyway
and can't find the packed weights it expects.

Cause: llm-compressor records `ignore` using the *live* transformers module
hierarchy at quantization time. Some architectures rename modules between
that live hierarchy and the on-disk checkpoint/vLLM naming via a checkpoint
conversion mapping (seen on Gemma 4: transformers calls it
`embed_vision.patch_dense`, the checkpoint and vLLM both call it
`vision_embedder.patch_dense`). The two names refer to the same tensor, but
`ignore` only has the transformers-side name, so vLLM's own name doesn't
match.

`quantize.py` now runs `fix_ignore_list()` after every save: it scans the
saved safetensors headers for 2D float weights that aren't in packed NVFP4
form and adds their actual on-disk tensor names to
`quantization_config.ignore`, which is guaranteed to match what vLLM loads.
The recorded transformers-side names are kept alongside, since transformers
matches against those when reloading the checkpoint itself. No manual
`config.json` editing should be needed. If you hit the error above on an
*older* checkpoint produced before this fix, re-run just the fix step:
```bash
python -c "from pathlib import Path; from quantize import fix_ignore_list; fix_ignore_list(Path('./your-model-NVFP4'))"
```

---

## 7. References

- [NVIDIA: Introducing NVFP4](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/)
- [vLLM + llm-compressor NVFP4 docs](https://docs.vllm.ai/projects/llm-compressor/en/latest/examples/quantization_w4a4_fp4/)
- [llm-compressor GitHub](https://github.com/vllm-project/llm-compressor)
- [NVIDIA Model Optimizer GitHub](https://github.com/NVIDIA/Model-Optimizer)
- [vLLM ModelOpt integration](https://docs.vllm.ai/en/stable/features/quantization/modelopt/)
