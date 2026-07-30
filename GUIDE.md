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

In both modes `quantize.py` keeps the attention q/k/v/o projections at FP8 by
default (and, for W4A4, calibrates an FP8 KV cache scale) — FP8 attention is
~3.5x closer to the original weights than NVFP4 for a small size cost. Pass
`--no-fp8-attn` for uniform NVFP4.

For W4A4 on dense models, the MLP gate/up/down projections are additionally
quantized with GPTQ error compensation by default (`--gptq-mlp auto`) — a
calibration-only change with the same on-disk format and serving cost that
measures ~20% lower KL divergence vs the BF16 original than plain minmax.
MoE models are skipped automatically (per-expert calibration coverage is too
thin); pass `--gptq-mlp off` to disable.

Layers differ widely in how much accuracy NVFP4 costs them, so `--fp8-mlp` can
keep the most sensitive MLP layers at FP8 and leave the rest at NVFP4 — off by
default, since it trades size for fidelity. See
[Mixed-precision MLP](#mixed-precision-mlp).

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

### Calibration data

W4A4 quantization derives activation scales from whatever text you feed it, so
the calibration corpus is a real accuracy knob — not just a formality. The
default `--dataset mix` streams a weighted blend of seven public datasets:

| Share | Source | Contributes |
|-------|--------|-------------|
| 20% | [`HuggingFaceH4/ultrachat_200k`](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k) | multi-turn chat |
| 15% | [`allenai/tulu-3-sft-mixture`](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture) | general instruction following |
| 12% | [`open-r1/Mixture-of-Thoughts`](https://huggingface.co/datasets/open-r1/Mixture-of-Thoughts) (`code`) | code reasoning |
| 12% | `open-r1/Mixture-of-Thoughts` (`math`) | math reasoning |
| 15% | [`CohereLabs/aya_dataset`](https://huggingface.co/datasets/CohereLabs/aya_dataset) | 65-language prompts |
| 10% | [`HuggingFaceTB/smoltalk`](https://huggingface.co/datasets/HuggingFaceTB/smoltalk) (`apigen-80k`) | tool / function calls |
| 16% | [`eaddario/imatrix-calibration`](https://huggingface.co/datasets/eaddario/imatrix-calibration) (`text_all_small`) | raw 18-language web text |

Everything with turns is rendered through the model's **own chat template**, so
the template's control tokens appear in the activation statistics. This matters
for instruct models: a plain-text corpus never shows the model the format it
will actually be served in. The last row is deliberately *not* templated — it
is the component that exercises the model outside its instruct format.

Sources are read with `streaming=True`, so a run downloads only the rows it
samples — `tulu-3` alone is 1.4 GB on disk, and a 512-sample run pulls a few
tens of MB in total.

`--dataset ultrachat` restores the old single-source behaviour, and any
HuggingFace dataset ID still works as a single-source mixture (rows with a
`messages` column are chat-templated; `inputs`/`targets`, `content`, `text`,
and `article` columns are also recognised, and oversized rows are split into
`--max-len` chunks rather than truncated).

**Vision.** For a checkpoint with an image processor, 12.5% of the samples come
from [`unsloth/llava-instruct-mix-vsft-mini`](https://huggingface.co/datasets/unsloth/llava-instruct-mix-vsft-mini)
as real image+text turns, pushed through the model's own processor. The vision
tower stays unquantized, but its output embeddings are spliced into the
decoder's input sequence and their distribution is nothing like a text
embedding's — without them the decoder's NVFP4 input scales and FP8 KV scales
are calibrated on half the input distribution the model actually sees. Set
`--vision-samples 0` for text-only calibration, or an integer to override the
share. If the processor cannot render inline images the run warns and falls
back to text only rather than failing.

**Token budget.** `quantize.py` prints the real total after building the set:

```
Calibration set: 512 samples, 402,131 tokens (vision 64, chat 90, instruct 67, ...)
```

Samples are truncated at `--max-len`, and most chat rows are shorter than that,
so the total lands well under `samples × max-len`. Reach for `--samples 1024
--max-len 2048` (~1.1M tokens) if you want a larger budget, though on a 5B
model that made no measurable difference to KL against the BF16 original.

### Mixed-precision MLP

NVFP4 stores a weight in about 4.5 bits against FP8's 8, but layers do not all
pay the same accuracy price for that discount. `--fp8-mlp` keeps the layers
that pay the most at FP8 and leaves the rest at NVFP4, spending size where it
buys the most fidelity.

Start by looking at the ranking without committing to anything:

```bash
python quantize.py --model google/gemma-4-E2B-it --pipeline basic \
  --fp8-mlp scan --sensitivity-report e2b-sensitivity.json
```

```
rank  layer   NVFP4 KL    FP8 KL      gain    +MiB
   1     21    0.10793   0.00242   0.10551     5.5
   2      3    0.04713   0.00351   0.04362     5.5
   3      2    0.04173   0.00229   0.03943     5.5
...
  24      9    0.00460   0.00423   0.00037     5.5
```

Each row is a direct measurement, not a heuristic: one layer's MLP is
fake-quantized while every other weight in the model stays at full precision,
and the model's output distribution is compared against the untouched model by
mean KL over calibration tokens. `gain` is what promoting that layer to FP8
buys, and `+MiB` is what it costs on disk. The spread is large — two orders of
magnitude between the most and least sensitive layer is normal — which is the
whole reason a uniform layout leaves accuracy on the table.

Then commit, either by count or by naming layers outright:

```bash
python quantize.py --model google/gemma-4-E2B-it --pipeline basic --fp8-mlp top:12
python quantize.py --model google/gemma-4-E2B-it --pipeline basic --fp8-mlp 1,2,3,10-15,19,20,23
```

Promotion is always whole-layer. vLLM fuses `gate_proj` and `up_proj` into a
single `gate_up_proj` module and resolves its scheme from whichever shard name
its matcher reaches first, so a layer holding both precisions would load as one
of them arbitrarily. The promoted layers are written to their own config group
targeted by exact module name, and the NVFP4 group's targets are narrowed to
the complement — two disjoint lists, so no module can match both groups.

Three things worth knowing before trusting the ranking:

- **Per-layer KL does not add up.** Quantization errors in different layers
  interact, so the top N by marginal KL is a greedy pick, not a proven-optimal
  subset. The scan tells you where to spend, not exactly how much you'll get.
- **It cannot see GPTQ.** Both formats are measured with plain minmax. GPTQ
  then claws back part of the NVFP4 error on whatever stays at NVFP4, which
  shrinks the real gain relative to what the table shows.
- **It costs forward passes.** Two per layer per sample, plus one reference
  pass. Lower `--sensitivity-samples` (default 64) to trade resolution between
  close-scoring layers for scan time.
- **The ranking depends on `--dataset`.** Scanning gemma-4-E2B on the `mix`
  corpus and on ultrachat alone agrees on only 8 of the top 12 layers
  (Spearman ρ = 0.67 across all 35). The mix pushes the earliest layers far up
  the ranking — layer 0 goes from 18th on ultrachat to 1st — which is what you
  would expect when image embeddings and non-English text are spliced into the
  input, since that is where those distributions differ most from plain
  English chat. Scan on the data you intend to serve, and prefer the corpus you
  are going to calibrate with.

`--fp8-mlp` needs calibration data for `scan` and `top:N`, so neither combines
with `--weight-only`; an explicit layer list works in either mode.

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
| `--samples` | `512` | Calibration samples (more = better accuracy) |
| `--max-len` | `1024` | Max tokens per calibration sample |
| `--weight-only` | off | Use W4A16 instead of W4A4 |
| `--fp8-mlp` | `off` | Keep the most quantization-sensitive MLP layers at FP8. See [Mixed-precision MLP](#mixed-precision-mlp) |
| `--sensitivity-samples` | `64` | Calibration samples the `--fp8-mlp` ranking measures KL over |
| `--sensitivity-report` | none | Write the full `--fp8-mlp` ranking to a path as JSON |
| `--ignore` | `lm_head` | Layer names/regex patterns to exclude (use `re:` prefix for regex) |
| `--pipeline` | `auto` | Calibration pipeline: `auto`, `sequential`, or `basic`. See [Untraceable architectures](#untraceable-architectures) |
| `--dtype` | `auto` | Model dtype: auto, bfloat16, float16 |
| `--trust-remote-code` | off | Trust remote code when loading model/tokenizer |
| `--dataset` | `mix` | Calibration data: a named mixture or a HuggingFace dataset ID. See [Calibration data](#calibration-data) |
| `--vision-samples` | `auto` | Share of `--samples` that carry an image. See [Calibration data](#calibration-data) |
| `--split` | auto | Dataset split (`train_sft` for ultrachat, `train` otherwise); ignored for multi-source mixtures |
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

# Enable tool/function calling (parser must match the model family,
# e.g. gemma4 for Gemma 4 checkpoints, hermes for Hermes-style models):
python serve.py --tool-call-parser gemma4 --enable-auto-tool-choice

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
| `--moe-backend` | auto (`cutlass` if nvcc missing) | Force the fused-MoE kernel backend (e.g. `cutlass`, `marlin`, `flashinfer_cutlass`) |
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

### Tool calling

`chat.py --tools` offers the model a built-in `web_fetch` tool (fetches a URL
and returns the page's visible text, stdlib only). When the model requests a
tool, chat.py prints `[tool] web_fetch({...})`, runs it locally, feeds the
result back, and lets the model continue — up to 5 tool rounds per turn.

The server must be started with a tool parser matching the model family:

```bash
python serve.py --model ./gemma-4-26B-A4B-it-NVFP4 --trust-remote-code \
  --tool-call-parser gemma4 --enable-auto-tool-choice
```

```
You: Fetch https://example.com and tell me in one sentence what the page says.
Assistant:
[tool] web_fetch({"url": "https://example.com"})
Assistant: The page states that Example Domain is a domain intended for use in
documentation examples without needing permission.
```

Adding your own tool is three steps in chat.py: append an OpenAI-format spec
to `TOOL_SPECS`, add the implementation to `TOOL_IMPLS`, done — the loop
handles the rest.

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
The same applies to vLLM's fused-MoE NVFP4 backend (auto-selection prefers
`FLASHINFER_CUTLASS`, which dies the same way during the startup profile run)
and to the default top-k/top-p sampler, which also comes from FlashInfer.
`serve.py` handles all three with one check: when nvcc is not found it
defaults to `--linear-backend cutlass` and `--moe-backend cutlass` (vLLM's
built-in kernels, no JIT) and sets `VLLM_USE_FLASHINFER_SAMPLER=0`; with the
CUDA toolkit installed, vLLM auto-selects freely, FlashInfer backends
included.

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

### KV cache scales land on vision and audio towers
Symptom at `vllm serve` time, on a multimodal checkpoint quantized with FP8 KV
cache (the default):
```
ValueError: There is no module or parameter named
'audio_tower.layers.0.self_attn.attn' in Gemma4ForConditionalGeneration
```

`self_attn.attn` is where vLLM keeps a layer's KV cache scales, and it only
builds one for the decoder. The checkpoint has `k_scale`/`v_scale` saved for
the audio (or vision) tower's attention as well, and there is nowhere to put
them, so the load fails outright.

Cause: compressed-tensors applies `kv_cache_scheme` by walking *every*
attention module in the model and never consults the ignore list
(`_apply_kv_cache_scheme`, 0.17.1), so passing
`--ignore 're:model\.audio_tower\..*'` excludes the tower's Linear layers from
quantization but not its attention from KV scaling.

`quantize.py` now runs `strip_ignored_kv_scales()` before saving, dropping KV
scales from any attention module the ignore patterns cover. The scales were
inert — nothing reads them for a tower that is not KV-cached — so this changes
nothing but loadability. To repair an older checkpoint without requantizing,
delete the offending tensors and rewrite the shard:
```bash
python - <<'PY'
import re
from pathlib import Path
from safetensors.torch import load_file, save_file
path = Path("./your-model-NVFP4/model.safetensors")
drop = re.compile(r"^model\.(audio_tower|vision_tower)\..*\.(k_scale|v_scale)$")
tensors = load_file(path)
kept = {k: v for k, v in tensors.items() if not drop.match(k)}
print(f"removing {len(tensors) - len(kept)} tower KV scales")
save_file(kept, path, metadata={"format": "pt"})
PY
```
Sharded checkpoints additionally need the matching keys pruned from
`model.safetensors.index.json`.

### Untraceable architectures
Symptom during calibration:
```
Expected 64 subgraphs, traced 36
KeyError: 'sliding_attention'
```
llm-compressor's default pipeline traces the model's forward into per-layer
subgraphs so it can calibrate one layer at a time. Architectures that index
into runtime-built dicts (gemma-4 E-series looks up shared KV states by
attention type) break the tracer. Pass `--pipeline basic` to run plain
full-model forwards instead. Two consequences:

- With GPTQ, every target module's Hessian is live at once, so peak memory is
  much higher than sequential calibration.
- The imatrix gatherer can't share a calibration epoch with GPTQ under
  `basic` (GPTQ's observers claim the module accumulators before any data has
  flowed), so `quantize.py` runs a separate gathering pass first. Two
  `Applying/Gathering ...` phases in the log is expected there. Confirm the
  statistics survived by checking the log has no
  `Falling back to uniform MSE` warnings.

### Checkpoint heads transformers doesn't implement
Some checkpoints ship auxiliary weights that upstream transformers has no
module for — e.g. Qwen3.6's `mtp.*` multi-token-prediction head, which vLLM
uses for speculative decoding. transformers silently drops them on load, so
they would be missing from the quantized output and speculative decoding
would be unavailable. After saving, `quantize.py` copies any whole top-level
component that exists in the source checkpoint but in neither the live module
tree nor the saved output into an extra shard, and writes a
`model.safetensors.index.json` covering all shards (a lone
`model.safetensors` would otherwise be read on its own, ignoring the extra
file). `fix_ignore_list()` then adds the copied weights to
`quantization_config.ignore` so loaders don't expect packed tensors for them.

Relatedly, `quantize.py` loads through the class named in the checkpoint's
`config.json` `architectures` field rather than `AutoModelForCausalLM`, which
would drop the whole vision tower on multimodal checkpoints.

---

## 7. References

- [NVIDIA: Introducing NVFP4](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/)
- [vLLM + llm-compressor NVFP4 docs](https://docs.vllm.ai/projects/llm-compressor/en/latest/examples/quantization_w4a4_fp4/)
- [llm-compressor GitHub](https://github.com/vllm-project/llm-compressor)
- [NVIDIA Model Optimizer GitHub](https://github.com/NVIDIA/Model-Optimizer)
- [vLLM ModelOpt integration](https://docs.vllm.ai/en/stable/features/quantization/modelopt/)
