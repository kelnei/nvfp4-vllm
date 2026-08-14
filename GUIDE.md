# NVFP4 Quantization with vLLM

End-to-end walkthrough: quantize a model to NVFP4 and serve it with vLLM.

**Hardware used:** NVIDIA RTX PRO 6000 Blackwell Workstation (SM 12.0, 96 GB VRAM)
**Confirmed working:** vLLM 0.27.1, torch 2.13.0+cu130, llmcompressor 0.12.0

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
(vLLM 0.27.1, llmcompressor 0.12.0, transformers 5.10.1). Two quirks: vLLM 0.27.1
pins `compressed-tensors==0.17.0` while llmcompressor 0.12.0 pins `==0.17.1`, and
vLLM's `torch==2.13.0` sits above llmcompressor's conservative `<=2.12.0` cap, so
the `[tool.uv]` section uses `override-dependencies` for both — without it the
packages cannot resolve together.

For hybrid linear-attention models (Qwen3.5/3.6/3.8), the optional
`fast-calib` extra installs `flash-linear-attention` and `causal-conv1d` so
Gated DeltaNet calibration doesn't run on transformers' slow torch fallback.
`causal-conv1d` builds from source and needs nvcc on PATH (see the CUDA
toolkit note below): `uv sync --extra fast-calib`.

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
2.13.0+cu130 True
0.27.1
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

This flag sets the corpus used for *calibration*. The `--fp8-mlp` sensitivity
ranking draws from `--sensitivity-dataset` instead, which defaults to
`ultrachat` for a measured reason — see
[Mixed-precision MLP](#mixed-precision-mlp).

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
fake-quantized — weights *and* input activations, the same two formats W4A4
serving uses — while every other layer stays at full precision, and the
model's output distribution is compared against the untouched model by
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

Four things worth knowing before trusting the ranking:

- **Per-layer KL does not add up.** Quantization errors in different layers
  interact, so the top N by marginal KL is a greedy pick, not a proven-optimal
  subset. The scan tells you where to spend, not exactly how much you'll get.
- **It cannot see GPTQ.** Both formats are measured with plain minmax. GPTQ
  then claws back part of the NVFP4 error on whatever stays at NVFP4, which
  shrinks the real gain relative to what the table shows — and on Qwen-family
  hybrids it does more than shrink it, it reorders the layers entirely (see
  the next section).
- **It costs forward passes.** Two per layer per sample, plus one reference
  pass. Lower `--sensitivity-samples` (default 64) to trade resolution between
  close-scoring layers for scan time.
- **The ranking corpus is not the calibration corpus.** This is why
  `--sensitivity-dataset` exists and defaults to `ultrachat` while `--dataset`
  defaults to `mix`. Ranking on the wide mixture instead gives back about a
  third of what the feature buys (31-39% across four measurements) — see below.

`--fp8-mlp` needs calibration data for the ranked modes (`scan`, `top:N`,
`gptq-loss[:N]`), so none of them combine with `--weight-only`; an explicit
layer list works in either mode.

#### Qwen-family hybrids: rank with `gptq-loss`, not the KL scan

On Qwen3.6/3.8-class hybrids the scan's picks measurably lose. Qwen3.8-27B is
the documented case: the scan promoted eight early/mid layers, and rebuilding
the checkpoint with the last eight (56–63, where `down_proj` input amax runs
300–717 against ~30 mid-stack) measured ~0.0016 lower emulated KL end to end
on identical evaluation sequences. An amax-ranked top-8 (54, 55, 58–63 — read
for free from the stored `input_global_scale`, amax = 2688 / scale) tied the
last-8 list exactly. gemma is the opposite: its sensitive layers are genuinely
local, unsloth's own picks there are early/mid and irregular, and the scan
finds them — the failure is Qwen-specific, not general.

Chasing this down ruled out every cheap explanation. `--sensitivity-context`
ships the experiments: `quantized` measures each layer's FP8-promotion
marginal against a baseline with *every* MLP layer fake-quantized (weights and
activations; the BF16 originals park in CPU RAM for the scan), and
`quantized-acts` quantizes activations only, isolating the activation-format
half of a promotion. Both still rank early/mid layers on top and put 54–63
near the bottom on Qwen3.8, and a third of in-context marginals come out
*negative* — promoting a single layer can worsen a fully-quantized model by
removing error cancellation. Whatever makes the late layers worth promoting in
a shipped checkpoint is not visible to any minmax fake-quant trial, isolated
or in context: it emerges through the GPTQ pipeline, whose sequential error
compensation reshapes where the residual error lives.

The metric that does recover the list is GPTQ's own proxy loss, shipped as
`--fp8-mlp gptq-loss[:N]`. For each MLP projection it accumulates the input
Hessian from clean forward passes over the sensitivity corpus, runs
llm-compressor's `quantize_weight` under both formats, and keeps the loss
that call returns — the Hessian-weighted output MSE left after GPTQ's
sequential compensation, i.e. the exact quantity GPTQ minimizes, measured at
the injection site *before* downstream layers and error cancellation absorb
it. On Qwen3.8-27B its top eight layers are exactly 56–63 — the
checkpoint-validated winner no KL trial finds — with 54 and 55 (the
amax-ranked complement) at ranks nine and ten. The NVFP4/FP8 loss ratio is
nearly constant across layers (~13x), so the ranking effectively measures
per-layer Hessian energy; the free amax read from `input_global_scale` is
its crude one-number approximation.

It is not a depth artifact, and gemma keeps its own answer. On E2B the
gptq-loss ranking picks early/mid layers (0–5 plus a 23–29 block), and an
A/B under the identical July recipe on identical evaluation sequences
measured `gptq-loss:12` at 0.0500 / 0.1168 KL (weight-only / emulated) —
better than an amax-ranked top-12 (0.0541 / 0.1242) and far better than
uniform NVFP4 (0.0648 / 0.1502), a statistical tie with unsloth — but still
behind the KL scan's picks (0.0455 / 0.0985). Where sensitivity is genuinely
local, the direct KL measurement stays the better ranker.

Practical rule: on Qwen-family hybrids use `--fp8-mlp gptq-loss:N` (or the
equivalent explicit late-layer list); on gemma-class models keep `top:N`.
The non-clean contexts stay available as diagnostics for the next
architecture that disagrees with its scan.

#### Why the scan uses a different corpus than calibration

`--dataset` defaults to `mix` for coverage: activation scales and GPTQ should
see code, math, tool calls, images and 18 languages, not just English chat.
`--sensitivity-dataset` defaults to `ultrachat` anyway, because ranking layers
on the wide mixture measurably picks worse layers.

Scanning gemma-4-E2B both ways produces two nearly disjoint pictures of the
same model. Ultrachat blames the middle of the stack; the mix blames the input
side, moving layer 2 from 35th (dead last, gain 0.00014) to 6th (0.02198):

```
ultrachat  top-12: [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 23, 24]
mix        top-12: [0, 1, 2, 4, 5, 6, 7, 8, 9, 11, 12, 13]     (Spearman ρ = 0.59)
```

Four gemma-4-E2B checkpoints, same recipe and ignore list throughout, measured
as KL against the BF16 original on 30 English instruction prompts and on 30
broad prompts (posed in ten languages, plus tool use, code and encyclopedic
prose). Lower is better; `wo` is weight-only, `em` keeps activation fake-quant:

| calibrated on | ranked on | english wo | english em | broad wo | broad em | size |
|---|---|---|---|---|---|---|
| ultrachat | — *(uniform NVFP4)* | 0.0648 | 0.1502 | 0.0668 | 0.1434 | 7367 MB |
| ultrachat | ultrachat | **0.0455** | **0.0985** | 0.0496 | 0.0983 | 7545 MB |
| mix | mix | 0.0574 | 0.1250 | 0.0520 | 0.1131 | 7509 MB |
| mix | ultrachat *(default)* | 0.0527 | 0.1101 | **0.0452** | **0.0964** | 7545 MB |

Rows 3 and 4 differ *only* in the ranking corpus, which is the clean test.
Ranking on ultrachat wins all four measurements, every confidence interval
excluding zero (paired by-prompt bootstrap: −0.0047 [−0.0071, −0.0024] and
−0.0149 [−0.0256, −0.0058] on english, −0.0068 [−0.0087, −0.0048] and −0.0167
[−0.0234, −0.0102] on broad). It wins on the broad set too — the one built
specifically to give the wide mixture its best case.

Rows 2 and 4 show the other half of the trade, and why `--dataset` still
defaults to `mix`: calibrating on the mixture costs a little English chat
fidelity and buys more back everywhere else. Row 4 is the best broad-domain
checkpoint of the four, and beats unsloth's own E2B on both broad measures
(−0.0074 [−0.0121, −0.0031] and −0.0111 [−0.0189, −0.0043]) while trailing it
slightly on English.

The obvious suspect is the 12.5% image turns pulling the input side up, but
that is not it: rescanning the mix with `--vision-samples 0` barely moves the
ranking (ρ = 0.948 against the mix scan, still 0.64 against ultrachat). The
text mixture itself does it.

The likely reason is that the scan and the metric look at different ends of the
model. A diverse corpus differs from chat mostly in its *inputs* — languages,
raw web text, code — so it stresses the layers nearest the embedding. But KL is
measured over *output* positions, and the model answers in much the same
assistant register whatever went in. Ranking layers by input diversity spends
the FP8 budget where the divergence is not being scored. So the corpus to rank
on is the one that looks like what the model emits, which for an instruct
checkpoint is chat — while calibration still gets the wider mixture, where
coverage is free.

Set them equal if you have a reason to (`--sensitivity-dataset mix`), and the
run builds one corpus instead of two.

### FP8 DeltaNet (hybrid linear-attention models)

Hybrid Qwen models (3.5/3.6/3.8) replace attention with a Gated DeltaNet
recurrence in three of every four layers. Those `linear_attn` projections
quantize to NVFP4 by default, and pay for it: on Qwen3.6-27B, DeltaNet at
NVFP4 costs about +0.0045 weight-only KL versus leaving it at higher
precision — roughly half the total quantization error of the checkpoint.

```bash
python quantize.py --model Qwen/Qwen3.6-27B --fp8-deltanet
```

`--fp8-deltanet` keeps `in_proj_qkv`, `in_proj_z`, and `out_proj` at FP8
(channel-wise weights, dynamic per-token activations — the same scheme as
`--fp8-attn`) for about +1.5 GB on a 27B. The tiny `in_proj_a`/`in_proj_b`
gating projections stay unquantized either way, and the flag is a no-op on
models without `linear_attn` modules. FP8's per-channel scales also ride
through vLLM's `in_proj_qkvz` fusion without the requantize-to-min-scale
step NVFP4's per-tensor global scales trigger at load.

### Per-layer embeddings (Gemma 4 E-series)

The E-series carries a second embedding table, `embed_tokens_per_layer`, that
supplies a per-layer vector for every vocab entry. It is enormous relative to
the rest of the model — 4.38 GB of a 7.34 GB E2B checkpoint, 5.25 GB on E4B —
and the NVFP4 recipe leaves it at BF16, because it is a lookup table rather
than a matmul.

`--int8-ple on` quantizes it to weight-only INT8 with one scale per vocab row:

| | E2B disk | E2B VRAM |
|---|---|---|
| BF16 table (default) | 7545 MB | 7.72 GiB |
| `--int8-ple on` | 5305 MB | 5.53 GiB |

The VRAM figure matters as much as the disk one. vLLM's
`CompressedTensorsEmbeddingWNA16Int` unpacks only the rows a batch actually
gathers, so the table stays packed in memory instead of being densified at
load. Nothing else in the recipe changes, and the checkpoint serves on the
usual `CutlassNvFp4LinearKernel`.

**It is off by default because it is a real trade.** Measured on gemma-4-E2B
with both arms derived from one checkpoint, so that exactly one tensor of 2324
differs and GPTQ's non-reproducibility cannot contaminate the comparison:

| | weight-only | emulated (W4A4) |
|---|---|---|
| english chat | -0.0001 `[-.0005,+.0003]` | +0.0003 `[-.0050,+.0054]` |
| multilingual / tools / code | -0.0001 `[-.0006,+.0003]` | **+0.0080** `[+.0022,+.0135]` |

Three cells are nulls. The fourth is not: under W4A4 the broad set pays about
8% relative KL (0.0993 -> 0.1073). That comparison is exact, not approximate —
re-running the identical checkpoint twice reproduces bit-identical per-token KL,
so the measurement has no noise floor to hide behind.

Two things worth knowing about that number:

- **A weight-error proxy will not find it.** Quantizing the table perturbs it by
  1.1% relative Frobenius error, and ablating it in isolation against a BF16
  model costs 0.0008 KL — under 2% of what the recipe already spends, on both
  prompt sets equally. The damage only appears once activations are also
  quantized, which an isolated ablation cannot see by construction. Measure the
  finished checkpoint in emulated mode or you will conclude this is free.
- **It explains a gap we had attributed elsewhere.** unsloth's E2B ships the
  same INT8 table, and our BF16-table checkpoint measured 0.0091 better than
  theirs on broad/emulated. Turning this flag on lands us at 0.10730 against
  their 0.10748 — essentially all of that advantage was the table.

So: take it when 2.2 GB matters more than non-English fidelity, or when serving
English chat, where it costs nothing measurable. Leave it off otherwise.

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
| `--fp8-deltanet` | off | Keep Gated DeltaNet projections at FP8 on hybrid models. See [FP8 DeltaNet](#fp8-deltanet-hybrid-linear-attention-models) |
| `--fp8-mlp` | `off` | Keep the most quantization-sensitive MLP layers at FP8: `top:N` / `scan` (KL trials), `gptq-loss[:N]` (GPTQ proxy loss — use on Qwen-family hybrids), or an explicit layer list. See [Mixed-precision MLP](#mixed-precision-mlp) |
| `--sensitivity-dataset` | `ultrachat` | Data the `--fp8-mlp` ranking is measured on. Defaults differently from `--dataset` deliberately — see [Mixed-precision MLP](#mixed-precision-mlp) |
| `--sensitivity-samples` | `64` | Calibration samples behind the `--fp8-mlp` ranking (KL trials or Hessian accumulation) |
| `--sensitivity-context` | `clean` | Model state during each `--fp8-mlp` trial: `clean` (one layer at a time), `quantized` / `quantized-acts` (promotion marginals against an all-quantized baseline — diagnostics; see [Mixed-precision MLP](#mixed-precision-mlp)) |
| `--sensitivity-report` | none | Write the full `--fp8-mlp` ranking to a path as JSON |
| `--int8-ple` | `off` | INT8 the Gemma-4 E-series per-layer embedding table: ~30% smaller, at a measured cost to non-English fidelity. See [Per-layer embeddings](#per-layer-embeddings-gemma-4-e-series) |
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

# MTP speculative decoding for checkpoints that ship an mtp.* head
# (Qwen3.5/3.6/3.8). Use the generic "mtp" method — vLLM 0.27 resolves it
# to the Qwen3_5MTP draft arch via model_type, and warns that spelling it
# "qwen3_5_mtp" is deprecated:
python serve.py --model ./Qwen3.6-27B-NVFP4 --speculative-config '{"method": "mtp", "num_speculative_tokens": 1}'
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
| `--attention-backend` | auto (`TRITON_ATTN` if nvcc missing) | Force the attention backend (e.g. `FLASHINFER`, `TRITON_ATTN`); FlashInfer JIT-compiles with nvcc at startup |
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
`FLASHINFER_CUTLASS`, which dies the same way during the startup profile run),
to the default top-k/top-p sampler, which also comes from FlashInfer, and —
since vLLM 0.27 — to the attention backend itself: on Blackwell the selector
prefers `FLASHINFER`, whose batch prefill/decode modules JIT-compile during
the first metadata build (the error surfaces mid-startup from
`flashinfer/jit/core.py`).
`serve.py` handles all four with one check: when nvcc is not found it
defaults to `--linear-backend cutlass`, `--moe-backend cutlass`, and
`--attention-backend TRITON_ATTN` (vLLM's built-in kernels, no JIT) and sets
`VLLM_USE_FLASHINFER_SAMPLER=0`; with the CUDA toolkit installed, vLLM
auto-selects freely, FlashInfer backends included.

One subtlety: with `--speculative-config`, vLLM deliberately does **not**
inherit `--attention-backend` for the draft model — it re-autoselects
(preferring FlashInfer) unless the speculative config JSON names an
`attention_backend` itself. The failure is nasty because the JIT compile is
deferred to the first request: startup completes, then the engine dies on
the first generation. serve.py injects
`"attention_backend": "TRITON_ATTN"` into the speculative config when nvcc
is missing.

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

### Hybrid linear-attention models (Qwen3.5/3.6/3.8)

Hybrid checkpoints interleave Gated DeltaNet `linear_attn` layers with
ordinary `self_attn` layers (3:1 on Qwen3.5-family). All five DeltaNet
projections are plain `nn.Linear`, but with `--gptq-mlp on` the recipe drops
the catch-all Linear group and neither surviving regex (self_attn FP8, `.mlp`
GPTQ) matches `linear_attn.*` — which silently left every DeltaNet projection
at BF16 (a 25% size penalty on Qwen3.6-27B). `quantize.py` now detects
`linear_attn` modules in the live tree and adds an explicit NVFP4 group for
`in_proj_qkv`/`in_proj_z`/`out_proj`, leaving the tiny `in_proj_a`/`in_proj_b`
decay/beta gating projections unquantized (unsloth's split). The split aligns
with vLLM's weight fusion — `in_proj_qkvz` fuses qkv+z, `in_proj_ba` fuses
b+a, and vLLM requires every component of a fused module to resolve to the
same scheme — so don't move one projection across the line without moving its
fusion partner.

Two serving notes for hybrids: the Mamba-style recurrent state is allocated
per sequence, so `--max-num-seqs` must be at or below the reported cache
block count (64 works on Qwen3.6-27B; the 1024 default does not), and the
checkpoint's `mtp.*` head enables MTP speculative decoding (see the
speculative decoding example above).

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
