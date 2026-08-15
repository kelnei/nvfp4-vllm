"""
Quantize a model to NVFP4 (W4A4) using llm-compressor.

By default attention q/k/v/o projections are kept at FP8 (channel-wise
weights, dynamic per-token activations) and an FP8 KV cache scale is
calibrated — the mixed-precision layout vLLM's NVFP4 kernels expect. FP8
attention is ~3.5x closer to the original weights than NVFP4 for a small
size cost (~13% on a dense 12B, ~1% on a MoE).

On hybrid linear-attention models (Qwen3.5/3.6/3.8), the Gated DeltaNet
qkv/z/out projections get an explicit NVFP4 group of their own (FP8 with
--fp8-deltanet) and the tiny in_proj_a/in_proj_b gating projections stay
unquantized — automatic, keyed off the live module tree.

Usage:
    python quantize.py [--model MODEL_ID] [--output OUTPUT_DIR]
                       [--samples N] [--max-len N] [--weight-only]
                       [--no-fp8-attn] [--fp8-deltanet] [--fp8-lm-head]
                       [--gptq-mlp {auto,on,off}]
                       [--fp8-mlp SPEC] [--sensitivity-samples N]
                       [--sensitivity-report PATH]
                       [--sensitivity-context {clean,quantized}]
                       [--ignore PATTERN ...] [--dtype TYPE]
                       [--trust-remote-code] [--dataset DATASET]
                       [--split SPLIT] [--vision-samples N]

Defaults:
    model      = Qwen/Qwen2.5-0.5B-Instruct
    output     = <model-basename>-NVFP4
    samples    = 512
    max-len    = 1024
    weight-only = False (W4A4; set flag for W4A16)
    fp8-attn   = True (FP8 attention + FP8 KV cache; the KV cache scale
                 needs calibration, so it is skipped with --weight-only)
    fp8-deltanet = False (NVFP4 DeltaNet projections on hybrid models;
                 set the flag to keep them at FP8 instead — ~+1.5GB on a
                 27B for most of the DeltaNet quantization KL back)
    fp8-lm-head = False (output head stays at model dtype; set the flag to
                 quantize it to FP8 and take it off the ignore list — half
                 the head's bytes, worth +14% decode throughput at k=2 MTP
                 against 1.7 points of draft acceptance)
    gptq-mlp   = auto (GPTQ + imatrix_mse observer + static actorder on
                 dense MLP gate/up/down projections; calibration-only, same
                 on-disk format, ~20% lower KL vs BF16 than plain minmax.
                 auto enables it for dense models and skips it for MoE,
                 --weight-only, and --no-fp8-attn runs)
    fp8-mlp    = off (uniform NVFP4 across every dense MLP layer; 'top:N'
                 keeps the N most quantization-sensitive layers at FP8
                 instead, 'gptq-loss:N' ranks by GPTQ's own Hessian-weighted
                 proxy loss rather than per-layer KL trials, an explicit
                 list like '1,2,3,10-15' names them outright, and 'scan' or
                 'gptq-loss' just print the ranking. KL sensitivity is
                 measured W4A4-faithfully: each trial fake-quantizes the
                 layer's input activations along with its weights, the same
                 two formats serving uses. On Qwen-family hybrids the KL
                 trials do not predict the measured promotion value --
                 use gptq-loss there; see GUIDE.md)
    sensitivity-samples = 64 (calibration samples behind the --fp8-mlp rank)
    sensitivity-context = clean (each trial quantizes one layer against an
                 otherwise-BF16 model; 'quantized' scores each layer's FP8
                 promotion against an all-MLP-quantized baseline instead,
                 and 'quantized-acts' quantizes activations only -- both
                 diagnostics, not better default rankers)
    ignore     = lm_head (dropped automatically by --fp8-lm-head)
    dtype      = auto
    dataset    = mix (chat + instruct + code + math + multilingual + tool
                 calls + raw web text, streamed from seven HuggingFace
                 datasets and rendered through the model's chat template;
                 'ultrachat' selects the old single-source set, and any
                 HuggingFace dataset id still works)
    vision-samples = auto (12.5% of --samples carry an image when the
                 checkpoint has an image processor, 0 otherwise)
    split      = auto (train_sft for ultrachat, train otherwise; ignored for
                 multi-source mixtures)
"""

import argparse
import json
import random
from contextlib import contextmanager
from pathlib import Path
from typing import NamedTuple

import torch
from compressed_tensors.config import CompressionFormat
from compressed_tensors.quantization import (
    QuantizationArgs,
    preset_name_to_scheme,
)
from compressed_tensors.quantization.lifecycle.forward import fake_quantize
from compressed_tensors.utils import match_name
from compressed_tensors.utils.safetensors_load import (
    get_safetensors_header,
    get_weight_mappings,
)
from datasets import load_dataset
from llmcompressor import oneshot
from llmcompressor.modifiers.gptq import GPTQModifier
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.observers import Observer
from torch.utils.data import DataLoader

# llm-compressor 0.12.0 (current stable) only collects imatrix importance
# statistics when an IMatrixGatherer is prepended to the recipe; without it,
# observer=imatrix_mse silently falls back to a uniform MSE grid search.
# Nightlies (0.12.1a+) remove the class and make the observer self-collecting,
# so treat its absence as "no gatherer needed" rather than an error.
try:
    from llmcompressor.modifiers.transform.imatrix import IMatrixGatherer
except ImportError:
    IMatrixGatherer = None

import transformers
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    PreTrainedTokenizerBase,
)

if IMatrixGatherer is not None:

    class _PersistentIMatrixGatherer(IMatrixGatherer):
        """IMatrixGatherer whose collected statistics survive session finalize.

        With --pipeline basic the gatherer cannot share a calibration epoch
        with GPTQ (GPTQ's imatrix observers pick up — and delete — the
        module accumulators at epoch start, before any data has flowed), so
        the gathering runs as its own oneshot session first. The base class
        deletes the accumulators on finalize; this subclass leaves them on
        the modules for the next session's observers to pick up.
        """

        def on_finalize(self, state, **kwargs) -> bool:
            if not self.ended_:
                self.on_end(state, None)
            return True


# Attention projections kept at FP8 by --fp8-attn. Name-based, so fused-QKV
# and MLA architectures won't match; main() verifies the pattern hits at
# least one module before quantizing.
FP8_ATTN_TARGET = r"re:.*self_attn\.(q|k|v|o)_proj$"
# vLLM fuses q/k/v into a single qkv_proj module and resolves its scheme by
# layer name before class name, so the saved config must name the fused module
# too — otherwise the broad Linear/NVFP4 target wins and vLLM tries to load
# the FP8 shards as NVFP4. Matches nothing at quantization time, where the
# projections are still unfused.
FP8_ATTN_FUSED_TARGET = r"re:.*self_attn\.qkv_proj$"

# Dense MLP projections given the GPTQ + imatrix_mse + actorder treatment by
# --gptq-mlp. MoE expert projections (".mlp.experts.N.gate_proj") do not
# match, deliberately: per-expert calibration coverage is too thin for
# activation-statistics observers to be trustworthy there.
GPTQ_MLP_TARGET = r"re:.*\.mlp\.(gate|up|down)_proj$"

# Gated DeltaNet projections on hybrid linear-attention models (Qwen3.5/3.6/
# 3.8 `linear_attn` layers). All five are plain nn.Linear, so the catch-all
# Linear group covers them -- but --gptq-mlp on deletes that group, and
# neither surviving regex (self_attn FP8, .mlp GPTQ) matches linear_attn, so
# every DeltaNet projection silently stayed BF16 (the 25% size gap against
# unsloth's Qwen3.6-27B-NVFP4). Naming them here keeps them quantized in both
# recipe shapes. in_proj_a and in_proj_b produce the recurrence's decay and
# beta gating scalars and round to nothing beside the qkv/z/out projections;
# following unsloth's split they stay unquantized via the ignore list.
DELTANET_TARGET = r"re:.*linear_attn\.(in_proj_qkv|in_proj_z|out_proj)$"
DELTANET_IGNORE = (
    r"re:.*linear_attn\.in_proj_a$",
    r"re:.*linear_attn\.in_proj_b$",
)

# The output head, promoted from BF16 to FP8 by --fp8-lm-head. It is a plain
# Linear, so the only thing keeping it unquantized is the default ignore list;
# the flag drops it from there and gives it a group of its own.
LM_HEAD_TARGET = r"re:.*lm_head$"

# The Gemma-4 E-series per-layer embedding table, quantized to INT8 by
# --int8-ple. It is a lookup table, not a matmul, so this is weight-only; vLLM
# unpacks only the gathered rows (CompressedTensorsEmbeddingWNA16Int), which
# makes the saving VRAM as well as disk. The trailing "_per_layer" keeps this
# off the ordinary embed_tokens table, which is not a candidate: it is read
# through lm_head, which is on the ignore list unless --fp8-lm-head takes it
# off (and even then the table itself stays at model dtype).
PLE_TARGET = r"re:.*embed_tokens_per_layer$"


# ---------------------------------------------------------------------------
# Calibration data
# ---------------------------------------------------------------------------
#
# W4A4 fits activation scales to whatever text it is shown, so the corpus is
# an accuracy knob rather than a formality. `mix` spreads that exposure across
# what a served model actually sees: multi-turn chat, general instruction
# following, code and math reasoning, tool calls, 65-language prompts, and raw
# multilingual web text.
#
# Chat sources are rendered through the model's own chat template, which puts
# the template's control tokens into the activation statistics -- an instruct
# model calibrated on plain text never sees the format it is served in. The
# raw-text component is the deliberate exception.
#
# Every source is read with streaming=True, so a run pulls only the rows it
# samples rather than the whole repo (tulu-3 alone is 1.4 GB on disk).
#
# Note that on a 5B model, neither this mixture nor a 4.4x larger token budget
# moved KL against the BF16 original by more than measurement noise; the case
# for it is breadth of coverage, which an English-chat KL harness cannot see.


class MixSource(NamedTuple):
    """One component of a calibration mixture."""

    id: str
    role: str
    weight: float
    split: str = "train"
    config: str | None = None
    data_files: str | None = None


TEXT_MIX: tuple[MixSource, ...] = (
    MixSource("HuggingFaceH4/ultrachat_200k", "chat", 0.20, split="train_sft"),
    MixSource("allenai/tulu-3-sft-mixture", "instruct", 0.15),
    MixSource("open-r1/Mixture-of-Thoughts", "code", 0.12, config="code"),
    MixSource("open-r1/Mixture-of-Thoughts", "math", 0.12, config="math"),
    MixSource("CohereLabs/aya_dataset", "multilingual", 0.15),
    MixSource("HuggingFaceTB/smoltalk", "tools", 0.10, config="apigen-80k"),
    # Purpose-built imatrix calibration text: cleaned, de-duplicated FineWeb
    # across 18 languages. Not chat-templated, deliberately -- it is the one
    # component that exercises the model outside its instruct format.
    MixSource(
        "eaddario/imatrix-calibration",
        "raw-text",
        0.16,
        data_files="text_all_small.parquet",
    ),
)

# Image+text turns for multimodal checkpoints. The vision tower itself stays
# unquantized, but its output embeddings are spliced into the decoder's input
# sequence, and their distribution is nothing like a text embedding's -- so
# without image samples the decoder's NVFP4 input scales and FP8 KV scales are
# calibrated on half the input distribution the model actually sees.
VISION_MIX: tuple[MixSource, ...] = (
    MixSource("unsloth/llava-instruct-mix-vsft-mini", "vision", 1.0),
)

CALIBRATION_MIXES: dict[str, tuple[MixSource, ...]] = {
    "mix": TEXT_MIX,
    "ultrachat": (
        MixSource("HuggingFaceH4/ultrachat_200k", "chat", 1.0, split="train_sft"),
    ),
}

# Share of --samples given to image turns when --vision-samples is auto and
# the checkpoint has an image processor.
VISION_FRACTION = 0.125


def _allocate(sources: tuple[MixSource, ...], total: int) -> list[int]:
    """Split `total` samples across sources by weight (largest remainder)."""
    if total <= 0:
        return [0] * len(sources)
    scale = sum(s.weight for s in sources) or 1.0
    exact = [total * s.weight / scale for s in sources]
    quotas = [int(x) for x in exact]
    order = sorted(
        range(len(sources)), key=lambda i: exact[i] - quotas[i], reverse=True
    )
    for i in order[: total - sum(quotas)]:
        quotas[i] += 1
    return quotas


def _stream(src: MixSource, seed: int, buffer: int):
    kwargs = {}
    if src.config is not None:
        kwargs["name"] = src.config
    if src.data_files is not None:
        kwargs["data_files"] = src.data_files
    ds = load_dataset(src.id, split=src.split, streaming=True, **kwargs)
    return ds.shuffle(seed=seed, buffer_size=buffer)


def _row_to_messages(row: dict) -> list | None:
    """Normalise a row to chat turns, or None if it is unstructured text."""
    messages = row.get("messages")
    if isinstance(messages, list) and messages:
        return messages
    # CohereLabs/aya_dataset: single-turn prompt/completion columns
    if isinstance(row.get("inputs"), str) and isinstance(row.get("targets"), str):
        return [
            {"role": "user", "content": row["inputs"]},
            {"role": "assistant", "content": row["targets"]},
        ]
    return None


def _row_to_text(row: dict) -> str | None:
    for key in ("content", "text", "article"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return next(
        (v for v in row.values() if isinstance(v, str) and len(v.strip()) > 32), None
    )


def _render_chat(tokenizer, messages: list) -> str | None:
    """Apply the chat template, folding a system turn into the first user turn
    if the template rejects it (gemma's does)."""
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False)
    except Exception:
        pass

    folded: list[dict] = []
    carried = ""
    for message in messages:
        content = message.get("content")
        if not isinstance(content, str):
            return None
        if message.get("role") == "system":
            carried += content.strip() + "\n\n"
            continue
        if carried and message.get("role") == "user":
            message = {**message, "content": carried + content}
            carried = ""
        folded.append(message)
    if not folded:
        return None
    try:
        return tokenizer.apply_chat_template(folded, tokenize=False)
    except Exception:
        return None


def _chunk_text(text: str | None, max_len: int) -> list[str]:
    """Split a raw-text row into chunks of roughly `max_len` tokens each.

    Splits on line boundaries and budgets 3 characters per token, which
    under-fills for English and over-fills for CJK; the tokenizer truncates
    either way, so the only cost of being wrong is chunk size drift.
    """
    if not text:
        return []
    budget = 3 * max_len
    if len(text) <= budget:
        return [text]

    chunks: list[str] = []
    current: list[str] = []
    size = 0
    for line in text.splitlines(keepends=True):
        current.append(line)
        size += len(line)
        if size >= budget:
            chunks.append("".join(current))
            current, size = [], 0
    if size > budget // 4:
        chunks.append("".join(current))
    return chunks


def build_text_samples(
    sources: tuple[MixSource, ...],
    total: int,
    tokenizer,
    max_len: int,
    seed: int = 42,
) -> tuple[list[dict], dict[str, int]]:
    """Stream `total` tokenized samples from a weighted mixture of sources."""
    samples: list[dict] = []
    counts: dict[str, int] = {}

    for src, quota in zip(sources, _allocate(sources, total)):
        if quota <= 0:
            continue
        taken = 0
        # Cap the scan so a source that keeps failing to render can't spin.
        for scanned, row in enumerate(_stream(src, seed, max(512, quota * 4))):
            if taken >= quota or scanned >= quota * 20 + 64:
                break
            messages = _row_to_messages(row)
            if messages is not None:
                rendered = _render_chat(tokenizer, messages)
                texts = [rendered] if rendered else []
                add_special_tokens = False
            else:
                # Corpus rows can be whole documents (eaddario ships each of
                # its files as one multi-megabyte string), so split rather
                # than truncate away 99% of the row.
                texts = _chunk_text(_row_to_text(row), max_len)
                # Chunks come out in document order; sample across the whole
                # document instead of only its opening pages.
                random.Random(seed).shuffle(texts)
                add_special_tokens = True
            for text in texts:
                if taken >= quota:
                    break
                encoded = tokenizer(
                    text,
                    padding=False,
                    truncation=True,
                    max_length=max_len,
                    add_special_tokens=add_special_tokens,
                    return_tensors="pt",
                )
                if encoded["input_ids"].shape[-1] < 8:
                    continue
                samples.append(dict(encoded))
                taken += 1
        counts[src.role] = counts.get(src.role, 0) + taken
        if taken < quota:
            print(f"  warning: {src.id} yielded {taken}/{quota} usable samples")

    return samples, counts


def _vision_messages(row: dict) -> list | None:
    """Rewrite a llava-style row into chat turns with inline PIL images."""
    images = list(row.get("images") or [])
    turns: list[dict] = []
    for message in row.get("messages") or []:
        content = []
        for part in message.get("content") or []:
            if part.get("type") == "image":
                index = part.get("index") or 0
                if index >= len(images):
                    return None
                content.append({"type": "image", "image": images[index]})
            elif part.get("text"):
                content.append({"type": "text", "text": part["text"]})
        if content:
            turns.append({"role": message["role"], "content": content})
    return turns or None


def build_vision_samples(
    sources: tuple[MixSource, ...],
    total: int,
    processor,
    max_len: int,
    seed: int = 42,
) -> tuple[list[dict], dict[str, int]]:
    """Stream `total` image+text samples through the model's own processor.

    Returns empty if the processor cannot render inline images -- a text-only
    calibration set is a worse calibration set, not a failed run.
    """
    samples: list[dict] = []
    counts: dict[str, int] = {}

    for src, quota in zip(sources, _allocate(sources, total)):
        if quota <= 0:
            continue
        taken = 0
        for scanned, row in enumerate(_stream(src, seed, max(256, quota * 4))):
            if taken >= quota or scanned >= quota * 10 + 32:
                break
            messages = _vision_messages(row)
            if messages is None:
                continue
            try:
                encoded = processor.apply_chat_template(
                    messages,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                    add_generation_prompt=False,
                )
            except Exception as exc:
                print(
                    f"  warning: {type(processor).__name__} cannot render inline "
                    f"images ({exc}); calibrating on text only"
                )
                return [], {}
            # Image placeholders must survive intact, so these are filtered on
            # length rather than truncated.
            if encoded["input_ids"].shape[-1] > 4 * max_len:
                continue
            samples.append(dict(encoded))
            taken += 1
        counts[src.role] = counts.get(src.role, 0) + taken
        if taken < quota:
            print(f"  warning: {src.id} yielded {taken}/{quota} usable samples")

    return samples, counts


def _collate_single(batch: list[dict]) -> dict:
    # batch_size=1: text and image samples carry different keys, so there is
    # nothing to stack. The processor/tokenizer already produced a batch dim.
    assert len(batch) == 1
    return batch[0]


def build_calibration_loader(samples: list[dict], seed: int = 42) -> DataLoader:
    """Shuffle the merged mixture and wrap it for llm-compressor.

    oneshot() accepts a DataLoader directly, which is what lets one mixture
    hold text-only and image-bearing samples with different key sets.
    """
    shuffled = list(samples)
    random.Random(seed).shuffle(shuffled)
    return DataLoader(shuffled, batch_size=1, collate_fn=_collate_single)


def build_dataset(args, dataset: str, label: str, tokenizer, processor) -> DataLoader:
    """Assemble one named mixture (or bare HF dataset id) into a DataLoader.

    Called once for the calibration corpus and, when they differ, again for the
    --fp8-mlp ranking corpus.
    """
    print(f"Loading {label.lower()} dataset ({dataset})...")

    sources = CALIBRATION_MIXES.get(dataset)
    if sources is None:
        # Any HuggingFace dataset id still works as a single-source mix.
        # ultrachat names its split "train_sft"; most datasets use "train".
        default_split = (
            "train_sft" if dataset == "HuggingFaceH4/ultrachat_200k" else "train"
        )
        sources = (
            MixSource(dataset, "custom", 1.0, split=args.split or default_split),
        )
    elif args.split:
        if len(sources) == 1:
            sources = (sources[0]._replace(split=args.split),)
        else:
            print(f"  note: --split {args.split} ignored for mixture {dataset}")

    n_vision = args.vision_samples
    if n_vision == "auto":
        has_image_processor = processor is not None and (
            getattr(processor, "image_processor", None) is not None
        )
        n_vision = round(args.samples * VISION_FRACTION) if has_image_processor else 0
    elif n_vision > 0 and processor is None:
        raise SystemExit(
            "--vision-samples needs an image processor, but "
            "AutoProcessor.from_pretrained() returned none."
        )

    samples: list[dict] = []
    counts: dict[str, int] = {}
    if n_vision > 0:
        vision_samples, vision_counts = build_vision_samples(
            VISION_MIX, n_vision, processor, args.max_len
        )
        samples += vision_samples
        counts.update(vision_counts)
    text_samples, text_counts = build_text_samples(
        sources, args.samples - len(samples), tokenizer, args.max_len
    )
    samples += text_samples
    counts.update(text_counts)

    if not samples:
        raise SystemExit(f"Calibration mixture {dataset!r} yielded no samples")

    total_tokens = sum(s["input_ids"].shape[-1] for s in samples)
    breakdown = ", ".join(f"{role} {n}" for role, n in counts.items())
    print(f"{label} set: {len(samples)} samples, {total_tokens:,} tokens ({breakdown})")
    return build_calibration_loader(samples)


# ---------------------------------------------------------------------------
# Mixed-precision MLP selection
# ---------------------------------------------------------------------------
#
# NVFP4 stores a weight in ~4.5 bits against FP8's ~8, but layers do not all
# pay the same accuracy price for that. --fp8-mlp promotes the ones that pay
# the most back to FP8, buying fidelity with size where it goes furthest.
#
# Promotion is per decoder layer, never per projection: vLLM fuses gate_proj
# and up_proj into a single gate_up_proj and resolves its scheme from whichever
# shard name its matcher reaches first, so a layer holding both precisions
# would load as one of them arbitrarily.


class LayerSensitivity(NamedTuple):
    """What promoting one decoder layer's MLP from NVFP4 to FP8 would buy."""

    key: str  # module prefix, e.g. "model.language_model.layers.7"
    index: int  # decoder layer index
    nvfp4: float  # this layer's cost at NVFP4 (mean KL, or GPTQ proxy loss)
    fp8: float  # ... and at FP8, same metric
    modules: tuple[str, ...]
    extra_bytes: int  # on-disk cost of the promotion

    @property
    def gain(self) -> float:
        """Output divergence removed by promoting this layer."""
        return self.nvfp4 - self.fp8


def _layer_key(name: str) -> str:
    """Module prefix shared by one layer's MLP projections."""
    return name.rsplit(".mlp.", 1)[0]


def _layer_index(key: str) -> int:
    """Decoder index from a layer prefix; -1 when the name has no `.layers.N`."""
    _, sep, tail = key.rpartition(".layers.")
    return int(tail) if sep and tail.isdigit() else -1


def parse_layer_spec(spec: str) -> set[int]:
    """Parse a layer selection like "1,2,3,10-15,19" into a set of indices."""
    indices: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        low, sep, high = part.partition("-")
        try:
            indices.update(range(int(low), int(high) + 1) if sep else [int(part)])
        except ValueError:
            raise SystemExit(
                f"--fp8-mlp: cannot parse {part!r} in layer spec {spec!r}. Expected "
                "'off', 'scan', 'top:N', 'gptq-loss[:N]', or indices like "
                "'1,2,3,10-15'."
            ) from None
    return indices


def _stored_bytes(weight, args: QuantizationArgs) -> int:
    """On-disk size of one weight under `args`, scales included."""
    numel, rows = weight.numel(), weight.shape[0]
    if args.num_bits == 8:
        # One byte per weight plus an fp32 scale per output channel.
        return numel + 4 * rows
    # Four bits per weight, one fp8 group scale per `group_size`, one global.
    return numel // 2 + numel // (args.group_size or 16) + 4


def _fake_quant_weight(weight, args: QuantizationArgs, chunk: int = 4096):
    """Quantize-dequantize `weight` under `args`, same dtype and device.

    Quantization parameters come from the whole tensor -- NVFP4's global scale
    is tensor-wide -- and only the fake-quant is chunked over output rows,
    which every strategy here computes row-locally. Chunking keeps a 12B
    model's largest projection from needing an fp32 copy of itself.
    """
    observer = Observer.load_from_registry(args.observer, base_name="weight", args=args)
    observer(weight)
    qparams = observer.get_qparams()
    scale, zero_point = qparams["scale"], qparams["zero_point"]
    global_scale = qparams.get("global_scale")

    out = torch.empty_like(weight)
    for start in range(0, weight.shape[0], chunk):
        stop = start + chunk
        out[start:stop] = fake_quantize(
            weight[start:stop],
            scale[start:stop],
            None if zero_point is None else zero_point[start:stop],
            args,
            global_scale=global_scale,
        ).to(weight.dtype)
    return out


@contextmanager
def _swapped_weights(modules: dict, args: QuantizationArgs):
    """Temporarily replace each module's weight with its fake-quantized self."""
    originals = {name: module.weight.data for name, module in modules.items()}
    try:
        for name, module in modules.items():
            module.weight.data = _fake_quant_weight(originals[name], args)
        yield
    finally:
        for name, module in modules.items():
            module.weight.data = originals[name]


# FP4 E2M1 representable magnitudes and the midpoints between them, for
# round-to-nearest. Kept on CPU; moved to the activation's device on use.
_FP4_LEVELS = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
_FP4_MIDPOINTS = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])


def _fake_quant_nvfp4_activation(x, global_scale: float):
    """Quantize-dequantize one activation tensor the way NVFP4 W4A4 serving
    does: group-16 FP4 with per-group FP8-E4M3 scales computed per token, under
    a static per-tensor global scale (448*6/amax) fixed at calibration.

    The static global scale is the part that hurts on outlier-heavy layers:
    one huge calibration amax shrinks it for every token thereafter, pushing
    ordinary groups' scales toward the bottom of the FP8 range.
    """
    hidden = x.shape[-1]
    pad = (-hidden) % 16
    v = torch.nn.functional.pad(x, (0, pad)) if pad else x
    groups = v.float().reshape(-1, (hidden + pad) // 16, 16)

    group_amax = groups.abs().amax(-1, keepdim=True)
    group_scale = (group_amax * (global_scale / 6.0)).to(torch.float8_e4m3fn).float()
    group_scale = torch.where(
        group_scale == 0, torch.ones_like(group_scale), group_scale
    )

    scaled = (groups * (global_scale / group_scale)).clamp(-6.0, 6.0)
    idx = torch.bucketize(scaled.abs(), _FP4_MIDPOINTS.to(scaled.device))
    quant = scaled.sign() * _FP4_LEVELS.to(scaled.device)[idx]

    out = (quant * (group_scale / global_scale)).reshape(*x.shape[:-1], -1)
    return (out[..., :hidden] if pad else out).to(x.dtype)


def _fake_quant_fp8_activation(x):
    """Quantize-dequantize one activation tensor as FP8 serving does: dynamic
    per-token E4M3 scales, so outlier tokens cannot poison anyone else's."""
    scale = x.abs().amax(-1, keepdim=True).float().clamp(min=1e-12) / 448.0
    quant = (x.float() / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    return (quant.float() * scale).to(x.dtype)


@contextmanager
def _recorded_input_amax(modules: dict, store: dict):
    """Record each module's largest input magnitude seen while active -- the
    scan's stand-in for the calibration amax behind input_global_scale."""
    handles = []

    def recorder(name):
        def hook(_module, inputs):
            amax = inputs[0].abs().amax().item()
            store[name] = max(store.get(name, 0.0), amax)

        return hook

    try:
        for name, module in modules.items():
            handles.append(module.register_forward_pre_hook(recorder(name)))
        yield
    finally:
        for handle in handles:
            handle.remove()


def _activation_pre_hook(fmt: str, global_scale: float = 0.0):
    """Forward-pre-hook that fake-quantizes a module's input as `fmt`."""
    if fmt == "nvfp4":

        def hook(_module, inputs):
            return (_fake_quant_nvfp4_activation(inputs[0], global_scale),) + tuple(
                inputs[1:]
            )

    else:

        def hook(_module, inputs):
            return (_fake_quant_fp8_activation(inputs[0]),) + tuple(inputs[1:])

    return hook


@contextmanager
def _quantized_activations(modules: dict, fmt: str, amax_by_name: dict):
    """Temporarily fake-quantize each module's input activations as `fmt`."""
    handles = []
    try:
        for name, module in modules.items():
            if fmt == "nvfp4":
                amax = amax_by_name.get(name, 0.0)
                if amax <= 0:
                    continue
                hook = _activation_pre_hook("nvfp4", 448.0 * 6.0 / amax)
            else:
                hook = _activation_pre_hook("fp8")
            handles.append(module.register_forward_pre_hook(hook))
        yield
    finally:
        for handle in handles:
            handle.remove()


@torch.no_grad()
def scan_mlp_sensitivity(
    model,
    loader: DataLoader,
    module_names: list[str],
    nvfp4_args: QuantizationArgs,
    fp8_args: QuantizationArgs,
    n_samples: int,
    keep_positions: int = 64,
    context: str = "clean",
) -> list[LayerSensitivity]:
    """Rank decoder layers by what each format costs the model's own output.

    One layer's MLP at a time is fake-quantized -- weights AND input
    activations, matching W4A4 serving -- while every other layer stays at
    full precision, and the model's output distribution is compared against
    the untouched model by mean KL over calibration tokens. That is a direct
    measurement in the same currency the checkpoint is finally judged in.

    With context='quantized', the control flips: every scanned layer's MLP
    is fake-quantized at once -- weights (BF16 originals parked in CPU
    memory, roughly their on-disk size in host RAM) and input activations
    -- and each trial promotes one layer back to FP8 inside that context,
    scoring it by the whole-model KL it removes. The isolated scan asks
    "how much damage does quantizing this layer do to a clean model?"; the
    in-context scan asks "how much does promoting this layer repair a
    quantized one?". context='quantized-acts' leaves weights BF16
    everywhere and quantizes activations only, each trial toggling one
    layer's hook to dynamic FP8, isolating the activation-format half of a
    promotion.

    Treat the non-clean contexts as diagnostics, not better rankers. They
    were built to explain why promoting Qwen3.8's outlier-heavy last eight
    layers (down_proj input amax 300-717 against ~30 mid-stack) measures
    ~0.0016 lower emulated KL end to end than promoting this scan's picks
    -- and measured on Qwen3.8, both still rank early/mid layers on top
    and put the late outlier region near the bottom. So the late-layer
    advantage is invisible to every minmax fake-quant trial, isolated or
    in-context, weights or activations: it emerges only through the full
    GPTQ pipeline, whose sequential error compensation reshapes where a
    shipped checkpoint's residual error actually lives. On Qwen-family
    hybrids, use --fp8-mlp gptq-loss (rank_mlp_gptq_loss below), which
    measures error where GPTQ does and recovers the winning list exactly.
    The in-context marginals also interact strongly -- a third of
    layers measure negative, i.e. promoting them alone makes the
    fully-quantized model worse through lost error cancellation -- which
    is itself evidence that single-layer marginals in a high-error context
    do not predict small differences between shipped checkpoints.

    Activations are part of every trial because promotion changes their
    format too: NVFP4 quantizes them group-16 under a static per-tensor
    global scale, FP8 with dynamic per-token scales. The scan's earlier
    proxies measured strictly worse: a weight-only trial has nothing to
    say about activation outliers at all, and activation-weighted weight
    reconstruction error could not separate gemma-4-E2B's layers (all
    within 11% of each other, because relative weight error is mostly a
    property of how Gaussian a matrix is).

    Three caveats on the ranking. Per-layer KL measured in isolation does not
    sum to whole-model KL, since quantization errors in different layers
    interact, so taking the top N is a greedy heuristic rather than an optimal
    subset. The NVFP4 trial uses plain minmax weights: GPTQ later claws back
    part of the NVFP4 error on whatever stays at NVFP4, which this cannot
    model. And the ranking is a property of the data it is measured on, which
    is why --sensitivity-dataset exists and defaults to chat rather than to
    the wider calibration mixture -- see GUIDE.md for the measurement behind
    that.
    """
    modules = dict(model.named_modules())
    layers: dict[str, list[str]] = {}
    for name in module_names:
        if name in modules:
            layers.setdefault(_layer_key(name), []).append(name)
    if not layers:
        return []

    device = next(model.parameters()).device
    batches = []
    for i, batch in enumerate(loader):
        if i >= n_samples:
            break
        batches.append(
            {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
        )
    if not batches:
        return []

    def logprobs(batch):
        """Log-probs at evenly spaced positions -- the full sequence's worth of
        vocab logits would be tens of GB to hold as a reference."""
        logits = model(**batch).logits[0]
        keep = torch.linspace(
            0,
            logits.shape[0] - 1,
            min(keep_positions, logits.shape[0]),
            device=logits.device,
        ).long()
        return logits[keep].float().log_softmax(-1)

    def mean_kl(reference) -> float:
        total = count = 0
        for batch, ref in zip(batches, reference):
            ref = ref.to(device).float()
            kl = (ref.exp() * (ref - logprobs(batch))).sum(-1)
            total += kl.sum().item()
            count += kl.numel()
        return total / count

    what = (
        "activations only" if context == "quantized-acts" else "weights + activations"
    )
    print(
        f"Scanning MLP sensitivity: {len(layers)} layers x "
        f"{len(batches)} samples ({context} context, {what})"
    )
    scanned = {name: modules[name] for names in layers.values() for name in names}
    input_amax: dict[str, float] = {}
    with _recorded_input_amax(scanned, input_amax):
        reference = [logprobs(batch).half().cpu() for batch in batches]

    def extra_bytes(names) -> int:
        return sum(
            _stored_bytes(modules[name].weight, fp8_args)
            - _stored_bytes(modules[name].weight, nvfp4_args)
            for name in names
        )

    ranking = []
    if context != "clean":
        ranking = _scan_in_context(
            modules,
            layers,
            scanned,
            input_amax,
            nvfp4_args,
            fp8_args,
            mean_kl,
            reference,
            extra_bytes,
            device,
            quantize_weights=context == "quantized",
        )
    else:
        for position, (key, names) in enumerate(sorted(layers.items()), start=1):
            group = {name: modules[name] for name in names}
            with (
                _swapped_weights(group, nvfp4_args),
                _quantized_activations(group, "nvfp4", input_amax),
            ):
                kl_nvfp4 = mean_kl(reference)
            with (
                _swapped_weights(group, fp8_args),
                _quantized_activations(group, "fp8", input_amax),
            ):
                kl_fp8 = mean_kl(reference)
            ranking.append(
                LayerSensitivity(
                    key=key,
                    index=_layer_index(key),
                    nvfp4=kl_nvfp4,
                    fp8=kl_fp8,
                    modules=tuple(sorted(names)),
                    extra_bytes=extra_bytes(names),
                )
            )
            print(
                f"  [{position}/{len(layers)}] layer {ranking[-1].index}: "
                f"NVFP4 {kl_nvfp4:.5f}  FP8 {kl_fp8:.5f}"
            )

    ranking.sort(key=lambda layer: layer.gain, reverse=True)
    return ranking


def _scan_in_context(
    modules: dict,
    layers: dict[str, list[str]],
    scanned: dict,
    input_amax: dict[str, float],
    nvfp4_args: QuantizationArgs,
    fp8_args: QuantizationArgs,
    mean_kl,
    reference,
    extra_bytes,
    device,
    quantize_weights: bool = True,
) -> list[LayerSensitivity]:
    """The in-context trials behind scan_mlp_sensitivity(context != 'clean').

    Baseline first: every scanned module's weight is replaced with its NVFP4
    fake-quantized self (BF16 original parked on CPU) and gets an NVFP4
    input-activation hook, and the whole-model KL of that state is measured.
    Each trial then lifts one layer to FP8 -- weights re-fake-quantized from
    the parked original, hook swapped to dynamic per-token FP8 -- remeasures,
    and restores. A layer's LayerSensitivity carries the shared baseline as
    `nvfp4` and its promoted KL as `fp8`, so `gain` is the promotion marginal
    and the ranking sorts exactly as the isolated scan's does.

    Without `quantize_weights` (the 'quantized-acts' context), weights are
    untouched everywhere and the trials only toggle the activation hook, so
    the marginal isolates the activation-format half of a promotion.

    The NVFP4 activation hooks reuse the amax recorded on the clean reference
    pass. Calibration would instead see amax under quantized-upstream drift,
    but the drift is small in scale terms and a static global scale off by a
    few percent changes little next to 4-bit group rounding.
    """
    handles: dict[str, object] = {}

    def install_nvfp4_hook(name):
        amax = input_amax.get(name, 0.0)
        if amax <= 0:
            return  # module never reached by the data; nothing to quantize
        handles[name] = modules[name].register_forward_pre_hook(
            _activation_pre_hook("nvfp4", 448.0 * 6.0 / amax)
        )

    originals: dict[str, torch.Tensor] = {}
    ranking = []
    try:
        for name, module in scanned.items():
            if quantize_weights:
                originals[name] = module.weight.data.to("cpu")
                module.weight.data = _fake_quant_weight(module.weight.data, nvfp4_args)
            install_nvfp4_hook(name)
        baseline = mean_kl(reference)
        print(f"  all-NVFP4 baseline: KL {baseline:.5f}")

        for position, (key, names) in enumerate(sorted(layers.items()), start=1):
            held = {}
            for name in names:
                module = modules[name]
                if quantize_weights:
                    held[name] = module.weight.data
                    module.weight.data = _fake_quant_weight(
                        originals[name].to(device), fp8_args
                    )
                if name in handles:
                    handles.pop(name).remove()
                handles[name] = module.register_forward_pre_hook(
                    _activation_pre_hook("fp8")
                )
            kl_promoted = mean_kl(reference)
            for name in names:
                if quantize_weights:
                    modules[name].weight.data = held[name]
                handles.pop(name).remove()
                install_nvfp4_hook(name)
            ranking.append(
                LayerSensitivity(
                    key=key,
                    index=_layer_index(key),
                    nvfp4=baseline,
                    fp8=kl_promoted,
                    modules=tuple(sorted(names)),
                    extra_bytes=extra_bytes(names),
                )
            )
            print(
                f"  [{position}/{len(layers)}] layer {ranking[-1].index}: "
                f"promoted KL {kl_promoted:.5f}  "
                f"marginal {baseline - kl_promoted:+.5f}"
            )
    finally:
        for handle in handles.values():
            handle.remove()
        for name, original in originals.items():
            modules[name].weight.data = original.to(device)
    return ranking


@torch.no_grad()
def rank_mlp_gptq_loss(
    model,
    loader: DataLoader,
    module_names: list[str],
    nvfp4_scheme,
    fp8_scheme,
    n_samples: int,
    chunk_layers: int = 8,
) -> list[LayerSensitivity]:
    """Rank decoder layers by GPTQ's own per-module proxy loss.

    For each MLP projection, accumulate the input Hessian from clean-model
    calibration passes and run llm-compressor's quantize_weight under both
    formats, keeping the loss it returns: the Hessian-weighted output MSE
    left after GPTQ's sequential error compensation, i.e. the exact
    quantity GPTQ minimizes. A layer's score is the sum over its
    projections; the NVFP4-minus-FP8 difference is the predicted promotion
    benefit.

    This is the ranker that works where the KL trials above fail. On
    Qwen3.8-27B its top eight layers are exactly the measured-best 56-63
    promotion list, with 54 and 55 -- the amax-8 complement -- at ranks
    nine and ten; no KL trial in scan_mlp_sensitivity recovers that list in
    any context. It is not a depth artifact: on gemma-4-E2B the same metric
    ranks early/mid layers on top, matching that model's known sensitivity
    character. The NVFP4/FP8 loss ratio is nearly constant across layers
    (~13x), so the ranking effectively measures per-layer Hessian energy --
    the quantity output-KL toggles cannot see, because downstream layers
    and GPTQ's own compensation absorb local error before it reaches the
    output.

    Hessians come from the model's own clean forward passes over `loader`,
    one pass per chunk of `chunk_layers` layers to bound Hessian memory.
    --sensitivity-context does not apply here.
    """
    from llmcompressor.modifiers.gptq.gptq_quantize import (
        accumulate_hessian,
        make_empty_hessian,
        quantize_weight,
    )
    from llmcompressor.modifiers.quantization.calibration import initialize_observer

    modules = dict(model.named_modules())
    layers: dict[str, list[str]] = {}
    for name in module_names:
        if name in modules:
            layers.setdefault(_layer_key(name), []).append(name)
    if not layers:
        return []

    device = next(model.parameters()).device
    batches = []
    for i, batch in enumerate(loader):
        if i >= n_samples:
            break
        batches.append(
            {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
        )
    if not batches:
        return []

    ordered = sorted(layers, key=_layer_index)
    print(
        f"Ranking MLP layers by GPTQ proxy loss: {len(ordered)} layers x "
        f"{len(batches)} samples, {chunk_layers} layers per pass"
    )

    losses: dict[str, dict[str, float]] = {}
    for start in range(0, len(ordered), chunk_layers):
        chunk = ordered[start : start + chunk_layers]
        names = [name for key in chunk for name in layers[key]]

        hessians: dict[str, torch.Tensor] = {}
        counts: dict[str, torch.Tensor] = {}
        handles = []

        def make_hook(name, module):
            def hook(_module, inputs):
                inp = inputs[0].detach()
                if name not in hessians:
                    hessians[name] = make_empty_hessian(module, device=inp.device)
                    counts[name] = torch.zeros((), device=inp.device)
                hessians[name], counts[name] = accumulate_hessian(
                    inp, module, hessians[name], counts[name]
                )

            return hook

        for name in names:
            handles.append(
                modules[name].register_forward_pre_hook(make_hook(name, modules[name]))
            )
        try:
            for batch in batches:
                model(**batch)
        finally:
            for handle in handles:
                handle.remove()

        for name in names:
            if name not in hessians:
                continue
            module = modules[name]
            hessian = hessians.pop(name) / counts.pop(name)
            entry = {}
            for label, scheme in (("nvfp4", nvfp4_scheme), ("fp8", fp8_scheme)):
                module.quantization_scheme = scheme
                initialize_observer(module, "weight")
                module.weight_observer(module.weight)
                # quantize_weight consumes the Hessian in place; clone for
                # the first format so the second sees the real one.
                h = hessian.clone() if label == "nvfp4" else hessian
                loss, _ = quantize_weight(
                    module=module,
                    quant_args=scheme.weights,
                    hessian=h,
                    blocksize=128,
                    percdamp=0.01,
                )
                entry[label] = loss
                del module.weight_observer
                del module.quantization_scheme
            losses[name] = entry
        for position, key in enumerate(chunk, start=start + 1):
            done = [n for n in layers[key] if n in losses]
            print(
                f"  [{position}/{len(ordered)}] layer {_layer_index(key)}: "
                f"NVFP4 {sum(losses[n]['nvfp4'] for n in done):.4g}  "
                f"FP8 {sum(losses[n]['fp8'] for n in done):.4g}"
            )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    ranking = []
    for key in ordered:
        done = [n for n in layers[key] if n in losses]
        if not done:
            continue
        ranking.append(
            LayerSensitivity(
                key=key,
                index=_layer_index(key),
                nvfp4=sum(losses[n]["nvfp4"] for n in done),
                fp8=sum(losses[n]["fp8"] for n in done),
                modules=tuple(sorted(done)),
                extra_bytes=sum(
                    _stored_bytes(modules[n].weight, fp8_scheme.weights)
                    - _stored_bytes(modules[n].weight, nvfp4_scheme.weights)
                    for n in done
                ),
            )
        )
    ranking.sort(key=lambda layer: layer.gain, reverse=True)
    return ranking


def print_sensitivity_table(
    ranking: list[LayerSensitivity],
    promoted: set[str] | None = None,
    metric: str = "KL",
) -> None:
    """Print the ranking most-sensitive first, marking the promoted layers."""
    promoted = promoted or set()
    # KL values live in a narrow decade; GPTQ losses span several, so they
    # get significant digits instead of fixed decimals.
    spec = ".5f" if metric == "KL" else ".4g"
    print(
        f"\n{'rank':>4} {'layer':>6} {f'NVFP4 {metric}':>10} {f'FP8 {metric}':>9} "
        f"{'gain':>9} {'+MiB':>7}"
    )
    for rank, layer in enumerate(ranking, start=1):
        mark = " <- FP8" if layer.key in promoted else ""
        print(
            f"{rank:>4} {layer.index:>6} {layer.nvfp4:>10{spec}} "
            f"{layer.fp8:>9{spec}} {layer.gain:>9{spec}} "
            f"{layer.extra_bytes / 1024**2:>7.1f}{mark}"
        )


def write_sensitivity_report(
    path: Path,
    ranking: list[LayerSensitivity],
    promoted: set[str],
    metric: str = "kl",
) -> None:
    """Dump the full ranking as JSON so a selection can be revisited later."""
    path.write_text(
        json.dumps(
            {
                "metric": metric,
                "promoted_layers": sorted(
                    layer.index for layer in ranking if layer.key in promoted
                ),
                "layers": [
                    {
                        "index": layer.index,
                        "key": layer.key,
                        f"nvfp4_{metric}": layer.nvfp4,
                        f"fp8_{metric}": layer.fp8,
                        "gain": layer.gain,
                        "extra_bytes": layer.extra_bytes,
                        "promoted": layer.key in promoted,
                        "modules": list(layer.modules),
                    }
                    for layer in ranking
                ],
            },
            indent=2,
        )
    )
    print(f"Wrote sensitivity report to {path}")


def _collect_tensor_meta(output_dir: Path) -> dict:
    """Map every saved tensor name to its header entry ({dtype, shape, ...})."""
    weight_map = get_weight_mappings(str(output_dir))
    meta = {}
    for shard_path in set(weight_map.values()):
        header = get_safetensors_header(shard_path)
        header.pop("__metadata__", None)
        meta.update(header)
    return meta


def _resolve_checkpoint_dir(model_id: str) -> Path | None:
    """Local directory holding model_id's safetensors (hub cache if needed)."""
    local = Path(model_id)
    if local.is_dir():
        return local
    try:
        from huggingface_hub import snapshot_download

        return Path(
            snapshot_download(
                model_id,
                allow_patterns=["*.safetensors", "*.safetensors.index.json"],
            )
        )
    except Exception as exc:  # offline, gated, or not a hub id
        print(f"Could not resolve source checkpoint for {model_id}: {exc}")
        return None


def preserve_dropped_tensors(model_id: str, output_dir: Path, model) -> None:
    """
    Copy source-checkpoint tensors that the loaded model class has no home for.

    transformers only materializes modules its architecture class implements.
    Auxiliary heads shipped in a checkpoint but unimplemented upstream (e.g.
    Qwen3.6's `mtp.*` multi-token-prediction head, which vLLM uses for
    speculative decoding) are dropped on load and would be silently absent
    from the quantized output. Copy them across verbatim.

    Only whole top-level components missing from both the live module tree and
    the saved checkpoint are copied, so renamed-on-save modules (which do have
    a live counterpart) are never duplicated.
    """
    src_dir = _resolve_checkpoint_dir(model_id)
    if src_dir is None:
        return
    try:
        src_map = get_weight_mappings(str(src_dir))
    except Exception as exc:
        print(f"Could not read source checkpoint tensor list: {exc}")
        return

    live_roots = {
        name.split(".")[0]
        for name, _ in list(model.named_parameters()) + list(model.named_buffers())
    }
    saved = _collect_tensor_meta(output_dir)
    saved_roots = {name.split(".")[0] for name in saved}
    dropped_roots = {name.split(".")[0] for name in src_map} - live_roots - saved_roots
    if not dropped_roots:
        return

    names = sorted(name for name in src_map if name.split(".")[0] in dropped_roots)
    print(
        f"Preserving {len(names)} tensor(s) the model class dropped: "
        f"{', '.join(sorted(dropped_roots))}"
    )

    from safetensors import safe_open
    from safetensors.torch import save_file

    tensors = {}
    by_shard = {}
    for name in names:
        by_shard.setdefault(src_map[name], []).append(name)
    for shard, shard_names in by_shard.items():
        with safe_open(shard, framework="pt") as f:
            for name in shard_names:
                tensors[name] = f.get_tensor(name)

    # The extra tensors go in a shard of their own, which means renumbering the
    # existing ones into a sharded layout: loaders that find a lone
    # model.safetensors read only that file and ignore any index, so a
    # single-file output would silently drop the additions again. Renames are
    # metadata-only, so this is cheap even for a 27 GB shard.
    existing = sorted(output_dir.glob("*.safetensors"))
    total = len(existing) + 1
    shards = []
    for i, path in enumerate(existing, start=1):
        renamed = output_dir / f"model-{i:05d}-of-{total:05d}.safetensors"
        if path != renamed:
            path.rename(renamed)
        shards.append(renamed)
    new_shard = output_dir / f"model-{total:05d}-of-{total:05d}.safetensors"
    save_file(tensors, str(new_shard), metadata={"format": "pt"})
    shards.append(new_shard)

    weight_map, total_size = {}, 0
    for path in shards:
        header = get_safetensors_header(str(path))
        header.pop("__metadata__", None)
        for key, meta in header.items():
            weight_map[key] = path.name
            start, end = meta["data_offsets"]
            total_size += end - start
    (output_dir / "model.safetensors.index.json").write_text(
        json.dumps(
            {"metadata": {"total_size": total_size}, "weight_map": weight_map}, indent=2
        )
    )


def strip_ignored_kv_scales(model, ignore: list[str]) -> None:
    """Drop KV cache scales from attention modules the user asked to ignore.

    compressed-tensors applies `kv_cache_scheme` by walking every attention
    module in the model (`_apply_kv_cache_scheme`, 0.17.1) and never consults
    the ignore list, so a multimodal checkpoint comes out with k_scale/v_scale
    on its vision and audio towers as well as its decoder. vLLM only builds a
    KV cache for the decoder, so those extra scales have no parameter to load
    into and it refuses the whole checkpoint:

        ValueError: There is no module or parameter named
        'audio_tower.layers.0.self_attn.attn' in Gemma4ForConditionalGeneration

    The scales are inert either way -- nothing reads them for a tower that
    isn't KV-cached -- so dropping them before save costs nothing and is what
    --ignore already means everywhere else.
    """
    stripped = []
    for name, module in model.named_modules():
        if not any(match_name(name, pattern) for pattern in ignore):
            continue
        removed = False
        for attr in ("k_scale", "v_scale", "k_zero_point", "v_zero_point"):
            if hasattr(module, attr):
                delattr(module, attr)
                removed = True
        if removed:
            # Leaving the scheme behind would make the compressor look for the
            # parameters that were just removed. It has to be deleted rather
            # than set to None: is_module_quantized() tests hasattr and then
            # dereferences, so a None scheme raises instead of reading false.
            if hasattr(module, "quantization_scheme"):
                delattr(module, "quantization_scheme")
            stripped.append(name)

    if stripped:
        print(
            f"Dropped KV cache scales from {len(stripped)} ignored attention "
            f"modules (e.g. {stripped[0]})."
        )


def fix_ignore_list(output_dir: Path, extra_ignore: list[str] | None = None) -> None:
    """
    llm-compressor records the quantization ignore list using the live
    transformers module hierarchy. Some architectures (e.g. Gemma 4) rename
    modules between that hierarchy and the on-disk checkpoint via a checkpoint
    conversion mapping, so the recorded names don't match what vLLM (which
    loads directly from checkpoint tensor names) expects. Add the on-disk name
    of every unquantized 2D float weight to the ignore list. The recorded
    transformers-side names are kept alongside: transformers matches ignore
    against its live module names when reloading the checkpoint, so both
    spellings are needed.

    extra_ignore re-appends the user's --ignore patterns: llm-compressor
    expands regex patterns to the concrete modules it skipped, which loses
    weightless wrapper containers (e.g. the gemma-4 E-series
    Gemma4ClippableLinear around each tower projection). Group targets still
    match those wrappers when transformers re-applies the config on reload,
    so without the patterns the reload crashes on an unquantizable type.
    """
    config_path = output_dir / "config.json"
    config = json.loads(config_path.read_text())
    qconfig = config.get("quantization_config")
    if qconfig is None:
        return

    tensors = _collect_tensor_meta(output_dir)
    quantized_prefixes = {
        k[: -len(".weight_packed")] for k in tensors if k.endswith(".weight_packed")
    }

    original = set(qconfig.get("ignore", []))
    ignore = set(original)
    ignore.update(extra_ignore or [])
    for key, meta in tensors.items():
        if not key.endswith(".weight"):
            continue
        prefix = key[: -len(".weight")]
        if prefix in quantized_prefixes:
            continue
        if len(meta["shape"]) != 2 or meta["dtype"] not in (
            "F64",
            "F32",
            "F16",
            "BF16",
        ):
            continue
        ignore.add(prefix)

    if ignore == original:
        return

    qconfig["ignore"] = sorted(ignore)
    config_path.write_text(json.dumps(config, indent=2))
    print(
        f"Added on-disk layer names to quantization ignore list ({len(ignore)} entries)."
    )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument(
        "--output",
        default=None,
        help="Output directory (default: <model-basename>-NVFP4)",
    )
    p.add_argument(
        "--samples",
        type=int,
        default=512,
        help="Number of calibration samples (more = better accuracy)",
    )
    p.add_argument(
        "--max-len",
        type=int,
        default=1024,
        help="Max token length per calibration sample",
    )
    p.add_argument(
        "--weight-only",
        action="store_true",
        help="W4A16 (weights only, no calibration data needed)",
    )
    p.add_argument(
        "--fp8-attn",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep attention q/k/v/o projections at FP8 (channel-wise weights, "
        "dynamic per-token activations) instead of NVFP4, and calibrate an "
        "FP8 KV cache scale. The KV cache scale needs calibration data, so "
        "it is skipped with --weight-only. Disable with --no-fp8-attn for "
        "uniform NVFP4.",
    )
    p.add_argument(
        "--fp8-deltanet",
        action="store_true",
        help="On hybrid linear-attention models, keep the Gated DeltaNet "
        "projections (in_proj_qkv/in_proj_z/out_proj) at FP8 (channel-wise "
        "weights, dynamic per-token activations) instead of NVFP4, trading "
        "size for fidelity — quantizing DeltaNet to NVFP4 costs ~+0.0045 "
        "weight-only KL on Qwen3.6-27B. No effect on models without "
        "linear_attn modules.",
    )
    p.add_argument(
        "--fp8-lm-head",
        action="store_true",
        help="Quantize the output head to FP8 (channel-wise weights, dynamic "
        "per-token activations) instead of leaving it at model dtype, and drop "
        "it from --ignore. The head is one of the largest tensors left in an "
        "NVFP4 checkpoint (2.4 GiB of Qwen3.8-27B's 23 GB) and single-stream "
        "decode reads all of it once per lm_head pass — three times per step "
        "under k=2 MTP, which shares the head with the draft. Measured on "
        "Qwen3.8-27B against a checkpoint differing in this tensor alone: "
        "+14%% decode throughput at k=2 (+4.5%% with speculation off), for "
        "1.7 points of draft acceptance and +0.0005 weight-only KL.",
    )
    p.add_argument(
        "--gptq-mlp",
        choices=["auto", "on", "off"],
        default="auto",
        help="Quantize dense MLP gate/up/down projections with GPTQ error "
        "compensation (imatrix_mse observer + static activation ordering) "
        "instead of plain minmax. Calibration-only change: same NVFP4 "
        "on-disk format and serving cost, but ~20%% lower KL vs the BF16 "
        "original (plain minmax and scale-only imatrix both measure no "
        "better than each other). "
        "'auto' (default) enables it for dense models and skips it for MoE "
        "(per-expert calibration coverage is too thin), --weight-only, and "
        "--no-fp8-attn runs. 'on' requires calibration data, --fp8-attn, "
        "and a non-MoE model.",
    )
    p.add_argument(
        "--fp8-mlp",
        default="off",
        metavar="SPEC",
        help="Keep the most quantization-sensitive dense MLP layers at FP8 "
        "instead of NVFP4, trading size for fidelity. 'off' (default) is "
        "uniform NVFP4. 'top:N' promotes the N layers with the largest "
        "NVFP4-over-FP8 KL against the unquantized model. 'gptq-loss:N' "
        "ranks by GPTQ's own per-layer proxy loss instead (Hessian-weighted "
        "output MSE, the quantity GPTQ minimizes) -- the ranker that "
        "recovers the measured late-layer promotion advantage on Qwen-family "
        "hybrids where every KL trial misses it, and the recommended ranked "
        "mode there. An explicit list like '1,2,3,10-15,19' promotes exactly "
        "those decoder layers. 'scan' and 'gptq-loss' print their full "
        "ranking and exit without quantizing. The ranked modes need "
        "calibration data, so they cannot be combined with --weight-only; an "
        "explicit list can. Promotion is always whole-layer, because vLLM "
        "fuses gate_proj and up_proj into one module that can only carry one "
        "scheme.",
    )
    p.add_argument(
        "--sensitivity-samples",
        type=int,
        default=64,
        help="Calibration samples the --fp8-mlp ranking measures KL over "
        "(default: 64). The scan costs two forward passes per layer per "
        "sample, so this trades scan time against how finely it can separate "
        "layers whose KL is close.",
    )
    p.add_argument(
        "--sensitivity-report",
        default=None,
        metavar="PATH",
        help="Write the full --fp8-mlp sensitivity ranking to PATH as JSON.",
    )
    p.add_argument(
        "--sensitivity-context",
        choices=["clean", "quantized", "quantized-acts"],
        default="clean",
        help="What the rest of the model looks like during each --fp8-mlp "
        "trial. 'clean' (default) fake-quantizes one layer at a time against "
        "an otherwise-BF16 model. 'quantized' fake-quantizes every dense MLP "
        "layer at once (weights and activations; the BF16 originals are "
        "parked in CPU memory, roughly their on-disk size in host RAM) and "
        "scores each layer by the KL its FP8 promotion removes from that "
        "baseline. 'quantized-acts' quantizes activations only, isolating "
        "the activation-format half of each promotion. The non-clean modes "
        "are diagnostics: on Qwen3.8 neither reproduces the measured "
        "late-layer promotion advantage (it emerges only through the GPTQ "
        "pipeline), so on Qwen-family hybrids pass an explicit late-layer "
        "list instead of trusting any scan mode -- see the Mixed-precision "
        "MLP section of GUIDE.md.",
    )
    p.add_argument(
        "--int8-ple",
        choices=["auto", "on", "off"],
        default="off",
        help="Quantize the Gemma-4 E-series per-layer embedding table to "
        "weight-only INT8, per vocab row ('auto' enables it wherever such a "
        "table exists and is a no-op elsewhere; default: off). The table is a "
        "third of an E-series checkpoint, so this takes ~30%% off E2B and "
        "~26%% off E4B, in VRAM as well as on disk. It is off by default "
        "because it is a real trade, not free: English chat is unaffected, but "
        "under W4A4 it costs about 8%% KL on multilingual, tool-calling and "
        "code prompts. See the Per-layer embeddings section of GUIDE.md.",
    )
    p.add_argument(
        "--ignore",
        nargs="+",
        default=["lm_head"],
        help="Layer names/regex patterns to exclude from quantization "
        "(default: lm_head). Use re: prefix for regex patterns.",
    )
    p.add_argument(
        "--dtype",
        default="auto",
        help="Model dtype: auto, bfloat16, float16 (default: auto)",
    )
    p.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code when loading model/tokenizer",
    )
    p.add_argument(
        "--dataset",
        default="mix",
        help="Calibration data: a named mixture ('mix' = chat + instruct + "
        "code + math + multilingual + tool calls + raw web text, streamed "
        "from seven HuggingFace datasets and rendered through the model's "
        "chat template; 'ultrachat' = HuggingFaceH4/ultrachat_200k alone) or "
        "any HuggingFace dataset id (default: mix)",
    )
    p.add_argument(
        "--sensitivity-dataset",
        default="ultrachat",
        help="Data the --fp8-mlp ranking is measured on, same vocabulary of "
        "values as --dataset (default: ultrachat). It defaults differently "
        "from --dataset on purpose: the scan compares output distributions, "
        "and a corpus chosen for input coverage ranks layers worse. See the "
        "Mixed-precision MLP section of GUIDE.md.",
    )
    p.add_argument(
        "--vision-samples",
        default="auto",
        help="How many of --samples carry an image, for multimodal "
        "checkpoints. 'auto' (default) uses 12.5%% when the checkpoint has an "
        "image processor and 0 otherwise; pass an integer to override, or 0 "
        "to calibrate on text only.",
    )
    p.add_argument(
        "--split",
        default=None,
        help="Dataset split for calibration (default: train_sft for "
        "ultrachat_200k, train otherwise). Ignored for multi-source mixtures, "
        "which pin their own splits.",
    )
    p.add_argument(
        "--pipeline",
        choices=["auto", "sequential", "basic"],
        default="auto",
        help="Calibration pipeline. 'auto' lets llm-compressor infer one "
        "(sequential tracing for GPTQ). 'basic' runs plain full-model "
        "forwards; use it for architectures whose forward cannot be traced "
        "into per-layer subgraphs (e.g. gemma-4 E-series shared-KV lookups "
        "fail under the sequential tracer). With GPTQ, basic keeps every "
        "target module's hessian in memory at once.",
    )
    p.add_argument(
        "--cpu-offload",
        action="store_true",
        help="Load model to CPU/system RAM; llm-compressor dispatches "
        "layers to GPU during calibration. Use for large MoE models "
        "that don't fit alongside expert-unpacking overhead.",
    )
    return p.parse_args()


def main():
    args = parse_args()

    model_id = args.model
    basename = model_id.rstrip("/").split("/")[-1]
    suffix = "-NVFP4-W4A16" if args.weight_only else "-NVFP4"
    output_dir = args.output or (basename + suffix)

    scheme = "NVFP4A16" if args.weight_only else "NVFP4"
    # Regex targets outrank the class-name target when a module matches both
    # groups, so attention projections land in the FP8 group and every other
    # Linear falls through to NVFP4.
    config_groups = {"group_0": preset_name_to_scheme(scheme, ["Linear"])}
    kv_cache_scheme = None
    if args.fp8_attn:
        attn_scheme = preset_name_to_scheme(
            "FP8_DYNAMIC", [FP8_ATTN_TARGET, FP8_ATTN_FUSED_TARGET]
        )
        if args.weight_only:
            # Honor the A16 contract: FP8 weights only (W8A16), activations
            # stay at model dtype. The KV cache scale is skipped because its
            # static observer needs calibration data.
            attn_scheme.input_activations = None
        else:
            kv_cache_scheme = QuantizationArgs(
                num_bits=8,
                type="float",
                strategy="tensor",
                symmetric=True,
                dynamic=False,
                observer="static_minmax",
            )
        config_groups["group_1"] = attn_scheme
    gptq_mlp = args.gptq_mlp
    if gptq_mlp == "on":
        if args.weight_only:
            raise SystemExit(
                "--gptq-mlp on needs calibration data to collect activation "
                "statistics; it cannot be combined with --weight-only."
            )
        if not args.fp8_attn:
            raise SystemExit(
                "--gptq-mlp on moves the MLP projections into a GPTQ modifier "
                "and drops the catch-all NVFP4 Linear group, so it needs "
                "--fp8-attn to cover the attention projections."
            )
    elif gptq_mlp == "auto" and (args.weight_only or not args.fp8_attn):
        gptq_mlp = "off"

    # --fp8-mlp resolves to a mode now and to concrete module names once the
    # model (and, for the ranked modes, the calibration data) is loaded.
    # 'gptq-loss' and 'gptq-loss:N' share the scan/top plumbing and differ
    # only in the ranker behind the table.
    fp8_mlp_mode, fp8_mlp_top, fp8_mlp_indices = "off", 0, set()
    fp8_mlp_ranker, fp8_spec = "kl", args.fp8_mlp
    if fp8_spec == "gptq-loss" or fp8_spec.startswith("gptq-loss:"):
        fp8_mlp_ranker = "gptq"
        fp8_spec = (
            "scan"
            if fp8_spec == "gptq-loss"
            else "top:" + fp8_spec[len("gptq-loss:") :]
        )
    if fp8_spec == "scan":
        fp8_mlp_mode = "scan"
    elif fp8_spec.startswith("top:"):
        fp8_mlp_mode = "top"
        try:
            fp8_mlp_top = int(fp8_spec[len("top:") :])
        except ValueError:
            raise SystemExit(
                f"--fp8-mlp: {args.fp8_mlp!r} is not a number of layers. "
                "Expected 'top:N' or 'gptq-loss:N', e.g. 'top:12'."
            ) from None
        if fp8_mlp_top < 0:
            raise SystemExit(f"--fp8-mlp top:N needs N >= 0, got {fp8_mlp_top}")
    elif fp8_spec != "off":
        fp8_mlp_mode = "explicit"
        fp8_mlp_indices = parse_layer_spec(args.fp8_mlp)

    if fp8_mlp_mode in ("scan", "top") and args.weight_only:
        raise SystemExit(
            "--fp8-mlp scan, top:N and gptq-loss rank layers by running data "
            "through the model, which needs the calibration pass --weight-only "
            "skips. Pass an explicit layer list instead."
        )
    if fp8_mlp_ranker == "gptq" and args.sensitivity_context != "clean":
        raise SystemExit(
            "--fp8-mlp gptq-loss accumulates Hessians from clean-model inputs; "
            "--sensitivity-context does not apply to it."
        )

    # --fp8-lm-head takes the head off the ignore list. Done here rather than
    # after the model loads so the status header below prints the ignore list
    # the run actually uses; the live module tree is checked once it exists.
    if args.fp8_lm_head:
        args.ignore = [
            pattern for pattern in args.ignore if not match_name("lm_head", pattern)
        ]

    mode = "W4A16 (weight-only)" if args.weight_only else "W4A4 (weights + activations)"
    if args.fp8_attn:
        mode += " + FP8 attention"
    if kv_cache_scheme is not None:
        mode += " + FP8 KV cache"
    if gptq_mlp == "on":
        mode += " + GPTQ MLP"
    if fp8_mlp_mode != "off":
        mode += f" + FP8 MLP fallback ({args.fp8_mlp})"
    if args.fp8_lm_head:
        mode += " + FP8 lm_head"
    if args.int8_ple == "on":
        # "auto" is reported once the module tree says whether it applies.
        mode += " + INT8 PLE"

    print(f"Model:       {model_id}")
    print(f"Mode:        {mode}")
    print(f"Output dir:  {output_dir}")
    print(f"Dtype:       {args.dtype}")
    print(f"Ignore:      {args.ignore}")
    if not args.weight_only:
        print(f"Calibration: {args.samples} samples, max {args.max_len} tokens each")
        print(f"Dataset:     {args.dataset}")
        if fp8_mlp_mode in ("scan", "top"):
            how = (
                "GPTQ proxy loss"
                if fp8_mlp_ranker == "gptq"
                else f"{args.sensitivity_context} context"
            )
            print(f"Ranked on:   {args.sensitivity_dataset} ({how})")
        if args.vision_samples != "auto":
            try:
                args.vision_samples = int(args.vision_samples)
            except ValueError:
                raise SystemExit(
                    f"--vision-samples must be 'auto' or an integer, got "
                    f"{args.vision_samples!r}"
                ) from None
            if not 0 <= args.vision_samples <= args.samples:
                raise SystemExit(
                    f"--vision-samples must be between 0 and --samples "
                    f"({args.samples}), got {args.vision_samples}"
                )

    print("\nLoading model...")
    load_kwargs = dict(dtype=args.dtype)
    if not args.cpu_offload:
        load_kwargs["device_map"] = "auto"
    if args.trust_remote_code:
        load_kwargs["trust_remote_code"] = True
    # Load the class the checkpoint declares. AutoModelForCausalLM strips some
    # multimodal models to their text submodel (e.g. Qwen3.5's
    # ...ForConditionalGeneration resolves to ...ForCausalLM), which would
    # silently drop the vision tower from the saved checkpoint.
    model_cls = AutoModelForCausalLM
    config = AutoConfig.from_pretrained(
        model_id, trust_remote_code=args.trust_remote_code
    )
    declared = (getattr(config, "architectures", None) or [None])[0]
    if declared is not None and hasattr(transformers, declared):
        model_cls = getattr(transformers, declared)
        if model_cls is not AutoModelForCausalLM:
            print(f"Model class: {declared} (declared by checkpoint config)")
    model = model_cls.from_pretrained(model_id, **load_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=args.trust_remote_code
    )
    # For text-only models AutoProcessor just hands back the tokenizer. Loaded
    # here rather than at save time because image calibration samples have to
    # go through the model's own processor.
    try:
        processor = AutoProcessor.from_pretrained(
            model_id, trust_remote_code=args.trust_remote_code
        )
    except Exception:
        processor = None
    if isinstance(processor, PreTrainedTokenizerBase):
        processor = None

    if args.fp8_attn and not any(
        match_name(name, FP8_ATTN_TARGET) for name, _ in model.named_modules()
    ):
        raise SystemExit(
            f"--fp8-attn matched no modules ({FP8_ATTN_TARGET!r}); this "
            "architecture likely fuses QKV or uses MLA naming. Rerun with "
            "--no-fp8-attn or adjust FP8_ATTN_TARGET."
        )

    # --fp8-lm-head resolves against the live module tree for the same reason
    # --fp8-attn does: an architecture that names its head differently should
    # say so rather than quietly produce a checkpoint without the promotion.
    lm_head_modules: list[str] = []
    if args.fp8_lm_head:
        lm_head_modules = sorted(
            name
            for name, module in model.named_modules()
            if isinstance(module, torch.nn.Linear) and match_name(name, LM_HEAD_TARGET)
        )
        if not lm_head_modules:
            raise SystemExit(
                f"--fp8-lm-head matched no Linear module ({LM_HEAD_TARGET!r}); "
                "this architecture names its output head differently."
            )
        conflicting = [
            name
            for name in lm_head_modules
            if any(match_name(name, pattern) for pattern in args.ignore)
        ]
        if conflicting:
            raise SystemExit(
                f"--fp8-lm-head conflicts with --ignore: {', '.join(conflicting)} "
                "is excluded from quantization. Drop the pattern covering it."
            )
        if getattr(model.config.get_text_config(), "tie_word_embeddings", False):
            raise SystemExit(
                "--fp8-lm-head: this model ties lm_head to embed_tokens, so the "
                "head and the input embedding table are the same tensor and "
                "quantizing one quantizes the other. vLLM reads the table with a "
                "gather kernel that takes weight-only INT group/channel and "
                "nothing else, so the result would not load."
            )
        # BF16 -> FP8 halves the weight; the per-row scales are one BF16 value
        # per vocab entry, which rounds to nothing beside it.
        saved = sum(model.get_submodule(n).weight.nbytes for n in lm_head_modules) // 2
        print(f"FP8 lm_head: {', '.join(lm_head_modules)} (-{saved / 1024**2:.0f} MiB)")

    # Hybrid linear-attention models get an explicit DeltaNet group so the
    # projections survive --gptq-mlp deleting the catch-all Linear group.
    # Resolved from the live module tree: non-hybrid models keep their
    # config_groups (and saved config.json) unchanged.
    if any(match_name(name, DELTANET_TARGET) for name, _ in model.named_modules()):
        args.ignore = list(args.ignore) + list(DELTANET_IGNORE)
        if args.fp8_deltanet:
            deltanet_scheme = preset_name_to_scheme("FP8_DYNAMIC", [DELTANET_TARGET])
            if args.weight_only:
                # Same A16 contract as --fp8-attn: FP8 weights only.
                deltanet_scheme.input_activations = None
            deltanet_name = "FP8"
        else:
            deltanet_scheme = preset_name_to_scheme(scheme, [DELTANET_TARGET])
            deltanet_name = scheme
        config_groups["group_5"] = deltanet_scheme
        print(
            f"DeltaNet:    {deltanet_name} on {DELTANET_TARGET!r} "
            "(in_proj_a/in_proj_b left unquantized)"
        )
    elif args.fp8_deltanet:
        print("FP8 DeltaNet: no linear_attn modules on this model; no effect")

    # --gptq-mlp needs the live module tree: auto skips MoE models, and the
    # recipe with GPTQ drops the catch-all Linear group, which would leave
    # MoE expert projections unquantized entirely.
    if gptq_mlp != "off":
        module_names = [name for name, _ in model.named_modules()]
        has_dense_mlp = any(match_name(n, GPTQ_MLP_TARGET) for n in module_names)
        is_moe = any(n.endswith(".experts") or ".experts." in n for n in module_names)
        if gptq_mlp == "auto":
            if is_moe:
                print(
                    "GPTQ MLP:    auto -> skipped (MoE experts detected; "
                    "per-expert calibration coverage is too thin)"
                )
                gptq_mlp = "off"
            elif not has_dense_mlp:
                print(
                    "GPTQ MLP:    auto -> skipped (no modules match "
                    f"{GPTQ_MLP_TARGET!r})"
                )
                gptq_mlp = "off"
            else:
                print("GPTQ MLP:    auto -> enabled (dense MLP projections)")
                gptq_mlp = "on"
        else:
            if is_moe:
                raise SystemExit(
                    "--gptq-mlp on drops the catch-all NVFP4 Linear group, "
                    "which would leave this model's MoE expert projections "
                    "unquantized. Use --gptq-mlp auto or off for MoE models."
                )
            if not has_dense_mlp:
                raise SystemExit(
                    f"--gptq-mlp on matched no modules ({GPTQ_MLP_TARGET!r}); "
                    "this architecture names its MLP projections differently. "
                    "Rerun with --gptq-mlp off or adjust GPTQ_MLP_TARGET."
                )

    # --int8-ple only applies to architectures carrying a per-layer embedding
    # table (the Gemma-4 E-series). Resolve against the live module tree rather
    # than a config flag so "auto" stays a no-op everywhere else.
    ple_matches = (
        {}
        if args.int8_ple == "off"
        else {
            name: module
            for name, module in model.named_modules()
            if isinstance(module, torch.nn.Embedding)
            and match_name(name, PLE_TARGET)
            and not any(match_name(name, pattern) for pattern in args.ignore)
        }
    )
    ple_modules = sorted(ple_matches)
    if args.int8_ple == "on" and not ple_modules:
        raise SystemExit(
            f"--int8-ple on matched no embedding module ({PLE_TARGET!r}); only "
            "the Gemma-4 E-series has a per-layer embedding table. Rerun with "
            "--int8-ple off or auto."
        )
    if args.int8_ple == "auto" and not ple_modules:
        print(f"INT8 PLE:    auto -> skipped (no {PLE_TARGET!r} module)")
    elif ple_modules:
        # BF16 -> INT8 halves the table; the per-row scales are one BF16 value
        # per vocab entry, which rounds to nothing beside it.
        saved = sum(m.weight.nbytes for m in ple_matches.values()) // 2
        print(f"INT8 PLE:    {', '.join(ple_modules)} (-{saved / 1024**2:.0f} MiB)")

    # Every dense MLP projection the NVFP4 recipe would claim, enumerated from
    # the live module tree. --fp8-mlp needs to split this set into two disjoint
    # explicit target lists: when two groups both match a module the winner is
    # decided by match order, which is a silent way to load the wrong scheme.
    mlp_modules = [
        name
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear)
        and match_name(name, GPTQ_MLP_TARGET)
        and not any(match_name(name, pattern) for pattern in args.ignore)
    ]

    # Calibration data is built before the recipe rather than alongside the
    # quantization call: --fp8-mlp top:N and scan rank layers by running this
    # data through the unquantized model, and that ranking sets recipe targets.
    ds = None
    if not args.weight_only:
        ds = build_dataset(args, args.dataset, "Calibration", tokenizer, processor)

    # The ranking corpus is deliberately separate from the calibration corpus;
    # see --sensitivity-dataset for why they default differently.
    scan_ds = ds
    if fp8_mlp_mode in ("scan", "top") and args.sensitivity_dataset != args.dataset:
        scan_ds = build_dataset(
            args, args.sensitivity_dataset, "Sensitivity", tokenizer, processor
        )

    fp8_mlp_modules: list[str] = []
    if fp8_mlp_mode != "off":
        if not mlp_modules:
            raise SystemExit(
                f"--fp8-mlp matched no dense MLP projections ({GPTQ_MLP_TARGET!r}); "
                "this architecture names its MLP projections differently."
            )
        by_index: dict[int, list[str]] = {}
        for name in mlp_modules:
            by_index.setdefault(_layer_index(_layer_key(name)), []).append(name)

        if fp8_mlp_mode == "explicit":
            missing = sorted(fp8_mlp_indices - by_index.keys())
            if missing:
                raise SystemExit(
                    f"--fp8-mlp: no MLP projections in layer(s) {missing}; this "
                    f"model has layers {min(by_index)}-{max(by_index)}."
                )
            fp8_mlp_modules = sorted(n for i in fp8_mlp_indices for n in by_index[i])
            print(f"FP8 MLP:     layers {sorted(fp8_mlp_indices)} (explicit)")
        else:
            if fp8_mlp_ranker == "gptq":
                ranking = rank_mlp_gptq_loss(
                    model,
                    scan_ds,
                    mlp_modules,
                    preset_name_to_scheme("NVFP4", ["Linear"]),
                    preset_name_to_scheme("FP8_DYNAMIC", ["Linear"]),
                    args.sensitivity_samples,
                )
            else:
                ranking = scan_mlp_sensitivity(
                    model,
                    scan_ds,
                    mlp_modules,
                    preset_name_to_scheme("NVFP4", ["Linear"]).weights,
                    preset_name_to_scheme("FP8_DYNAMIC", ["Linear"]).weights,
                    args.sensitivity_samples,
                    context=args.sensitivity_context,
                )
            if not ranking:
                raise SystemExit(
                    "--fp8-mlp: the ranking pass observed no activations, so "
                    "the calibration forward pass reached none of the MLP "
                    "projections. Nothing to rank."
                )
            if fp8_mlp_top > len(ranking):
                print(
                    f"  note: --fp8-mlp top:{fp8_mlp_top} exceeds the "
                    f"{len(ranking)} MLP layers this model has; promoting all "
                    "of them, which is plain FP8 with extra steps."
                )
            promoted = {layer.key for layer in ranking[:fp8_mlp_top]}
            print_sensitivity_table(
                ranking,
                promoted,
                metric="loss" if fp8_mlp_ranker == "gptq" else "KL",
            )
            if args.sensitivity_report:
                write_sensitivity_report(
                    Path(args.sensitivity_report),
                    ranking,
                    promoted,
                    metric="gptq_loss" if fp8_mlp_ranker == "gptq" else "kl",
                )
            if fp8_mlp_mode == "scan":
                print(f"\n--fp8-mlp {args.fp8_mlp}: ranking only, nothing quantized.")
                return
            chosen = [layer for layer in ranking if layer.key in promoted]
            fp8_mlp_modules = sorted(n for layer in chosen for n in layer.modules)
            print(
                f"\nFP8 MLP:     layers {sorted(c.index for c in chosen)} "
                f"(+{sum(c.extra_bytes for c in chosen) / 1024**2:.0f} MiB)"
            )

    if fp8_mlp_modules:
        # A group of its own, targeted by exact module name. Exact names outrank
        # both regex and class-name targets, so this resolves unambiguously
        # whichever way the NVFP4 side below is expressed.
        fp8_mlp_scheme = preset_name_to_scheme("FP8_DYNAMIC", fp8_mlp_modules)
        if args.weight_only:
            fp8_mlp_scheme.input_activations = None
        config_groups["group_3"] = fp8_mlp_scheme

    if lm_head_modules:
        # The regex, not the resolved module names — the opposite of the FP8 MLP
        # group above, and it matters. vLLM rewrites exact-name targets through
        # the model's hf_to_vllm_mapper before matching (apply_vllm_mapper), and
        # those mappers are prefix rules with a trailing dot: Qwen3-VL maps
        # "model.language_model." -> "language_model.model.", which fixes up the
        # MLP names, but its head rule is "lm_head." -> "language_model.lm_head."
        # and a bare "lm_head" target has no trailing dot to match. It survives
        # unrewritten, then fails to equal the real prefix "language_model.lm_head",
        # so vLLM builds an unquantized ParallelLMHead and the load dies on the
        # unexpected lm_head.weight_scale. Regex targets are passed through
        # untouched by that mapper and matched with re.match, so they hit the
        # head whatever the wrapper prefixes it with. Still outranks the
        # catch-all Linear group at both ends: compressed-tensors resolves regex
        # before class-name targets, and vLLM only falls back to class names
        # after the layer-name pass fails.
        lm_head_scheme = preset_name_to_scheme("FP8_DYNAMIC", [LM_HEAD_TARGET])
        if args.weight_only:
            lm_head_scheme.input_activations = None
        config_groups["group_6"] = lm_head_scheme

    if ple_modules:
        # Weight-only: an embedding is a gather, not a matmul, so there is no
        # input activation to quantize. The INT8 preset is already per-channel
        # (one scale per vocab row) symmetric minmax, which is what vLLM's
        # embedding kernel expects -- it rejects anything but weight-only INT
        # group/channel.
        ple_scheme = preset_name_to_scheme("INT8", ple_modules)
        ple_scheme.input_activations = None
        # Name the format instead of letting compressed-tensors infer it.
        # infer_module_format() gates on `type(module) in (Linear, Embedding)`,
        # an identity test that a subclass fails -- and Gemma 4's table is a
        # Gemma4TextScaledWordEmbedding. Inference would fall through to
        # `dense` and silently save the table uncompressed, at full BF16 size,
        # while the scales alongside it made it look quantized. Setting
        # scheme.format takes priority over inference (compressors/base.py
        # compress_module), and nothing downstream re-checks the module type.
        ple_scheme.format = CompressionFormat.pack_quantized.value
        config_groups["group_4"] = ple_scheme

    if gptq_mlp == "on":
        # GPTQ error compensation is what actually delivers the KL gain — a
        # plain QuantizationModifier with observer=imatrix_mse only refines
        # group scales and measures no better than minmax (verified by paired
        # per-token KL vs BF16 on gemma-4-12B, 2026-07-29). GPTQ owns the MLP
        # projections; the catch-all Linear group is dropped so the two
        # modifiers never claim the same module.
        del config_groups["group_0"]
        # Without a promotion the regex stays, so an unpromoted run produces
        # exactly the config it always did.
        promoted_set = set(fp8_mlp_modules)
        mlp_targets = (
            [n for n in mlp_modules if n not in promoted_set]
            if fp8_mlp_modules
            else [GPTQ_MLP_TARGET]
        )
        mlp_scheme = preset_name_to_scheme(scheme, mlp_targets)
        mlp_scheme.weights.observer = "imatrix_mse"
        mlp_scheme.weights.actorder = "static"
        gather_recipe = None
        recipe = [
            QuantizationModifier(
                config_groups=config_groups,
                ignore=args.ignore,
                kv_cache_scheme=kv_cache_scheme,
            ),
            GPTQModifier(
                config_groups={"group_2": mlp_scheme},
                ignore=args.ignore,
            ),
        ]
        if IMatrixGatherer is not None:
            # llm-compressor 0.12.0 only collects imatrix importance stats
            # when the gatherer runs first; nightlies self-collect (the class
            # is gone there, hence the guarded import).
            if args.pipeline == "basic":
                # The basic pipeline runs every modifier in one shared
                # calibration epoch, which breaks the gatherer-before-GPTQ
                # ordering — gather in a separate session instead.
                gather_recipe = _PersistentIMatrixGatherer(
                    targets=mlp_targets, ignore=args.ignore
                )
            else:
                recipe.insert(
                    0, IMatrixGatherer(targets=mlp_targets, ignore=args.ignore)
                )
    else:
        gather_recipe = None
        recipe = QuantizationModifier(
            config_groups=config_groups,
            ignore=args.ignore,
            kv_cache_scheme=kv_cache_scheme,
        )

    # compressed-tensors initializes KV cache scales from the top-level model
    # config, but multimodal configs (e.g. Gemma 4) nest the attention geometry
    # inside text_config. Mirror the attributes it reads onto the top-level
    # config, and drop them after quantization so they don't leak into the
    # saved config.json.
    kv_shim_attrs = []
    text_config = model.config.get_text_config()
    if kv_cache_scheme is not None and text_config is not model.config:
        for attr in ("num_attention_heads", "num_key_value_heads", "head_dim"):
            if not hasattr(model.config, attr) and hasattr(text_config, attr):
                setattr(model.config, attr, getattr(text_config, attr))
                kv_shim_attrs.append(attr)

    if args.weight_only:
        print("Running weight-only quantization (no calibration data needed)...")
        oneshot(model=model, recipe=recipe)
    else:
        oneshot_kwargs = {}
        if args.pipeline != "auto":
            oneshot_kwargs["pipeline"] = args.pipeline
        if gather_recipe is not None:
            print("Gathering imatrix statistics (separate calibration pass)...")
            oneshot(
                model=model,
                dataset=ds,
                recipe=gather_recipe,
                max_seq_length=args.max_len,
                num_calibration_samples=args.samples,
                **oneshot_kwargs,
            )
        print("Applying NVFP4 quantization with calibration data...")
        oneshot(
            model=model,
            dataset=ds,
            recipe=recipe,
            max_seq_length=args.max_len,
            num_calibration_samples=args.samples,
            **oneshot_kwargs,
        )

    for attr in kv_shim_attrs:
        delattr(model.config, attr)

    if kv_cache_scheme is not None:
        strip_ignored_kv_scales(model, args.ignore)

    print(f"\nSaving quantized model to ./{output_dir} ...")
    model.save_pretrained(output_dir, save_compressed=True)
    tokenizer.save_pretrained(output_dir)

    # Multimodal models (e.g. Gemma 4) need preprocessor_config.json for the
    # vision/audio feature extractor. `processor` is None for text-only models,
    # where AutoProcessor just returns the tokenizer again.
    if processor is not None:
        processor.save_pretrained(output_dir)
        if hasattr(processor, "image_processor"):
            processor.image_processor.save_pretrained(output_dir)
        print("Saved processor config (multimodal model)")

    preserve_dropped_tensors(model_id, Path(output_dir), model)
    fix_ignore_list(Path(output_dir), extra_ignore=args.ignore)

    size_mb = (
        sum(f.stat().st_size for f in Path(output_dir).rglob("*") if f.is_file())
        / 1024**2
    )
    print(f"Done. Output size: {size_mb:.0f} MB  (saved to ./{output_dir})")


if __name__ == "__main__":
    main()
