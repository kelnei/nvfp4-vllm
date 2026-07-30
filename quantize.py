"""
Quantize a model to NVFP4 (W4A4) using llm-compressor.

By default attention q/k/v/o projections are kept at FP8 (channel-wise
weights, dynamic per-token activations) and an FP8 KV cache scale is
calibrated — the mixed-precision layout vLLM's NVFP4 kernels expect. FP8
attention is ~3.5x closer to the original weights than NVFP4 for a small
size cost (~13% on a dense 12B, ~1% on a MoE).

Usage:
    python quantize.py [--model MODEL_ID] [--output OUTPUT_DIR]
                       [--samples N] [--max-len N] [--weight-only]
                       [--no-fp8-attn] [--gptq-mlp {auto,on,off}]
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
    gptq-mlp   = auto (GPTQ + imatrix_mse observer + static actorder on
                 dense MLP gate/up/down projections; calibration-only, same
                 on-disk format, ~20% lower KL vs BF16 than plain minmax.
                 auto enables it for dense models and skips it for MoE,
                 --weight-only, and --no-fp8-attn runs)
    ignore     = lm_head
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
from pathlib import Path
from typing import NamedTuple

from compressed_tensors.quantization import (
    QuantizationArgs,
    preset_name_to_scheme,
)
from compressed_tensors.utils import match_name
from compressed_tensors.utils.safetensors_load import (
    get_safetensors_header,
    get_weight_mappings,
)
from datasets import load_dataset
from llmcompressor import oneshot
from llmcompressor.modifiers.gptq import GPTQModifier
from llmcompressor.modifiers.quantization import QuantizationModifier
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

    mode = "W4A16 (weight-only)" if args.weight_only else "W4A4 (weights + activations)"
    if args.fp8_attn:
        mode += " + FP8 attention"
    if kv_cache_scheme is not None:
        mode += " + FP8 KV cache"
    if gptq_mlp == "on":
        mode += " + GPTQ MLP"

    print(f"Model:       {model_id}")
    print(f"Mode:        {mode}")
    print(f"Output dir:  {output_dir}")
    print(f"Dtype:       {args.dtype}")
    print(f"Ignore:      {args.ignore}")
    if not args.weight_only:
        print(f"Calibration: {args.samples} samples, max {args.max_len} tokens each")
        print(f"Dataset:     {args.dataset}")
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

    if gptq_mlp == "on":
        # GPTQ error compensation is what actually delivers the KL gain — a
        # plain QuantizationModifier with observer=imatrix_mse only refines
        # group scales and measures no better than minmax (verified by paired
        # per-token KL vs BF16 on gemma-4-12B, 2026-07-29). GPTQ owns the MLP
        # projections; the catch-all Linear group is dropped so the two
        # modifiers never claim the same module.
        del config_groups["group_0"]
        mlp_scheme = preset_name_to_scheme(scheme, [GPTQ_MLP_TARGET])
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
                    targets=[GPTQ_MLP_TARGET], ignore=args.ignore
                )
            else:
                recipe.insert(
                    0, IMatrixGatherer(targets=[GPTQ_MLP_TARGET], ignore=args.ignore)
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
        print(f"Loading calibration dataset ({args.dataset})...")

        sources = CALIBRATION_MIXES.get(args.dataset)
        if sources is None:
            # Any HuggingFace dataset id still works as a single-source mix.
            # ultrachat names its split "train_sft"; most datasets use "train".
            default_split = (
                "train_sft"
                if args.dataset == "HuggingFaceH4/ultrachat_200k"
                else "train"
            )
            sources = (
                MixSource(
                    args.dataset, "custom", 1.0, split=args.split or default_split
                ),
            )
        elif args.split:
            if len(sources) == 1:
                sources = (sources[0]._replace(split=args.split),)
            else:
                print(
                    f"  note: --split {args.split} ignored for mixture {args.dataset}"
                )

        n_vision = args.vision_samples
        if n_vision == "auto":
            has_image_processor = processor is not None and (
                getattr(processor, "image_processor", None) is not None
            )
            n_vision = (
                round(args.samples * VISION_FRACTION) if has_image_processor else 0
            )
        elif n_vision > 0 and processor is None:
            raise SystemExit(
                "--vision-samples needs an image processor, but "
                f"AutoProcessor.from_pretrained({model_id!r}) returned none."
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
            raise SystemExit(f"Calibration mixture {args.dataset!r} yielded no samples")

        total_tokens = sum(s["input_ids"].shape[-1] for s in samples)
        breakdown = ", ".join(f"{role} {n}" for role, n in counts.items())
        print(
            f"Calibration set: {len(samples)} samples, {total_tokens:,} tokens "
            f"({breakdown})"
        )
        ds = build_calibration_loader(samples)

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
