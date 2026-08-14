"""
Verify a quantized checkpoint against silent corruption (e.g. RAM or disk
bit flips picked up during quantization or at rest).

Four checks, cheapest first:

1. Pass-through tensors (embeddings, norms, lm_head, ignored projections,
   mtp.*) must be byte-identical to the source checkpoint they were copied
   from. Catches every possible flip in these tensors. Zero-centered
   RMSNorm weights (stored as bf16(1 + w) - 1) may legitimately differ in
   bytes; they are accepted when 1 + w still matches in bf16.
2. Global scales and KV-cache scales must be finite and positive. A high
   exponent-bit flip in one of these garbles an entire tensor.
3. Block and channel quantization scales are scanned for extreme outliers
   against their own tensor's distribution (reference-free). An exponent
   flip in an FP8 block scale multiplies 16 weights by up to 256x.
4. Quantized weights are dequantized and compared against the source
   weights, measured in units of the local decode scale. Legitimate
   quantization error is bounded (~1 unit for minmax groups; GPTQ error
   compensation reaches further, so those layers get a looser threshold).
   This catches sign-bit and exponent-bit nibble flips. A flip that moves
   a weight by less than ~2 scale-units sits inside normal quantization
   noise and is fundamentally undetectable — and equally harmless.

Every reported finding is double-checked with an independent recomputation
(or re-read), so a transient RAM flip striking this script's own arithmetic
— entirely possible on the faulty hardware it exists to detect — cannot
fabricate a finding.

Usage:
    python verify.py --model ./Model-NVFP4 --orig SOURCE_PATH_OR_HF_ID
    python verify.py --model ./Model-NVFP4 --quick   # reference-free checks only

Exit code is 0 when clean, 1 when any check found something.
"""

import argparse
import json
import re
import sys
from pathlib import Path

import torch
from compressed_tensors.compressors.nvfp4.helpers import unpack_fp4_from_uint8
from safetensors import safe_open

# Max legitimate |dequantized - original| in units of the block's decode
# scale. Minmax rounding stays within half the widest E2M1 code gap (1.0)
# plus FP8 scale-representation error; GPTQ error compensation deliberately
# moves weights further off their nearest grid point (measured tail ~11
# units on Qwen3.6-27B, recurring in the same low-importance input channels
# across layers — so in GPTQ layers only a sign flip of the largest code
# (12 units) is distinguishable from legitimate compensation).
DEV_THRESHOLD_MINMAX = 2.0
DEV_THRESHOLD_GPTQ = 12.0
# Max legitimate relative error for FP8 E4M3 weights (half of the 2^-3
# mantissa spacing = 6.25%; a mantissa-LSB flip lands at 12.5%).
FP8_REL_THRESHOLD = 0.09
# A block scale this many times the tensor's median scale is flagged.
SCALE_OUTLIER_RATIO = 64.0


def resolve_dir(spec: str) -> Path:
    path = Path(spec).expanduser()
    if path.exists():
        return path
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(spec))


def load_index(model_dir: Path) -> dict[str, Path]:
    """Map tensor name -> shard path, for sharded or single-file checkpoints."""
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        weight_map = json.loads(index_path.read_text())["weight_map"]
        return {name: model_dir / shard for name, shard in weight_map.items()}
    single = model_dir / "model.safetensors"
    with safe_open(single, framework="pt") as f:
        return {name: single for name in f.keys()}


class TensorLoader:
    """Load tensors by name, keeping at most one shard file open."""

    def __init__(self, index: dict[str, Path]):
        self.index = index
        self._path = None
        self._file = None

    def __call__(self, name: str) -> torch.Tensor:
        path = self.index[name]
        if path != self._path:
            self._file = safe_open(path, framework="pt")
            self._path = path
        return self._file.get_tensor(name)


def bit_pattern(t: torch.Tensor) -> str:
    if t.dtype == torch.bfloat16 or t.dtype == torch.float16:
        return f"0x{t.view(torch.uint16).item():04X}"
    if t.element_size() == 1:
        return f"0x{t.view(torch.uint8).item():02X}"
    return "?"


def zero_centered_equivalent(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Qwen3.5-family RMSNorm weights are stored zero-centered: transformers
    saves bf16(1 + w) - 1, and the model only ever computes with 1 + w. The
    round trip can change bytes (round-up near an exponent boundary, flush
    to zero below half an ulp of 1.0) without changing 1 + w in bf16."""
    if a.dtype != torch.bfloat16:
        return False
    lifted_a = (a.float() + 1).to(torch.bfloat16)
    lifted_b = (b.float() + 1).to(torch.bfloat16)
    return torch.equal(lifted_a, lifted_b)


def check_passthrough(quant, orig, findings) -> tuple[int, int, int]:
    """Tensors present in both checkpoints with identical dtype and shape
    were copied through quantization untouched and must match byte-for-byte."""
    common = sorted(set(quant.index) & set(orig.index))
    checked = 0
    zero_centered = 0
    transients = 0
    for name in common:
        a, b = quant(name), orig(name)
        if a.dtype != b.dtype or a.shape != b.shape:
            continue  # e.g. an FP8-quantized weight shadowing the BF16 name
        checked += 1
        if torch.equal(
            a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)
        ):
            continue
        if zero_centered_equivalent(a, b):
            zero_centered += 1
            continue
        # Re-read both tensors before reporting: a RAM flip in one of the
        # in-memory copies (or the comparison itself) fabricates a phantom
        # diff. Only positions that differ in both reads are real.
        a, b = quant(name), orig(name)
        flat_a, flat_b = a.flatten(), b.flatten()
        bad = (
            flat_a.view(torch.uint8).view(len(flat_a), -1)
            != flat_b.view(torch.uint8).view(len(flat_b), -1)
        ).any(-1)
        positions = bad.nonzero().flatten()
        if positions.numel() == 0:
            transients += 1  # in-memory corruption during this run, not on disk
            continue
        detail = ", ".join(
            f"elem {i}: {bit_pattern(flat_b[i : i + 1])} -> {bit_pattern(flat_a[i : i + 1])}"
            for i in positions[:5].tolist()
        )
        findings.append(
            f"[passthrough] {name}: {positions.numel()} differing element(s) "
            f"vs source ({detail})"
        )
    return checked, zero_centered, transients


def check_scalar_scales(quant, findings) -> int:
    suffixes = (".weight_global_scale", ".input_global_scale", ".k_scale", ".v_scale")
    checked = 0
    for name in quant.index:
        if not name.endswith(suffixes):
            continue
        checked += 1
        value = quant(name).float()
        if not torch.isfinite(value).all() or (value <= 0).any():
            findings.append(f"[scalar-scale] {name}: {value.flatten().tolist()}")
    return checked


def check_scale_outliers(quant, findings) -> int:
    checked = 0
    for name in quant.index:
        if not name.endswith(".weight_scale"):
            continue
        checked += 1
        scale = quant(name).float().abs()
        if not torch.isfinite(scale).all():
            findings.append(
                f"[scale-outlier] {name}: "
                f"{(~torch.isfinite(scale)).sum().item()} non-finite entries"
            )
            continue
        positive = scale[scale > 0]
        if positive.numel() == 0:
            findings.append(f"[scale-outlier] {name}: all scales are zero")
            continue
        median = positive.median()
        outliers = scale > median * SCALE_OUTLIER_RATIO
        if outliers.any():
            findings.append(
                f"[scale-outlier] {name}: {outliers.sum().item()} scale(s) "
                f"exceed {SCALE_OUTLIER_RATIO:.0f}x the median "
                f"(median {median.item():.3e}, max {scale.max().item():.3e})"
            )
    return checked


def group_thresholds(config: dict) -> list[tuple[re.Pattern | str, float]]:
    """Per-target deviation thresholds from the checkpoint's config groups."""
    thresholds = []
    groups = config.get("quantization_config", {}).get("config_groups", {})
    for group in groups.values():
        weights = group.get("weights") or {}
        gptq_like = "mse" in (weights.get("observer") or "") or weights.get("actorder")
        limit = DEV_THRESHOLD_GPTQ if gptq_like else DEV_THRESHOLD_MINMAX
        for target in group.get("targets", []):
            if target.startswith("re:"):
                thresholds.append((re.compile(target[3:]), limit))
            else:
                thresholds.append((target, limit))
    return thresholds


def threshold_for(module: str, thresholds) -> float:
    for target, limit in thresholds:
        if isinstance(target, re.Pattern):
            if target.match(module):
                return limit
        elif target == module:
            return limit
    return DEV_THRESHOLD_GPTQ  # unmatched: be conservative about flagging


def report_deviations(module, dev, deq, orig, limit, findings, confirm) -> int:
    """Report elements beyond the threshold; returns discarded transients.

    A deviation can be fabricated by a RAM flip striking this script's own
    buffers or temporaries — the fault this tool hunts also lives in the
    machine running it. Before reporting, every input is re-read from disk
    and the deviation recomputed via `confirm`; only reproducible
    deviations are real. The discard count doubles as a live RAM-fault
    canary, surfaced in the run summary.
    """
    flagged = (dev > limit).nonzero()
    if len(flagged) == 0:
        return 0
    dev, deq, orig = confirm()
    confirmed = (dev > limit).nonzero()
    discarded = len(flagged) - len(confirmed)
    if len(confirmed) == 0:
        return discarded
    detail = ", ".join(
        f"[{r},{c}] dev {dev[r, c].item():.1f} units "
        f"(orig {orig[r, c].item():.4f}, deq {deq[r, c].item():.4f})"
        for r, c in confirmed[:5].tolist()
    )
    findings.append(
        f"[deviation] {module}: {len(confirmed)} element(s) beyond "
        f"{limit:.1f} scale-units ({detail})"
    )
    return max(discarded, 0)


def load_nvfp4(module, quant, orig):
    """Dequantize an NVFP4 module and return (dev, deq, source)."""
    source = orig(f"{module}.weight").float()
    rows, cols = source.shape
    packed = quant(f"{module}.weight_packed")
    scale = quant(f"{module}.weight_scale").float()
    gscale = quant(f"{module}.weight_global_scale").float()
    codes = unpack_fp4_from_uint8(packed, rows, cols, dtype=torch.float32)
    unit = (scale / gscale).repeat_interleave(16, dim=1)
    deq = codes * unit
    dev = (deq - source).abs() / unit.clamp_min(1e-12)
    return dev, deq, source


def load_fp8(module, quant, orig):
    """Dequantize an FP8 module and return (dev, deq, source).

    Relative error, with the denominator floored at the smallest normal
    E4M3 magnitude so near-zero weights don't read as noise; rescaled so
    DEV_THRESHOLD_MINMAX corresponds to rel > FP8_REL_THRESHOLD.
    """
    source = orig(f"{module}.weight").float()
    weight = quant(f"{module}.weight").float()
    scale = quant(f"{module}.weight_scale").float()
    deq = weight * scale
    floor = (scale * 2**-5).expand_as(source)
    rel = (deq - source).abs() / torch.maximum(source.abs(), floor)
    return rel / FP8_REL_THRESHOLD * DEV_THRESHOLD_MINMAX, deq, source


def check_deviation(quant, orig, config, findings) -> tuple[int, int]:
    thresholds = group_thresholds(config)
    modules_nvfp4 = sorted(
        name.removesuffix(".weight_packed")
        for name in quant.index
        if name.endswith(".weight_packed")
    )
    modules_fp8 = sorted(
        name.removesuffix(".weight")
        for name in quant.index
        if name.endswith(".weight") and quant(name).dtype == torch.float8_e4m3fn
    )
    checked = 0
    transients = 0

    for module in modules_nvfp4:
        if f"{module}.weight" not in orig.index:
            continue
        checked += 1
        dev, deq, source = load_nvfp4(module, quant, orig)
        if dev.median() > 0.75:
            findings.append(
                f"[deviation] {module}: median deviation "
                f"{dev.median().item():.2f} units — decode math or scale "
                f"layout mismatch, per-element flags unreliable"
            )
            continue
        transients += report_deviations(
            module,
            dev,
            deq,
            source,
            threshold_for(module, thresholds),
            findings,
            confirm=lambda m=module: load_nvfp4(m, quant, orig),
        )

    for module in modules_fp8:
        if f"{module}.weight" not in orig.index:
            continue
        checked += 1
        dev, deq, source = load_fp8(module, quant, orig)
        transients += report_deviations(
            module,
            dev,
            deq,
            source,
            DEV_THRESHOLD_MINMAX,
            findings,
            confirm=lambda m=module: load_fp8(m, quant, orig),
        )

    return checked, transients


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", required=True, help="Quantized checkpoint directory to verify"
    )
    parser.add_argument(
        "--orig",
        default=None,
        help="Source checkpoint (path or HuggingFace ID) the "
        "quantization ran from; enables the byte-identity and "
        "weight-deviation checks",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Skip the weight-deviation scan (the slow check)",
    )
    args = parser.parse_args()

    model_dir = Path(args.model).expanduser()
    quant = TensorLoader(load_index(model_dir))
    config = json.loads((model_dir / "config.json").read_text())
    findings: list[str] = []

    checked = check_scalar_scales(quant, findings)
    print(f"scalar scales:   {checked} checked")
    checked = check_scale_outliers(quant, findings)
    print(f"scale outliers:  {checked} tensors scanned")

    transients = 0
    if args.orig:
        orig = TensorLoader(load_index(resolve_dir(args.orig)))
        checked, zero_centered, hits = check_passthrough(quant, orig, findings)
        transients += hits
        print(
            f"pass-through:    {checked} tensors byte-compared"
            + (
                f" ({zero_centered} benign zero-centered norm round-trips)"
                if zero_centered
                else ""
            )
        )
        if not args.quick:
            checked, hits = check_deviation(quant, orig, config, findings)
            transients += hits
            print(f"deviation scan:  {checked} quantized modules dequantized")
    else:
        print("no --orig given: skipping byte-identity and deviation checks")

    if transients:
        print(
            f"WARNING: {transients} transient in-memory corruption event(s) "
            "discarded — this machine's RAM flipped bits DURING this run. "
            "The checkpoint on disk is unaffected, but the hardware is not "
            "to be trusted."
        )

    print()
    if findings:
        print(f"{len(findings)} finding(s):")
        for finding in findings:
            print(f"  {finding}")
        sys.exit(1)
    print("clean: no corruption detected")


if __name__ == "__main__":
    main()
