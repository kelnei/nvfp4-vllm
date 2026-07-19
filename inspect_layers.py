"""
Inspect a model's architecture to find the Linear layers that quantize.py's
--ignore should exclude, without downloading any weights.

llm-compressor's QuantizationModifier matches --ignore patterns against the
*live* module hierarchy of the loaded transformers model, which for some
architectures (e.g. Gemma 4's vision/audio embedders) uses different names
than the on-disk checkpoint. Guessing --ignore from the checkpoint's
safetensors keys alone can miss those layers. This script instead builds the
model on a meta device (architecture only, ~instant, no download) through the
same AutoModelForCausalLM entry point quantize.py uses, and walks its real
named_modules(), so the names it prints are the ones QuantizationModifier
will see at quantization time.

Usage:
    python inspect_layers.py --model google/gemma-4-12B-it --trust-remote-code

    # For shell substitution: stdout is exactly the --ignore args, notes on stderr
    python quantize.py --model M --ignore $(python inspect_layers.py --model M --args-only)
"""

import argparse
import re
import sys

import torch
from transformers import AutoConfig, AutoModelForCausalLM

# Matches a numbered slot in a repeated block list, e.g.
# "model.language_model.layers.12.mlp.gate_proj" -> ".layers.12.".
# Linears inside the largest such stack are the decoder blocks NVFP4 should
# target; Linears in any *other* repeated stack (vision/audio encoder towers)
# and standalone Linears (embedders, projections, heads) are --ignore
# candidates.
REPEATED_BLOCK_RE = re.compile(r"(^|\.)(layers|blocks|h)\.\d+\.")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="HuggingFace model ID")
    p.add_argument("--trust-remote-code", action="store_true",
                   help="Trust remote code when loading the model config/class")
    p.add_argument("--args-only", action="store_true",
                   help="Print only the suggested --ignore args on stdout "
                        "(safe for shell substitution); notes go to stderr")
    return p.parse_args()


def main():
    args = parse_args()
    out = sys.stderr if args.args_only else sys.stdout

    print(f"Fetching config for {args.model}...", file=out)
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)

    print("Building model on meta device (architecture only, no weights downloaded)...", file=out)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=args.trust_remote_code)

    declared = (config.architectures or [None])[0]
    if declared and type(model).__name__ != declared:
        print(f"\nWARNING: checkpoint declares architecture {declared} but "
              f"AutoModelForCausalLM built {type(model).__name__}.\n"
              f"The suggestions below only cover layers that class actually has —\n"
              f"if it is a text-only variant, multimodal layers are missing from them.",
              file=out)

    num_experts = getattr(config.get_text_config(), "num_experts", None)

    stacks = {}      # repeated-block stack root -> [Linear names inside it]
    standalone = []  # Linear names outside any repeated block
    routers = []     # in-stack Linears that look like MoE routers
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        m = REPEATED_BLOCK_RE.search(name)
        if m is None:
            standalone.append(name)
        elif num_experts and module.out_features == num_experts:
            routers.append(name)
        else:
            stacks.setdefault(name[: m.start()], []).append(name)

    target_root = max(stacks, key=lambda r: len(stacks[r])) if stacks else None
    tower_roots = sorted(r for r in stacks if r != target_root)

    if target_root is not None:
        names = stacks[target_root]
        print(f"\n{len(names)} Linear layers in the main decoder stack under "
              f"'{target_root}' (quantization targets), e.g.:", file=out)
        for n in names[:3]:
            print(f"  {n}", file=out)

    ignore_args = sorted(standalone)

    if tower_roots:
        print(f"\nWARNING: found repeated-block stacks outside the main decoder — "
              f"assuming they are encoder towers and suggesting a re: pattern for "
              f"each. Verify '{target_root}' really is the text decoder:", file=out)
        for root in tower_roots:
            print(f"  {root} ({len(stacks[root])} Linear layers)", file=out)
        ignore_args += [f"re:{re.escape(root)}\\..*" for root in tower_roots]

    if routers:
        # One pattern per distinct per-layer suffix (e.g. "mlp.router") instead
        # of one entry per layer.
        suffixes = sorted({n[REPEATED_BLOCK_RE.search(n).end():] for n in routers})
        print(f"\n{len(routers)} in-stack Linear layers with out_features == "
              f"num_experts ({num_experts}) look like MoE routers:", file=out)
        for s in suffixes:
            print(f"  *.{s}", file=out)
        ignore_args += [f"re:.*\\.{re.escape(s)}$" for s in suffixes]

    print(f"\n{len(standalone)} standalone Linear layers (--ignore candidates):", file=out)
    for n in sorted(standalone):
        print(f"  {n}", file=out)

    print("\nSuggested --ignore args:", file=out)
    print(" ".join(ignore_args))


if __name__ == "__main__":
    main()
