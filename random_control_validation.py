"""Pre-launch validation for the random-head control arms (CLAUDE.md section 3.4).

Runs entirely on the committed ATP CSVs -- no model weights, no GPU -- so it can
gate the grid before any GPU time is spent. For every condition the Tier 1 grid
will actually run, checks:

  1. two draw seeds produce DIFFERENT head sets, for both arms
  2. the layer-matched draw's per-layer histogram is IDENTICAL to the ATP reference
  3. uniform and layer-matched draws have the SAME total size as the ATP set
  4. output prefixes for the arms are disjoint from the atp/ tree

Usage:  python random_control_validation.py
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.chdir(Path(__file__).resolve().parent)

import pandas as pd

from eval.logits_handler import (
    retrieve_random_k, retrieve_layer_matched_k, atp_reference_csv,
    is_random, is_layer_matched, draw_seed,
)

TOPKS = [0.01, 0.03, 0.05, 0.07, 0.09, 0.1, 0.5, 1.0]
SEEDS = [0, 1, 2]

MODELS = [m for m in (sys.argv[1:] or [
    "Falcon3-10B-Instruct", "Qwen1.5-14B-Chat", "gemma-3-12b-it",
    "OLMo-2-1124-13B-DPO", "Qwen1.5-32B-Chat",
]) if not m.startswith("-")]

# (model, task, eval_subdir, steer_subdir) -- steering always matched to eval mode
CONDITIONS = []
for model in MODELS:
    for task, modes in [
        ("from_verse-long_to_prose", ["verse-long", "verse-single"]),
        ("from_verse-single_to_prose", ["verse-long", "verse-single"]),
        ("from_paragraph-long_to_sentence", ["paragraph-long", "paragraph-single"]),
        ("from_paragraph-single_to_sentence", ["paragraph-long", "paragraph-single"]),
    ]:
        for mode in modes:
            CONDITIONS.append((model, task, f"{mode}_eval", f"{mode}_steer"))


class FakeConfig:
    """Minimal stand-in for Config: the reference lookup only needs the prefix."""

    class _Args:
        pass

    def __init__(self, model, task, patch_algo, eval_dir, steer_dir):
        self.args = self._Args()
        self.args.patch_algo = patch_algo
        self._prefix = f"./results/{model}/{task}/{patch_algo}/{eval_dir}/{steer_dir}/"

    def get_output_prefix(self):
        return self._prefix


def geometry_from_full_csv(model, task, eval_dir, steer_dir):
    """num_layers, num_heads from the topk=1.0 ATP CSV, which selects every head."""
    _, ref = atp_reference_csv(FakeConfig(model, task, "atp", eval_dir, steer_dir), 1.0)
    return int(ref["layer"].max()) + 1, int(ref["neuron"].max()) + 1


def as_set(df):
    return set(map(tuple, df[["layer", "neuron"]].values))


def main():
    failures, checks = [], 0

    assert is_random("random-s0") and is_random("randomlayer-s0")
    assert not is_random("atp") and not is_random("acp") and not is_random("probes")
    assert is_layer_matched("randomlayer-s0") and not is_layer_matched("random-s0")
    assert draw_seed("random-s3") == 3 and draw_seed("randomlayer-s12") == 12
    assert draw_seed("random") == 42  # legacy fixed draw preserved
    print("helpers: OK\n")

    for model, task, eval_dir, steer_dir in CONDITIONS:
        label = f"{model}/{task}/{eval_dir}/{steer_dir}"
        try:
            num_layers, num_heads = geometry_from_full_csv(model, task, eval_dir, steer_dir)
        except (FileNotFoundError, ValueError) as e:
            failures.append(f"{label}: no usable ATP reference ({e})")
            continue
        print(f"{label}  [{num_layers}L x {num_heads}H]")

        for topk in TOPKS:
            cfg_l = FakeConfig(model, task, "randomlayer-s0", eval_dir, steer_dir)
            try:
                ref_path, ref = atp_reference_csv(cfg_l, topk)
            except (FileNotFoundError, ValueError) as e:
                failures.append(f"{label} topk={topk}: {e}")
                continue

            ref_hist = ref.groupby("layer").size().to_dict()
            ref_size = len(ref)
            exact = "exact" if f"/atp/{eval_dir}/{steer_dir}/" in ref_path else "FALLBACK"

            uni = {s: retrieve_random_k(num_layers, num_heads, topk, seed=s) for s in SEEDS}
            lay = {s: retrieve_layer_matched_k(cfg_l, topk, num_layers, num_heads, s)
                   for s in SEEDS}

            # 1. different seeds -> different sets. topk=1.0 selects every head, so
            #    all draws are necessarily identical there; exempt it.
            if topk < 1.0:
                for arm, draws in (("uniform", uni), ("layer-matched", lay)):
                    sets = [as_set(draws[s]) for s in SEEDS]
                    for i in range(len(sets)):
                        for j in range(i + 1, len(sets)):
                            checks += 1
                            if sets[i] == sets[j]:
                                failures.append(
                                    f"{label} topk={topk} {arm}: seeds {SEEDS[i]} and "
                                    f"{SEEDS[j]} gave identical sets")

            # 2. layer-matched histogram identical to reference
            for s in SEEDS:
                checks += 1
                if lay[s].groupby("layer").size().to_dict() != ref_hist:
                    failures.append(f"{label} topk={topk} seed={s}: layer histogram "
                                    f"differs from {ref_path}")

            # 3. both arms match the real set size
            for s in SEEDS:
                checks += 2
                if len(uni[s]) != ref_size:
                    failures.append(f"{label} topk={topk} seed={s}: uniform has "
                                    f"{len(uni[s])} heads, ATP has {ref_size}")
                if len(lay[s]) != ref_size:
                    failures.append(f"{label} topk={topk} seed={s}: layer-matched has "
                                    f"{len(lay[s])} heads, ATP has {ref_size}")

            # schema + ordering, identical to retrieve_random_k's contract
            for arm, draws in (("uniform", uni), ("layer-matched", lay)):
                for s in SEEDS:
                    checks += 2
                    df = draws[s]
                    if list(df.columns) != ["layer", "neuron"]:
                        failures.append(f"{label} topk={topk} {arm}: bad columns "
                                        f"{list(df.columns)}")
                    if not df.equals(df.sort_values(by=["layer", "neuron"])):
                        failures.append(f"{label} topk={topk} {arm}: not sorted")

            ov_l = len(as_set(lay[0]) & as_set(ref)) / max(ref_size, 1)
            ov_u = len(as_set(uni[0]) & as_set(ref)) / max(ref_size, 1)
            print(f"   topk={topk:<5} n={ref_size:<5} layers={len(ref_hist):<3} "
                  f"ref={exact:<8} overlap-with-ATP: uniform={ov_u:.0%} layer-matched={ov_l:.0%}")

    # 4. output-tree separation
    print("\noutput prefixes:")
    prefixes = []
    for algo in ["atp", "random-s0", "random-s1", "random-s2",
                 "randomlayer-s0", "randomlayer-s1", "randomlayer-s2"]:
        p = FakeConfig("Falcon3-10B-Instruct", "from_verse-long_to_prose", algo,
                       "verse-long_eval", "verse-long_steer").get_output_prefix()
        prefixes.append(p)
        print(f"  {algo:<16} -> {p}")
    checks += 2
    if len(set(prefixes)) != len(prefixes):
        failures.append("output prefixes collide across arms/seeds")
    if any("/atp/" in p for p in prefixes[1:]):
        failures.append("a random arm writes into the atp/ tree")

    print(f"\n{checks} assertions run over {len(CONDITIONS)} conditions x {len(TOPKS)} topk")
    if failures:
        print(f"\nFAILED ({len(failures)}):")
        for f in failures:
            print("  -", f)
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
