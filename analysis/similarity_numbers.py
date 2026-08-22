#!/usr/bin/env python3
"""Do the two localizations pick the same attention heads?

Selection overlap between free-form and single-token localization at k=0.05,
over 19 model-task pairs. Falcon3-10B is excluded throughout (only 3 of 5 tasks
localized), matching the figures.

    head Jaccard     do the two methods pick the SAME heads?
    layer Jaccard    do they at least pick heads in the same LAYERS?

Both are computed on the SIGNED attribution order the pipeline actually selects
with (eval/logits_handler.py:92 uses flat.topk), not on magnitudes.

Reads localization_similarity.csv, written by localization_concentration.py.

WHY THE HEAD/LAYER SPLIT IS THE WHOLE POINT
-------------------------------------------
The two numbers disagree sharply and are meant to be read together: head Jaccard
~0.07 says the arms select almost entirely disjoint HEADS, while layer Jaccard
~0.54 says they largely agree on which LAYERS those heads live in. The
disagreement is the finding -- same neighbourhood, different occupants -- and
either number alone misreports it.

Keep the near-zero head Jaccard in view when reading any other comparison of
these two arms: ~93% disjoint selection pins every direct geometric measure
(cosine, principal angles, eigenvector overlap) near zero BY CONSTRUCTION, so a
zero there measures disjointness rather than representation. FINDINGS.md 10.3.

WHAT THIS SCRIPT USED TO ALSO REPORT
------------------------------------
Five further sections (attribution-field agreement, SVCCA, containment, recall
curves, concept-energy density) and a per-cell table spanning them. They were
removed with the analyses behind them: their inputs were activation_geometry.csv
and containment_asymmetry.csv, whose producing scripts no longer exist. The CSVs
are still on disk, so those sections would still have PRINTED -- with numbers
frozen at their last computed values and no way to refresh them. The one
survivor of that group, a "head Jaccard (independent recompute)" cross-check
read from containment_asymmetry.csv, went for the same reason. Everything below
now traces to a CSV that can still be regenerated.

Usage:  python similarity_numbers.py
"""
import csv
import statistics as st
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
EXCLUDED = {"Falcon3-10B-Instruct"}
SHORT = {"gemma-3-12b-it": "Gemma-3-12B", "Qwen1.5-14B-Chat": "Qwen1.5-14B",
         "Qwen1.5-32B-Chat": "Qwen1.5-32B", "OLMo-2-1124-13B-DPO": "OLMo-2-13B"}


def read(name):
    p = HERE / name
    if not p.exists():
        print(f"  (missing {name} -- regenerate with localization_concentration.py)")
        return []
    return [r for r in csv.DictReader(open(p)) if r.get("model") not in EXCLUDED]


def boot(d, n=10000, seed=0):
    if len(d) < 2:
        return float("nan"), float("nan")
    r = np.random.default_rng(seed)
    m = np.sort(r.choice(np.asarray(d, float), (n, len(d))).mean(1))
    return float(m[int(.025 * n)]), float(m[int(.975 * n)])


def line(label, vals, fmt="{:+.4f}"):
    if not vals:
        print(f"  {label:<44}  no data")
        return
    lo, hi = boot(vals)
    print(f"  {label:<44}n={len(vals):>3}  {fmt.format(st.mean(vals))} "
          f"[{lo:+.4f},{hi:+.4f}]   min {min(vals):+.3f}  max {max(vals):+.3f}")


def main():
    sim = read("localization_similarity.csv")

    print("=" * 104)
    print("DO THEY PICK THE SAME ATTENTION HEADS?   (selection overlap, k=0.05)")
    print("=" * 104)
    line("head Jaccard", [float(r["unit_jaccard"]) for r in sim])
    line("layer Jaccard", [float(r["layer_jaccard"]) for r in sim])

    print("\n" + "=" * 104)
    print("PER-CELL")
    print("=" * 104)
    print(f"  {'model':<13}{'task':<16}{'headJac':>9}{'layerJac':>10}")
    for r in sorted(sim, key=lambda z: (z["model"], z["task"])):
        print(f"  {SHORT.get(r['model'], r['model']):<13}{r['task']:<16}"
              f"{float(r['unit_jaccard']):>9.3f}{float(r['layer_jaccard']):>10.3f}")
    print(f"\n  {len(sim)} model-task pairs (Falcon3-10B excluded)")


if __name__ == "__main__":
    main()
