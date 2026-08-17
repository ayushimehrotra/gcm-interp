#!/usr/bin/env python3
"""Every overlap / similarity number for free-form vs single-token localization.

Pulls from the analysis CSVs and reports mean, bootstrap 95% CI, range and n for
each measure, plus a per-cell table. Falcon3-10B is excluded throughout (only 3
of 5 tasks localized), matching the figures.

WHAT EACH MEASURE ANSWERS
-------------------------
head Jaccard      do the two methods pick the SAME attention heads?
layer Jaccard     do they at least pick heads in the same LAYERS?
Spearman/cosine   do their full attribution fields agree in ordering/direction?
SVCCA             do the heads they pick carry the same INFORMATION?
containment       is one method's ranking a superset of the other's?
recall(m)         how much of one method's top-k sits in the other's top-m?

The set measures (Jaccard, recall) and the field measures (Spearman, cosine) are
computed on the SIGNED attribution order the pipeline selects with; SVCCA is
computed on activations, so it is invariant to which coordinates each arm owns
and is the only measure here that can see through disjoint selection.
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
        print(f"  (missing {name})")
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
    geo = read("activation_geometry.csv")
    con = read("containment_asymmetry.csv")

    print("=" * 104)
    print("1. DO THEY PICK THE SAME ATTENTION HEADS?   (selection overlap, k=0.05)")
    print("=" * 104)
    line("head Jaccard", [float(r["unit_jaccard"]) for r in sim])
    line("layer Jaccard", [float(r["layer_jaccard"]) for r in sim])
    if con:
        line("head Jaccard (independent recompute)",
             [float(r["jaccard"]) for r in con])

    print("\n" + "=" * 104)
    print("2. DO THEIR ATTRIBUTION FIELDS AGREE?   (all heads, signed order)")
    print("=" * 104)
    line("Spearman over all heads", [float(r["spearman"]) for r in sim])
    line("cosine between fields", [float(r["cosine"]) for r in sim])

    print("\n" + "=" * 104)
    print("3. DO THE HEADS THEY PICK CARRY THE SAME INFORMATION?   (SVCCA)")
    print("=" * 104)
    line("top canonical correlation", [float(r["rho1"]) for r in geo], "{:.4f}")
    line("mean of top-5", [float(r["rho_mean"]) for r in geo], "{:.4f}")
    line("row-shuffled floor (estimator inflation)",
         [float(r["rho_perm"]) for r in geo], "{:.4f}")
    line("EXCESS over shuffle", [float(r["rho_excess"]) for r in geo], "{:.4f}")
    print()
    line("canonical variate vs concept, free-form",
         [float(r["lab_L"]) for r in geo], "{:.4f}")
    line("canonical variate vs concept, single-token",
         [float(r["lab_S"]) for r in geo], "{:.4f}")

    print("\n" + "=" * 104)
    print("4. IS ONE RANKING CONTAINED IN THE OTHER?   (0.5 = chance)")
    print("=" * 104)
    line("free-form's picks, percentile in ST's field",
         [float(r["pct_L_in_S"]) for r in con], "{:.4f}")
    line("single-token's picks, percentile in FF's field",
         [float(r["pct_S_in_L"]) for r in con], "{:.4f}")
    print()
    line("free-form's picks, AUC under ST's scores",
         [float(r["auc_L_in_S"]) for r in con], "{:.4f}")
    line("single-token's picks, AUC under FF's scores",
         [float(r["auc_S_in_L"]) for r in con], "{:.4f}")
    line("asymmetry  (FF-in-ST) - (ST-in-FF)",
         [float(r["asymmetry"]) for r in con])

    print("\n" + "=" * 104)
    print("5. RECALL CURVES: share of one arm's top-5% inside the other's top-m")
    print("   chance recall at m is m itself")
    print("=" * 104)
    print(f"  {'m':>6}{'FF picks in ST top-m':>24}{'ST picks in FF top-m':>24}"
          f"{'chance':>10}")
    for m in (0.05, 0.1, 0.2, 0.3, 0.5):
        a = [float(r[f"recall_L_in_S{m}"]) for r in con
             if f"recall_L_in_S{m}" in r]
        b = [float(r[f"recall_S_in_L{m}"]) for r in con
             if f"recall_S_in_L{m}" in r]
        if a and b:
            print(f"  {m:>6}{st.mean(a):>24.3f}{st.mean(b):>24.3f}{m:>10.2f}")

    print("\n" + "=" * 104)
    print("6. GEOMETRY OF THE SELECTED HEADS")
    print("=" * 104)
    line("concept-energy density, free-form",
         [float(r["density_L"]) for r in geo], "{:.4f}")
    line("concept-energy density, single-token",
         [float(r["density_S"]) for r in geo], "{:.4f}")
    line("paired density, FF - ST",
         [float(r["density_L"]) - float(r["density_S"]) for r in geo])

    print("\n" + "=" * 104)
    print("PER-CELL")
    print("=" * 104)
    idx = {(r["model"], r["task"]): r for r in sim}
    g = {(r["model"], r["task"]): r for r in geo}
    c = {(r["model"], r["task"]): r for r in con}
    keys = sorted(set(idx) & set(g) & set(c))
    print(f"  {'model':<13}{'task':<16}{'headJac':>9}{'layerJac':>10}"
          f"{'spearman':>10}{'rho5':>8}{'shuffle':>9}{'excess':>8}"
          f"{'aucFF>ST':>10}{'aucST>FF':>10}")
    for k in keys:
        print(f"  {SHORT.get(k[0], k[0]):<13}{k[1]:<16}"
              f"{float(idx[k]['unit_jaccard']):>9.3f}"
              f"{float(idx[k]['layer_jaccard']):>10.3f}"
              f"{float(idx[k]['spearman']):>10.3f}"
              f"{float(g[k]['rho_mean']):>8.3f}"
              f"{float(g[k]['rho_perm']):>9.3f}"
              f"{float(g[k]['rho_excess']):>8.3f}"
              f"{float(c[k]['auc_L_in_S']):>10.3f}"
              f"{float(c[k]['auc_S_in_L']):>10.3f}")
    print(f"\n  {len(keys)} model-task pairs (Falcon3-10B excluded)")


if __name__ == "__main__":
    main()
