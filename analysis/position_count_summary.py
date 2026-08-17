#!/usr/bin/env python3
"""Pool the two position-count runs and test the mechanism across the grid.

position_count_mechanism.py runs one repo per process (config.py:18 makes
data_path cwd-relative), so the ayushi cells (verse, summarization) and the umang
cells (bias, factual recall, persona) land in separate CSVs. This merges them and
runs the paired tests.

THE PREDICTION
--------------
If long-form's advantage comes from integrating the objective over many response
tokens, then truncating that objective to T tokens should walk the field back
toward single-token localization as T falls:

    capture@k    should FALL      (less concentrated with less supervision)
    mean layer   should RISE      (later, like single-token, which sits +3.5 deeper)
    rho vs LF    should FALL      (further from real long-form)

Each is tested paired within cell, T=1 against T=all, so model and task cancel.
Monotonicity across T in {1,4,16,all} is checked separately -- a real dose-response
should be ordered, not just different at the endpoints.

WHAT WOULD FALSIFY IT
---------------------
If T=1 leaves capture and depth where T=all had them, position count is not the
mechanism and the difference lies in the data/format change the two arms also
carry (MCQA letter responses vs prose), which this design deliberately holds
fixed.

Cells appear here only if their T=all field reproduced the on-disk long-form
field (the validation gate in position_count_mechanism.py); withheld cells are
absent by construction, and the count is reported so partial coverage is visible.
"""
import csv
import statistics as st
from collections import defaultdict
from math import comb
from pathlib import Path

import sys

import numpy as np

HERE = Path(__file__).parent
SHORT = {"gemma-3-12b-it": "Gemma-3-12B", "Qwen1.5-14B-Chat": "Qwen1.5-14B",
         "Falcon3-10B-Instruct": "Falcon3-10B", "OLMo-2-1124-13B-DPO": "OLMo-2-13B",
         "Qwen1.5-32B-Chat": "Qwen1.5-32B"}
ORDER = ["1", "4", "16", "all"]


def rank_(x):
    x = np.asarray(x, float)
    o = x.argsort()
    r = np.empty(len(x), float)
    r[o] = np.arange(len(x), dtype=float)
    for v in np.unique(x):
        m = x == v
        if m.sum() > 1:
            r[m] = r[m].mean()
    return r


def boot(d, n=10000, seed=0):
    if len(d) < 2:
        return float("nan"), float("nan")
    r = np.random.default_rng(seed)
    m = np.sort(r.choice(np.asarray(d, float), (n, len(d))).mean(1))
    return float(m[int(.025 * n)]), float(m[int(.975 * n)])


def sign_p(d):
    pos = sum(1 for x in d if x > 0)
    neg = sum(1 for x in d if x < 0)
    n = pos + neg
    if not n:
        return pos, neg, 1.0
    k = min(pos, neg)
    return pos, neg, min(sum(comb(n, i) for i in range(k + 1)) / 2 ** n * 2, 1.0)


def line(label, d, direction="higher"):
    if not d:
        print(f"  {label:<40}  no cells")
        return
    lo, hi = boot(d)
    pos, neg, p = sign_p(d)
    hits = neg if "lower" in direction else pos
    print(f"  {label:<40}n={len(d):>3}  {st.mean(d):+8.4f} [{lo:+.4f},{hi:+.4f}]"
          f"   {direction} {hits}/{pos+neg}   p={p:.4f}")


def main():
    rows = []
    for name in ("position_count.csv", "position_count_umang.csv"):
        p = HERE / name
        if p.exists():
            rows += list(csv.DictReader(open(p)))
        else:
            print(f"  (missing {name})")
    if not rows:
        print("no position-count output found")
        return
    for r in rows:
        for c in ("capture", "mean_layer", "jac_vs_LF", "jac_vs_ST",
                  "rho_vs_LF", "val_jaccard"):
            r[c] = float(r[c])

    idx = defaultdict(dict)
    for r in rows:
        idx[(r["model"], r["task"])][r["T"]] = r
    cells = {k: v for k, v in idx.items() if "1" in v and "all" in v}
    print(f"{len(cells)} validated cells | "
          f"{len({k[0] for k in cells})} models | "
          f"{len({k[1] for k in cells})} tasks")
    print(f"mean validation Jaccard (T=all vs disk long-form): "
          f"{st.mean(v['all']['val_jaccard'] for v in cells.values()):.3f}\n")

    print("=" * 100)
    print("1. POOLED BY T  (does the field walk back as supervision shrinks?)")
    print("=" * 100)
    print(f"  {'T':>6}{'n':>5}{'capture':>11}{'mean layer':>13}"
          f"{'jac vs LF':>12}{'jac vs ST':>12}{'rho vs LF':>12}")
    for T in ORDER:
        v = [c[T] for c in cells.values() if T in c]
        if not v:
            continue
        print(f"  {T:>6}{len(v):>5}{st.mean(x['capture'] for x in v):>11.4f}"
              f"{st.mean(x['mean_layer'] for x in v):>13.2f}"
              f"{st.mean(x['jac_vs_LF'] for x in v):>12.3f}"
              f"{st.mean(x['jac_vs_ST'] for x in v):>12.3f}"
              f"{st.mean(x['rho_vs_LF'] for x in v):>+12.3f}")

    print("\n" + "=" * 100)
    print("2. PAIRED T=1 vs T=all, within cell")
    print("=" * 100)
    line("capture,   T=all - T=1", [c["all"]["capture"] - c["1"]["capture"]
                                    for c in cells.values()],
         "T=all more concentrated")
    line("mean layer, T=1 - T=all", [c["1"]["mean_layer"] - c["all"]["mean_layer"]
                                     for c in cells.values()],
         "T=1 later")
    line("rho vs LF, T=all - T=1", [c["all"]["rho_vs_LF"] - c["1"]["rho_vs_LF"]
                                    for c in cells.values()],
         "T=all closer to LF")
    line("jac vs ST, T=1 - T=all", [c["1"]["jac_vs_ST"] - c["all"]["jac_vs_ST"]
                                    for c in cells.values()],
         "T=1 closer to ST")

    print("\n" + "=" * 100)
    print("3. IS IT MONOTONE IN T?  (a dose-response should be ordered)")
    print("=" * 100)
    for field, want in (("rho_vs_LF", "increasing"), ("jac_vs_LF", "increasing"),
                        ("capture", "increasing"), ("mean_layer", "decreasing")):
        mono = tot = 0
        for c in cells.values():
            seq = [c[T][field] for T in ORDER if T in c]
            if len(seq) < 3:
                continue
            tot += 1
            ok = (all(x <= y + 1e-12 for x, y in zip(seq, seq[1:]))
                  if want == "increasing"
                  else all(x >= y - 1e-12 for x, y in zip(seq, seq[1:])))
            mono += ok
        if tot:
            print(f"  {field:<14} {want:<12} monotone in {mono}/{tot} cells")

    print("\n" + "=" * 100)
    print("4. HOW MUCH OF THE REAL LF->ST GAP DOES TRUNCATION REPRODUCE?")
    print("   the real arms differ by +3.52 layers (single-token sits deeper)")
    print("=" * 100)
    d = [c["1"]["mean_layer"] - c["all"]["mean_layer"] for c in cells.values()]
    if d:
        print(f"  truncating to T=1 moves the mean layer {st.mean(d):+.2f}")
        print(f"  that is {100 * st.mean(d) / 3.52:.0f}% of the real gap")

    print("\n" + "=" * 100)
    print("PER-CELL  (T=1 vs T=all)")
    print("=" * 100)
    print(f"  {'model':<14}{'task':<16}{'val':>6}{'cap1':>8}{'capAll':>8}"
          f"{'lyr1':>7}{'lyrAll':>8}{'rho1':>7}{'rhoAll':>8}")
    for (m, t), c in sorted(cells.items()):
        print(f"  {SHORT.get(m, m):<14}{t:<16}{c['all']['val_jaccard']:>6.2f}"
              f"{c['1']['capture']:>8.3f}{c['all']['capture']:>8.3f}"
              f"{c['1']['mean_layer']:>7.1f}{c['all']['mean_layer']:>8.1f}"
              f"{c['1']['rho_vs_LF']:>+7.2f}{c['all']['rho_vs_LF']:>+8.2f}")

    print("\n" + "=" * 100)
    print("5. DOES THE DEPTH RESPONSE SCALE WITH THE REAL LF->ST GAP?")
    print("   reading cells one at a time suggested it does; this tests it.")
    print("   x = real gap  = meanlayer(real ST) - meanlayer(real LF)")
    print("   y = shift     = meanlayer(T=1)     - meanlayer(T=all)")
    print("=" * 100)
    try:
        sys.path.insert(0, str(HERE))
        from position_count_mechanism import load_fields, top_units
        xs, ys, lab = [], [], []
        by_model = {}
        for (m, t), c in sorted(cells.items()):
            if m not in by_model:
                by_model[m] = load_fields(m)
            F = by_model[m]
            if (t, "long") not in F or (t, "single") not in F:
                continue
            uL, uS = top_units(F[(t, "long")], 0.05), top_units(F[(t, "single")], 0.05)
            gap = (float(np.mean([l for l, _ in uS]))
                   - float(np.mean([l for l, _ in uL])))
            xs.append(gap)
            ys.append(c["1"]["mean_layer"] - c["all"]["mean_layer"])
            lab.append((SHORT.get(m, m), t))
        if len(xs) > 2:
            x, y = np.array(xs), np.array(ys)
            r = float(np.corrcoef(x, y)[0, 1]) if x.std() and y.std() else float("nan")
            rs = float(np.corrcoef(rank_(x), rank_(y))[0, 1]) if x.std() and y.std() else float("nan")
            slope = float(np.polyfit(x, y, 1)[0]) if x.std() else float("nan")
            print(f"  {'model':<14}{'task':<16}{'real gap':>10}{'shift':>9}")
            for (mm, tt), gx, gy in sorted(zip(lab, xs, ys), key=lambda z: -z[1]):
                print(f"  {mm:<14}{tt:<16}{gx:>+10.2f}{gy:>+9.2f}")
            print(f"\n  n={len(xs)}  Pearson r={r:+.3f}  Spearman rho={rs:+.3f}"
                  f"  slope={slope:+.2f}")
            print(f"  mean real gap {x.mean():+.2f}, mean shift {y.mean():+.2f} "
                  f"({100*y.mean()/x.mean():.0f}% of the gap)")
            print("\n  positive slope -> truncation reproduces more depth change in")
            print("  cells where the arms actually differ more in depth")
        else:
            print("  too few cells")
    except Exception as e:
        print(f"  could not compute: {type(e).__name__}: {e}")

    withheld = [k for k, v in idx.items() if k not in cells]
    if withheld:
        print(f"\n  {len(withheld)} cell(s) withheld by the validation gate: "
              f"{[(SHORT.get(m, m), t) for m, t in withheld]}")


if __name__ == "__main__":
    main()
