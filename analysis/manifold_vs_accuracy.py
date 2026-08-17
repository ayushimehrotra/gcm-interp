#!/usr/bin/env python3
"""Does the concept manifold's existence predict how well a cell actually works?

activation_geometry.py established that LF and ST localize to the same manifold
(SVCCA excess 0.660 over shuffle, 21/21 cells) and measured how strongly that
shared subspace carries the concept (canonical variate vs desired/undesired
label: 0.402 LF, 0.397 ST). That second number varies enormously by cell --
0.79 on Qwen1.5-32B verse down to 0.05 on Gemma bias.

If those geometry numbers are measuring something real rather than noise, the
cells with a strong concept manifold should be the cells where steering the
localized units actually produces the behaviour. That is testable: the judged
long-form accuracies are already on disk.

    corr( concept-bearingness of the shared subspace ,  behavioural accuracy )

A positive correlation means the geometry is behaviourally grounded, and cells
where localization underperforms are cells with a weak concept manifold. A flat
one means the geometry is decoupled from what the intervention achieves -- which
would undercut reading the manifold result as an explanation of anything.

MEASURES
--------
geometry, per (model, task), from activation_geometry.csv
  lab_L, lab_S   |corr| between an arm's leading canonical variates and the
                 desired/undesired label -- how concept-bearing the shared
                 subspace is for that arm
  rho_excess     SVCCA excess over the row-shuffle baseline -- how much shared
                 information exists at all, concept or otherwise
  density_L/S    concept energy per unit coordinate share

behaviour, per (model, task, arm), harvested from the judged evaluations
  peak           best accuracy the arm reaches at any budget -- the cell's
                 ceiling, i.e. does this localization work at all
  acc@k          accuracy at a fixed budget

Long-form eval only: the claim is about free-form generation, so the
single-token eval is not the target behaviour. atp arms only, random-control
arms excluded, superseded (_old) trees excluded.

CAUTION ON MULTIPLICITY
-----------------------
Several geometry measures are tested against several accuracy measures. With
~20 cells that is a lot of correlations for the number of points, so a single
p<0.05 among them is not evidence of much. The pre-registered question is the
first one -- concept-bearingness vs peak accuracy -- and the rest are reported
for completeness, not as independent tests.
"""
import argparse
import csv
import json
import re
import statistics as st
from collections import defaultdict
from math import sqrt
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
AYUSHI = Path("/home/ubuntu/gcm-interp")
UMANG = Path("/home/ubuntu/gcm-interp-umang")
# Priority-ordered per repo: a tree is read from the first root that has it, and
# later roots only fill gaps. 'results_pipeline_with_answers' is where the
# persona and bias cells live; omitting it silently dropped 6 of 21 cells from
# the join, which is what this ordering fixes. Mirrors figures.py -- keep the two
# in step, since the duplication here is what let them drift apart.
ACC_ROOTS = {"ayushi": [AYUSHI / "judge-evals" / "accuracy"],
             "umang": [UMANG / "results_pipeline",
                       UMANG / "results_pipeline_with_answers"]}
TASKNAME = {"verse": "verse", "paragraph": "summarization", "female": "bias",
            "lying": "factual recall", "extraversion": "persona"}
SHORT = {"gemma-3-12b-it": "Gemma-3-12B", "Qwen1.5-14B-Chat": "Qwen1.5-14B",
         "Falcon3-10B-Instruct": "Falcon3-10B", "OLMo-2-1124-13B-DPO": "OLMo-2-13B",
         "Qwen1.5-32B-Chat": "Qwen1.5-32B"}
FN_RE = re.compile(
    r"^(?P<N>\d+)_(?P<reps>random|targeted)_(?P<method>steer|mean)_topk_"
    r"(?P<topk>[\d.]+)_gen_accuracy_"
    r"(?P<metric>w_rf|wo_rf|comb|flu|rel|judge_3|judge_4|judge_5|mcqa)"
    r"\.json\.accuracy\.json$")
MODE_RE = re.compile(r"^(?P<task>.+)-(?P<mode>long|single)$")
MET = {("ayushi", "long"): "w_rf", ("ayushi", "single"): "w_rf",
       ("umang", "long"): "comb", ("umang", "single"): "w_rf"}
OLD = {"Llama-2-13b-chat-hf", "SOLAR-10.7B-Instruct-v1.0", "vicuna-13b-v1.5",
       "phi-4"}


def strip_mode(s):
    m = MODE_RE.match(s)
    return (m.group("task"), m.group("mode")) if m else (s, None)


def harvest():
    """(model, task, arm) -> {budget k: accuracy}, long-form eval, atp only."""
    # bucket[(model, task, arm)][k][(priority, tree)] -> [values over N]
    g = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for repo, roots in ACC_ROOTS.items():
        for prio, root in enumerate(roots):
            if not root.exists():
                continue
            _harvest_root(g, repo, prio, root)
    # Take the highest-priority root that supplied anything for this (key, k),
    # then the max over steering factors within it.
    out = {}
    for key, budgets in g.items():
        out[key] = {}
        for k, bucket in budgets.items():
            top = min(p for p, _ in bucket)
            out[key][k] = max(v for (p, _), vs in bucket.items() if p == top
                              for v in vs)
    return out


def _harvest_root(g, repo, prio, root):
        for p in root.rglob("*.accuracy.json"):
            parts = p.relative_to(root).parts
            m = FN_RE.match(parts[-1])
            if not m or any(c.startswith("random") for c in parts) \
                    or "atp" not in parts:
                continue
            if m.group("reps") != "targeted" or m.group("method") != "steer":
                continue
            srcbase = evald = steerd = model = None
            for i, c in enumerate(parts[:-1]):
                if c.startswith("from_") and "_to_" in c:
                    srcbase, model = c, parts[i - 1] if i else None
                elif c.endswith("_eval"):
                    evald = c
                elif c.endswith("_steer"):
                    steerd = c
            if not (srcbase and evald and steerd and model):
                continue
            if srcbase.endswith("_old") or model in OLD:
                continue
            sb = re.match(r"^from_(?P<src>.+?)_to_(?P<base>.+)$", srcbase)
            if not sb:
                continue
            task, arm = strip_mode(sb.group("src"))
            task = TASKNAME.get(task)
            _, ev = strip_mode(evald[:-5])
            _, stm = strip_mode(steerd[:-6])
            if task is None or arm is None or ev != "long" or stm != ev:
                continue
            if MET.get((repo, ev)) != m.group("metric"):
                continue
            try:
                v = json.load(open(p)).get("q1")
            except Exception:
                continue
            if v is not None:
                tree = str(p.parent)
                g[(model, task, arm)][float(m.group("topk"))][(prio, tree)] \
                    .append(float(v))


def rank(x):
    x = np.asarray(x, float)
    o = x.argsort()
    r = np.empty(len(x), float)
    r[o] = np.arange(len(x), dtype=float)
    for v in np.unique(x):
        m = x == v
        if m.sum() > 1:
            r[m] = r[m].mean()
    return r


def corr(x, y):
    """Pearson, Spearman, and a two-sided p for Pearson."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or x.std() == 0 or y.std() == 0:
        return float("nan"), float("nan"), float("nan")
    r = float(np.corrcoef(x, y)[0, 1])
    rs = float(np.corrcoef(rank(x), rank(y))[0, 1])
    n = len(x)
    if abs(r) >= 1:
        return r, rs, 0.0
    t = abs(r) * sqrt((n - 2) / (1 - r * r))
    # two-sided p from the t distribution, normal approx refined for small n
    try:
        from statistics import NormalDist
        df = n - 2
        z = t * (1 - 1 / (4 * df)) / sqrt(1 + t * t / (2 * df))
        p = 2 * (1 - NormalDist().cdf(z))
    except Exception:
        p = float("nan")
    return r, rs, p


def show(label, x, y, n_note=""):
    r, rs, p = corr(x, y)
    print(f"  {label:<44}n={len(x):>3}  r={r:+.3f}  rho={rs:+.3f}  p={p:.3f}{n_note}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geom", default="activation_geometry.csv")
    ap.add_argument("--k", type=float, default=0.05)
    a = ap.parse_args()

    gpath = HERE / a.geom
    if not gpath.exists():
        print(f"{gpath} missing -- run activation_geometry.py first")
        return
    geom = {(r["model"], r["task"]): {k: float(v) for k, v in r.items()
                                      if k not in ("model", "task")}
            for r in csv.DictReader(open(gpath))}
    acc = harvest()

    cells = []
    for (model, task), g in sorted(geom.items()):
        cL, cS = acc.get((model, task, "long")), acc.get((model, task, "single"))
        if not cL or not cS:
            continue
        row = dict(model=model, task=task, **g)
        row["peak_L"], row["peak_S"] = max(cL.values()), max(cS.values())
        row["peak"] = (row["peak_L"] + row["peak_S"]) / 2
        row["at_k_L"] = cL.get(a.k, float("nan"))
        row["at_k_S"] = cS.get(a.k, float("nan"))
        row["lab"] = (g["lab_L"] + g["lab_S"]) / 2
        cells.append(row)

    if len(cells) < 3:
        print(f"only {len(cells)} cells have both geometry and long-form accuracy")
        return

    print(f"{len(cells)} cells with BOTH activation geometry and judged "
          f"long-form accuracy\n")
    print("=" * 100)
    print("PER-CELL")
    print("=" * 100)
    print(f"  {'model':<14}{'task':<16}{'lab_L':>7}{'lab_S':>7}{'rho_exc':>9}"
          f"{'peak_L':>8}{'peak_S':>8}{'densL':>7}{'densS':>7}")
    for c in sorted(cells, key=lambda c: -c["lab"]):
        print(f"  {SHORT[c['model']]:<14}{c['task']:<16}{c['lab_L']:>7.2f}"
              f"{c['lab_S']:>7.2f}{c['rho_excess']:>9.3f}{c['peak_L']:>8.3f}"
              f"{c['peak_S']:>8.3f}{c['density_L']:>7.2f}{c['density_S']:>7.2f}")

    print("\n" + "=" * 100)
    print("1. PRIMARY: does a concept-bearing shared subspace predict accuracy?")
    print("=" * 100)
    show("concept-bearingness vs peak accuracy",
         [c["lab"] for c in cells], [c["peak"] for c in cells])
    show("  long-form arm only (lab_L vs peak_L)",
         [c["lab_L"] for c in cells], [c["peak_L"] for c in cells])
    show("  single-token arm only (lab_S vs peak_S)",
         [c["lab_S"] for c in cells], [c["peak_S"] for c in cells])

    pooled_x = [c["lab_L"] for c in cells] + [c["lab_S"] for c in cells]
    pooled_y = [c["peak_L"] for c in cells] + [c["peak_S"] for c in cells]
    show("  both arms pooled", pooled_x, pooled_y,
         "   (arms not independent)")

    print("\n" + "=" * 100)
    print("2. SECONDARY (reported for completeness, not independent tests)")
    print("=" * 100)
    show("SVCCA excess vs peak accuracy",
         [c["rho_excess"] for c in cells], [c["peak"] for c in cells])
    show("concept density vs peak accuracy",
         [(c["density_L"] + c["density_S"]) / 2 for c in cells],
         [c["peak"] for c in cells])
    ok = [c for c in cells if not np.isnan(c["at_k_L"]) and not np.isnan(c["at_k_S"])]
    if len(ok) >= 3:
        show(f"concept-bearingness vs accuracy at k={a.k}",
             [c["lab"] for c in ok],
             [(c["at_k_L"] + c["at_k_S"]) / 2 for c in ok])

    print("\n" + "=" * 100)
    print("3. WITHIN-CELL: does the arm with the stronger manifold win?")
    print("   removes all between-cell variation (task difficulty, model, judge)")
    print("=" * 100)
    dx = [c["lab_L"] - c["lab_S"] for c in cells]
    dy = [c["peak_L"] - c["peak_S"] for c in cells]
    show("d(concept-bearingness) vs d(peak accuracy)", dx, dy)
    agree = sum(1 for i in range(len(dx))
                if (dx[i] > 0) == (dy[i] > 0) and dx[i] != 0 and dy[i] != 0)
    tot = sum(1 for i in range(len(dx)) if dx[i] != 0 and dy[i] != 0)
    print(f"  sign agreement: {agree}/{tot} cells")

    print("\n" + "=" * 100)
    print("READING")
    print("=" * 100)
    r, _, p = corr([c["lab"] for c in cells], [c["peak"] for c in cells])
    if not np.isnan(p) and p < 0.05 and r > 0:
        print("  Concept-bearing geometry tracks behavioural accuracy: the manifold")
        print("  measurement is behaviourally grounded, and weak cells are cells")
        print("  where the shared subspace does not carry the concept.")
    else:
        print("  No reliable link at this sample size. The geometry says the arms")
        print("  share a subspace; that fact does not by itself predict whether")
        print("  steering the cell works. Treat 'same manifold' as a statement")
        print("  about representation, NOT as an explanation of accuracy.")

    with open(HERE / "manifold_vs_accuracy.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(cells[0].keys()))
        w.writeheader()
        w.writerows(cells)
    print("\nwrote manifold_vs_accuracy.csv")


if __name__ == "__main__":
    main()
