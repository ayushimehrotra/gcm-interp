#!/usr/bin/env python3
"""Is "long-form localization requires fewer heads to induce a behaviour" true?

THE ESTIMAND
------------
For arm a in {L, S} let f_a(k) = max_N acc(a, N, k)  -- the accuracy achievable at
budget k once the steering strength is tuned, which is how the paper reads the
curve. For a behaviour threshold B define the required budget

    k*_a(B) = min { k in K : f_a(k) >= B },      K = {.01,.03,.05,.07,.09,.1,.5,1}
            = +inf if the arm never reaches B.

The claim is  E[ k*_L(B) - k*_S(B) ] < 0  for behaviour levels B one would
actually target.

WHY THE PUBLISHED VERSION CANNOT SUPPORT IT
-------------------------------------------
The published statistic sets B_a = 0.8 * max_k f_a(k) -- a DIFFERENT threshold per
arm. It therefore estimates E[k*_L(B_L) - k*_S(B_S)], which is not the claim: the
two arms are being asked to clear different bars. To test the claim, B must be the
same number for both arms. Two arm-independent choices are used here:

  (i)  absolute  B in {0.3 ... 0.8}
  (ii) B = 0.8 * pooled ceiling, pooled = max over BOTH arms in that cell

CENSORING
---------
If an arm never reaches B, k* is undefined, and dropping those cells biases the
comparison toward whichever arm fails more often. Both are reported:
  complete-case  -- cells where both arms reach B
  censored-worst -- k* mapped to grid rank, non-reachers assigned rank 8 (one past
                    the grid). Conservative: it can only help the arm that reaches.

SCALE
-----
k is compared on its GRID RANK (0..7), not its raw value, because the grid is
0.01,0.03,0.05,0.07,0.09,0.1,0.5,1.0 -- the last two steps are 5x and 2x the whole
preceding range, so a raw-scale mean is dominated by whichever cells happen to
cross there.

TEST
----
Paired by cell. Bootstrap CI over cells (10k resamples) on the mean difference,
plus an exact sign test. A two-one-sided-test equivalence bound is reported: the
largest d for which the data are consistent with |E[delta]| < d.
"""
import csv
import json
import re
import statistics as st
from collections import defaultdict
from math import comb
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Self-contained: the accuracy grid is harvested directly from the two repos, so
# this script depends on nothing but the checkouts themselves.
# ---------------------------------------------------------------------------
ROOTS = {"ayushi": Path("/home/ubuntu/gcm-interp/judge-evals/accuracy"),
         "umang": Path("/home/ubuntu/gcm-interp-umang/results_pipeline")}
FN_RE = re.compile(
    r"^(?P<N>\d+)_(?P<reps>random|targeted)_(?P<method>steer|mean)_topk_(?P<topk>[\d.]+)"
    r"_gen_accuracy_(?P<metric>w_rf|wo_rf|comb|flu|rel|judge_3|judge_4|judge_5|mcqa)"
    r"\.json\.accuracy\.json$")
MODE_RE = re.compile(r"^(?P<task>.+)-(?P<mode>long|single)$")
TASKNAME = {"verse": "verse", "paragraph": "summarization", "female": "bias",
            "lying": "factual recall", "extraversion": "persona"}
KS = [0.01, 0.03, 0.05, 0.07, 0.09, 0.1, 0.5, 1.0]
RANK = {k: i for i, k in enumerate(KS)}
CENSORED_RANK = len(KS)          # one past the grid
OLD = {"Llama-2-13b-chat-hf", "SOLAR-10.7B-Instruct-v1.0", "vicuna-13b-v1.5", "phi-4"}
# long-form evals are judged (+ fluency/relevance); single-token evals are token-matched
MET = {("ayushi", "long"): "w_rf", ("ayushi", "single"): "w_rf",
       ("umang", "long"): "comb", ("umang", "single"): "w_rf"}


def strip_mode(s):
    m = MODE_RE.match(s)
    return (m.group("task"), m.group("mode")) if m else (s, None)


def harvest():
    """Scored atp conditions from both repos -> (model, task, loc, eval) -> {(N,k): acc}.

    Two filters matter and are easy to omit:
      * the accuracy tree also holds the random-control arms (random-s*,
        randomlayer-s*) beside atp; mixing them in silently blends controls into
        the real results
      * *_old trees are superseded runs, not extra conditions
    """
    g = defaultdict(dict)
    for repo, root in ROOTS.items():
        if not root.exists():
            print(f"  WARNING: {root} missing")
            continue
        for p in root.rglob("*.accuracy.json"):
            parts = p.relative_to(root).parts
            m = FN_RE.match(parts[-1])
            if not m or any(c.startswith("random") for c in parts) or "atp" not in parts:
                continue
            srcbase = evald = steerd = model = None
            for i, c in enumerate(parts[:-1]):
                if c.startswith("from_") and "_to_" in c:
                    srcbase, model = c, parts[i - 1] if i else None
                elif c.endswith("_eval"):
                    evald = c
                elif c.endswith("_steer"):
                    steerd = c
            if not (srcbase and evald and steerd and model) or srcbase.endswith("_old"):
                continue
            if model in OLD:
                continue
            sb = re.match(r"^from_(?P<src>.+?)_to_(?P<base>.+)$", srcbase)
            task, loc = strip_mode(sb.group("src"))
            task = TASKNAME.get(task)
            _, ev = strip_mode(evald[:-5])
            _, stm = strip_mode(steerd[:-6])
            if task is None or loc is None or ev is None or stm is None:
                continue
            if stm != ev:                      # steering vector matched to eval mode
                continue
            if MET.get((repo, ev)) != m.group("metric"):
                continue
            try:
                v = json.load(open(p)).get("q1")
            except Exception:
                continue
            if v is not None:
                g[(model, task, loc, ev)][(int(m.group("N")), float(m.group("topk")))] = float(v)
    return g


def curves():
    """f_a(k) = max over the steering-strength sweep, as the paper reads the curve."""
    out = {}
    for key, d in harvest().items():
        c = defaultdict(list)
        for (N, k), v in d.items():
            c[k].append(v)
        out[key] = {k: max(v) for k, v in c.items()}
    return out


def kstar(f, B):
    for k in KS:
        if f.get(k, 0) >= B:
            return RANK[k]
    return None


def boot(ds, n=10000, seed=0):
    if len(ds) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    a = np.asarray(ds, float)
    m = np.sort(rng.choice(a, (n, len(a))).mean(1))
    return float(m[int(.025 * n)]), float(m[int(.975 * n)])


def sign_p(ds, tol=1e-12):
    pos = sum(1 for d in ds if d > tol); neg = sum(1 for d in ds if d < -tol)
    n = pos + neg
    if n == 0:
        return pos, neg, 1.0
    k = min(pos, neg)
    return pos, neg, min(sum(comb(n, i) for i in range(k + 1)) / 2 ** n * 2, 1.0)


def report(label, ds):
    if not ds:
        print(f"  {label:<34}      no cell qualifies")
        return
    lo, hi = boot(ds)
    pos, neg, p = sign_p(ds)
    equiv = max(abs(lo), abs(hi))
    print(f"  {label:<34}n={len(ds):>3}  mean {st.mean(ds):+.3f} steps "
          f"[{lo:+.3f},{hi:+.3f}]  long fewer {neg}, single fewer {pos}, tie "
          f"{len(ds)-pos-neg}  p={p:.3f}  |equiv bound| {equiv:.2f}")


def main():
    C = curves()
    cells = sorted({(m, t, e) for (m, t, l, e) in C
                    if (m, t, "long", e) in C and (m, t, "single", e) in C})
    print(f"{len(cells)} (model, task, eval) cells\n")

    print("=" * 112)
    print("A. THE CLAIM, ON AN ABSOLUTE BEHAVIOUR THRESHOLD (same bar for both arms)")
    print("   delta = rank(k*_long) - rank(k*_single);  negative = long-form needs fewer heads")
    print("=" * 112)
    for B in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
        comp, cens = [], []
        for m, t, e in cells:
            a, b = kstar(C[(m, t, "long", e)], B), kstar(C[(m, t, "single", e)], B)
            if a is not None and b is not None:
                comp.append(a - b)
            cens.append((a if a is not None else CENSORED_RANK) -
                        (b if b is not None else CENSORED_RANK))
        print(f"\n  B = {B:.1f}")
        report("complete-case (both reach B)", comp)
        report("censored-worst (all cells)", cens)

    print("\n" + "=" * 112)
    print("B. ARM-INDEPENDENT RELATIVE THRESHOLD: B = 0.8 x POOLED ceiling")
    print("=" * 112)
    comp, cens = [], []
    for m, t, e in cells:
        L, S = C[(m, t, "long", e)], C[(m, t, "single", e)]
        B = 0.8 * max(max(L.values()), max(S.values()))
        a, b = kstar(L, B), kstar(S, B)
        if a is not None and b is not None:
            comp.append(a - b)
        cens.append((a if a is not None else CENSORED_RANK) -
                    (b if b is not None else CENSORED_RANK))
    report("complete-case", comp)
    report("censored-worst", cens)

    print("\n" + "=" * 112)
    print("C. FOR CONTRAST: THE PUBLISHED VERSION, B_a = 0.8 x EACH ARM'S OWN ceiling")
    print("   (a different bar per arm, so this does not estimate the claim)")
    print("=" * 112)
    own = []
    for m, t, e in cells:
        L, S = C[(m, t, "long", e)], C[(m, t, "single", e)]
        a = kstar(L, 0.8 * max(L.values()))
        b = kstar(S, 0.8 * max(S.values()))
        if a is not None and b is not None:
            own.append(a - b)
    report("min-k to 80% of OWN ceiling", own)

    print("\n" + "=" * 112)
    print("D. THE UNDERLYING LEVELS: is either arm higher at a FIXED budget?")
    print("   If f_L(k) = f_S(k) for every k, no threshold rule can separate them.")
    print("=" * 112)
    print(f"  {'k':>6}{'mean long':>12}{'mean single':>13}{'mean diff':>12}"
          f"{'95% CI':>22}{'long>single':>13}")
    for k in KS:
        d, ls, ss = [], [], []
        for m, t, e in cells:
            L, S = C[(m, t, "long", e)], C[(m, t, "single", e)]
            if k in L and k in S:
                d.append(L[k] - S[k]); ls.append(L[k]); ss.append(S[k])
        if not d:
            continue
        lo, hi = boot(d)
        pos = sum(1 for x in d if x > 0); neg = sum(1 for x in d if x < 0)
        print(f"  {k:>6}{st.mean(ls):>12.3f}{st.mean(ss):>13.3f}{st.mean(d):>+12.3f}"
              f"   [{lo:+.3f},{hi:+.3f}]{f'{pos} vs {neg}':>13}")


if __name__ == "__main__":
    main()
