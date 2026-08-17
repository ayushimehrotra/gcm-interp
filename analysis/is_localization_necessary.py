#!/usr/bin/env python3
"""Is localization necessary at all? ATP vs the random control arms.

The fewer-heads result compares two localizations against each other. It cannot
say whether *selecting* units matters, because both arms select. That needs an arm
that does not: the repo has two.

  random-s0       units drawn uniformly over all (layer, unit) pairs. The draw
                  ignores --source, so the -long and -single localizations get a
                  bit-identical random set (random_control_mirror_uniform.py).
  randomlayer-s0  units drawn at random but with the per-layer counts matched to
                  the real ATP selection for that arm. Isolates unit choice from
                  layer choice.

Three questions, in the same currency as the fewer-heads analysis:

  1. LEVEL     at a fixed budget k, is ATP more accurate than random?
  2. BUDGET    to reach an absolute behaviour level B, does ATP need a smaller k?
  3. WHERE     how much of any advantage is layer choice vs unit-within-layer?
                 ATP - uniform      = total selection advantage
                 ATP - layermatched = the part from choosing units within layers
                 difference          = the part from choosing layers

Everything is paired on (model, task, localization, eval, steer, N, k) so the only
thing differing between the compared numbers is which units were steered.

CAVEATS, both structural:
  * one draw seed (s0) exists. Every "random" number here is one particular random
    set, not an average over the random-set distribution, so the spread across
    draws is unmeasured. CLAUDE.md flags this too.
  * the random arms cover verse and summarization only, on four models.
"""
import csv
import json
import re
import statistics as st
from collections import defaultdict
from math import comb
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
ROOTS = {"ayushi": Path("/home/ubuntu/gcm-interp/judge-evals/accuracy"),
         "umang": Path("/home/ubuntu/gcm-interp-umang/results_pipeline")}
FN_RE = re.compile(
    r"^(?P<N>\d+)_(?P<reps>random|targeted)_(?P<method>steer|mean)_topk_(?P<topk>[\d.]+)"
    r"_gen_accuracy_(?P<metric>w_rf|wo_rf|comb|flu|rel|judge_3|judge_4|judge_5|mcqa)"
    r"\.json\.accuracy\.json$")
MODE_RE = re.compile(r"^(?P<task>.+)-(?P<mode>long|single)$")
TASKNAME = {"verse": "verse", "paragraph": "summarization", "female": "bias",
            "lying": "factual recall", "extraversion": "persona"}
ARMS = ("atp", "random-s0", "randomlayer-s0")
MET = {("ayushi", "long"): "w_rf", ("ayushi", "single"): "w_rf",
       ("umang", "long"): "comb", ("umang", "single"): "w_rf"}
SHORT = {"Falcon3-10B-Instruct": "Falcon3-10B", "OLMo-2-1124-13B-DPO": "OLMo-2-13B",
         "Qwen1.5-14B-Chat": "Qwen1.5-14B", "Qwen1.5-32B-Chat": "Qwen1.5-32B",
         "gemma-3-12b-it": "Gemma-3-12B"}
KS = [0.01, 0.03, 0.05, 0.07, 0.09, 0.1, 0.5, 1.0]
RANK = {k: i for i, k in enumerate(KS)}
SMALL = KS[:6]


def strip_mode(s):
    m = MODE_RE.match(s)
    return (m.group("task"), m.group("mode")) if m else (s, None)


def harvest():
    out = defaultdict(dict)
    for repo, root in ROOTS.items():
        if not root.exists():
            continue
        for p in root.rglob("*.accuracy.json"):
            parts = p.relative_to(root).parts
            m = FN_RE.match(parts[-1])
            if not m:
                continue
            arm = next((c for c in parts if c in ARMS), None)
            if arm is None:
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
            sb = re.match(r"^from_(?P<src>.+?)_to_(?P<base>.+)$", srcbase)
            task, loc = strip_mode(sb.group("src"))
            task = TASKNAME.get(task)
            _, ev = strip_mode(evald[:-5])
            _, stm = strip_mode(steerd[:-6])
            if task is None or loc is None or ev is None or stm is None:
                continue
            if MET.get((repo, ev)) != m.group("metric"):
                continue
            try:
                v = json.load(open(p)).get("q1")
            except Exception:
                continue
            if v is None:
                continue
            out[(model, task, loc, ev, stm, arm)][(int(m.group("N")),
                                                   float(m.group("topk")))] = float(v)
    return out


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


def line(label, ds, unit=""):
    if not ds:
        print(f"  {label:<40}   no paired points")
        return
    lo, hi = boot(ds)
    pos, neg, p = sign_p(ds)
    print(f"  {label:<40}n={len(ds):>5}  {st.mean(ds):+.4f}{unit} "
          f"[{lo:+.4f},{hi:+.4f}]  ATP better {pos}, random better {neg}, tie "
          f"{len(ds)-pos-neg}  p={p:.4f}")


def main():
    G = harvest()
    base = {k[:5] for k in G if k[5] == "atp"}

    print("=" * 112)
    print("COVERAGE: which cells have a random arm at all")
    print("=" * 112)
    cov = defaultdict(set)
    for k in G:
        if k[5] != "atp":
            cov[(k[0], k[1])].add(k[5])
    print(f"  {'model':<14}{'task':<16}{'arms present':<36}")
    for k in sorted(cov):
        print(f"  {SHORT.get(k[0],k[0]):<14}{k[1]:<16}{', '.join(sorted(cov[k])):<36}")
    print(f"\n  tasks with a random arm: {sorted({k[1] for k in cov})}")
    print(f"  tasks WITHOUT one:       "
          f"{sorted({k[1] for k in G if k[5]=='atp'} - {k[1] for k in cov})}")

    # ------------------------------------------------------------------ 1 ----
    print("\n" + "=" * 112)
    print("1. LEVEL: at a fixed budget k, is ATP more accurate than random?")
    print("   positive = ATP higher.  k<=0.1 only; k=0.5/1.0 steer half or all units.")
    print("=" * 112)
    for other, lbl in (("random-s0", "vs UNIFORM random"),
                       ("randomlayer-s0", "vs LAYER-MATCHED random")):
        print(f"\n  {lbl}")
        print(f"  {'k':>6}   {'n':>5}{'mean ATP - random':>20}{'95% CI':>22}"
              f"{'ATP better':>13}")
        for k in SMALL:
            d = []
            for key in base:
                a, b = G.get(key + ("atp",)), G.get(key + (other,))
                if not (a and b):
                    continue
                d += [a[pt] - b[pt] for pt in a if pt in b and pt[1] == k]
            if not d:
                continue
            lo, hi = boot(d)
            pos = sum(1 for x in d if x > 0); neg = sum(1 for x in d if x < 0)
            print(f"  {k:>6}   {len(d):>5}{st.mean(d):>+20.4f}   [{lo:+.4f},{hi:+.4f}]"
                  f"{f'{pos} vs {neg}':>13}")
        d = []
        for key in base:
            a, b = G.get(key + ("atp",)), G.get(key + (other,))
            if not (a and b):
                continue
            d += [a[pt] - b[pt] for pt in a if pt in b and pt[1] <= 0.1]
        line(f"pooled over k<=0.1  {lbl}", d)

    # ------------------------------------------------------------------ 2 ----
    print("\n" + "=" * 112)
    print("2. BUDGET: to reach an absolute behaviour level B, does ATP need a smaller k?")
    print("   delta = rank(k*_ATP) - rank(k*_random); negative = ATP needs fewer heads")
    print("   complete-case (both reach B); grid-rank scale")
    print("=" * 112)

    def curve(d):
        c = defaultdict(list)
        for (N, k), v in d.items():
            c[k].append(v)
        return {k: max(v) for k, v in c.items()}

    def kstar(f, B):
        for k in KS:
            if f.get(k, 0) >= B:
                return RANK[k]
        return None

    for other, lbl in (("random-s0", "vs UNIFORM random"),
                       ("randomlayer-s0", "vs LAYER-MATCHED random")):
        print(f"\n  {lbl}")
        for B in (0.5, 0.6, 0.7, 0.8):
            d, n_reach_atp, n_reach_rnd, n_cells = [], 0, 0, 0
            for key in base:
                a, b = G.get(key + ("atp",)), G.get(key + (other,))
                if not (a and b):
                    continue
                n_cells += 1
                ka, kb = kstar(curve(a), B), kstar(curve(b), B)
                n_reach_atp += ka is not None
                n_reach_rnd += kb is not None
                if ka is not None and kb is not None:
                    d.append(ka - kb)
            if n_cells:
                lo, hi = boot(d) if d else (float("nan"),) * 2
                pos, neg, p = sign_p(d) if d else (0, 0, 1.0)
                print(f"    B={B:.1f}  cells {n_cells:>2} | reaches B: ATP {n_reach_atp:>2}, "
                      f"random {n_reach_rnd:>2} | complete-case n={len(d):>2} "
                      f"mean {st.mean(d) if d else float('nan'):+.2f} steps "
                      f"[{lo:+.2f},{hi:+.2f}] p={p:.3f}")

    # ------------------------------------------------------------------ 3 ----
    print("\n" + "=" * 112)
    print("3. WHERE THE ADVANTAGE COMES FROM  (pooled over k<=0.1)")
    print("=" * 112)
    tot, unit_part = [], []
    for key in base:
        a = G.get(key + ("atp",))
        u = G.get(key + ("random-s0",))
        lm = G.get(key + ("randomlayer-s0",))
        if a and u:
            tot += [a[pt] - u[pt] for pt in a if pt in u and pt[1] <= 0.1]
        if a and lm:
            unit_part += [a[pt] - lm[pt] for pt in a if pt in lm and pt[1] <= 0.1]
    if tot and unit_part:
        T, U = st.mean(tot), st.mean(unit_part)
        lo_t, hi_t = boot(tot); lo_u, hi_u = boot(unit_part)
        print(f"  total selection advantage   (ATP - uniform)      {T:+.4f} [{lo_t:+.4f},{hi_t:+.4f}]")
        print(f"  from unit choice within layer (ATP - layermatched) {U:+.4f} [{lo_u:+.4f},{hi_u:+.4f}]")
        print(f"  from layer choice             (difference)         {T-U:+.4f}")
        print(f"\n  layer choice is {100*(T-U)/T:.0f}% of the advantage, "
              f"unit-within-layer {100*U/T:.0f}%")

    # ------------------------------------------------------------------ 4 ----
    print("\n" + "=" * 112)
    print("4. DOES ONE LOCALIZATION BEAT RANDOM BY MORE THAN THE OTHER?")
    print("   (if long-form is the better method it should clear random by more)")
    print("=" * 112)
    for loc in ("long", "single"):
        for other, lbl in (("random-s0", "uniform"), ("randomlayer-s0", "layer-matched")):
            d = []
            for key in base:
                if key[2] != loc:
                    continue
                a, b = G.get(key + ("atp",)), G.get(key + (other,))
                if not (a and b):
                    continue
                d += [a[pt] - b[pt] for pt in a if pt in b and pt[1] <= 0.1]
            line(f"{loc}-form ATP vs {lbl} random", d)


if __name__ == "__main__":
    main()
