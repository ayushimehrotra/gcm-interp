#!/usr/bin/env python
"""
Does the localization-vs-random comparison depend on the cell having any range?

THE QUESTION
------------
Two readings of the control-arm data are on the table:

  (A) Head-level selection earns nothing beyond the layer histogram. The
      depth-matched random arm ties the real localization almost everywhere, so
      what localization really buys is a depth profile.
  (B) The ties are an artefact of dead cells. Where accuracy is pinned at 0 or
      1 across the whole budget sweep, every arm scores the same by
      construction, and a tie there is not evidence of anything.

These make opposite predictions once cells are split by dynamic range. Under (A)
the tie rate should hold up among live cells; under (B) it should collapse.

DYNAMIC RANGE
-------------
A cell is LIVE for an arm if that arm's accuracy actually moves across the
budget sweep:

    range = max_k acc(k) - min_k acc(k)  >=  MIN_RANGE     (default 0.15)

The gate is applied to the REAL arm only, and deliberately so: gating on the
control would select cells where the control happens to work and bias the
comparison toward (B). Whether the real arm has range is a property of the cell
-- is this task steerable in this model at all -- not of the contrast.

WHAT IS COMPARED
----------------
min-k to 80% of the cell's COMMON ceiling (the best any arm reaches there).
Scoring each arm against its own ceiling flatters a flat, low control: it clears
80% of its own low asymptote at the first budget and scores as maximally
precise. See persona_provenance_assay.py, which uses the same definition.

Reported as a two-sided sign test on discordant pairs, with ties shown but not
counted -- on this coarse budget grid a "tie" usually means the same grid point,
which is a resolution limit rather than an equality.

SCOPE
-----
Seed 0 only. CLAUDE.md section 4 wants an across-seed spread before calling any
arm "within noise"; one draw gives a point estimate, so this gates a reading
rather than establishing one.

USAGE
    python dynamic_range_gate.py [--min_range 0.15]
"""
from __future__ import annotations

import argparse
import glob
import json
import re
from collections import defaultdict
from math import comb

AY = "/home/ubuntu/gcm-interp/judge-evals/accuracy"
UM = "/home/ubuntu/gcm-interp-umang"
KS = [0.01, 0.03, 0.05, 0.07, 0.09, 0.1, 0.5, 1.0]
MODEL_RE = re.compile(r"/([A-Za-z0-9.\-]+(?:Chat|it|DPO|Instruct))/")
TASKNAME = {"verse": "verse", "paragraph": "summarization", "female": "bias",
            "lying": "factual recall", "extraversion": "persona"}


def sign_test(w: int, l: int) -> float:
    n = w + l
    if n == 0:
        return 1.0
    tail = sum(comb(n, i) for i in range(min(w, l) + 1))
    return min(1.0, 2 * tail / 2 ** n)


def harvest():
    """(model, task, loc, eval, arm) -> {k: best accuracy over steering factors}."""
    # bucket[key][k][(priority, root)] -> best value in that root
    # Roots are PRIORITY-ORDERED, not merged. Taking a max across roots would
    # inflate whichever arm appears in more of them -- atp lives in several,
    # while the control arms exist only in ayushi's tree, so merging hands the
    # real arm a systematic advantage that has nothing to do with localization.
    bucket = defaultdict(lambda: defaultdict(dict))
    roots = [(AY, "ayushi"), (f"{UM}/results_pipeline_with_answers", "umang"),
             (f"{UM}/results_pipeline", "umang"),
             (f"{UM}/results_with_answers", "umang")]
    for prio, (root, repo) in enumerate(roots):
        for p in glob.glob(f"{root}/**/*accuracy.json", recursive=True):
            m = re.search(r"topk_([0-9.]+)_gen_accuracy_(\w+)\.json", p)
            sb = re.search(r"from_([\w-]+?)_to_", p)
            ev = re.search(r"/([\w-]+?)_eval/", p)
            st = re.search(r"/([\w-]+?)_steer/", p)
            arm = re.search(r"/(atp|random-s0|randomlayer-s0)/", p)
            mdl = MODEL_RE.search(p)
            if not (m and sb and ev and st and arm and mdl):
                continue
            k, metric = m.group(1), m.group(2)
            evmode = "long" if ev.group(1).endswith("-long") else "single"
            # metric depends on repo and eval mode, exactly as figures.py MET
            want = "comb" if (repo == "umang" and evmode == "long") else "w_rf"
            if metric != want:
                continue
            if ev.group(1) != st.group(1):        # matched steering only
                continue
            src = sb.group(1)
            base_task = re.sub(r"-(long|single)$", "", src)
            task = TASKNAME.get(base_task)
            loc = "long" if src.endswith("-long") else "single"
            if task is None or not src.endswith(("-long", "-single")):
                continue
            try:
                v = json.load(open(p)).get("q1")
            except Exception:
                continue
            if v is None:
                continue
            key = (mdl.group(1), task, loc, evmode, arm.group(1))
            slot = bucket[key][float(k)]
            slot[(prio, root)] = max(slot.get((prio, root), 0.0), float(v))
    # highest-priority root that has the cell wins; max over steering factors
    # happens within that root only
    g = defaultdict(dict)
    for key, budgets in bucket.items():
        for k, srcs in budgets.items():
            top = min(p for p, _ in srcs)
            g[key][k] = max(v for (p, _), v in srcs.items() if p == top)
    return g


def min_k(curve, ceiling):
    if not curve or ceiling <= 0:
        return None
    for k in KS:
        if k in curve and curve[k] >= 0.8 * ceiling:
            return k
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min_range", type=float, default=0.15)
    a = ap.parse_args()
    g = harvest()

    cells = sorted({k[:4] for k in g})
    buckets = {True: defaultdict(list), False: defaultdict(list)}
    detail = []
    for (mdl, task, loc, ev) in cells:
        arms = {n: g.get((mdl, task, loc, ev, n), {})
                for n in ("atp", "random-s0", "randomlayer-s0")}
        if not arms["atp"] or not (arms["random-s0"] or arms["randomlayer-s0"]):
            continue
        vals = arms["atp"].values()
        rng = max(vals) - min(vals)
        live = rng >= a.min_range
        ceiling = max(max(c.values()) for c in arms.values() if c)
        mk = {n: min_k(c, ceiling) for n, c in arms.items()}
        for label, ctrl in (("depth", "randomlayer-s0"), ("unif", "random-s0")):
            A, C = mk["atp"], mk[ctrl]
            if A is None and C is None:
                continue
            A_, C_ = (99 if A is None else A), (99 if C is None else C)
            buckets[live][label].append("tie" if A_ == C_ else
                                        ("real" if A_ < C_ else "ctrl"))
        detail.append((mdl, task, loc, ev, rng, live, mk))

    print(f"Dynamic-range gate: a cell is LIVE if the real arm's accuracy moves "
          f">= {a.min_range:.2f} across the sweep\n")
    print(f"  {'model':21} {'task':14} {'loc':7} {'ev':7} {'range':>6} {'live':>5}"
          f" {'atp':>5} {'unif':>5} {'depth':>5}")
    for mdl, task, loc, ev, rng, live, mk in sorted(detail,
                                                    key=lambda r: (-r[4],)):
        f = lambda v: "  n/r" if v is None else f"{v:5g}"
        print(f"  {mdl:21} {task:14} {loc:7} {ev:7} {rng:6.2f} {str(live):>5}"
              f" {f(mk['atp'])} {f(mk['random-s0'])} {f(mk['randomlayer-s0'])}")

    print(f"\n{'=' * 92}")
    print("  localization vs each control, split by whether the cell has any "
          "dynamic range")
    print("=" * 92)
    for live in (True, False):
        tag = "LIVE cells" if live else "DEAD cells (pinned at 0 or 1)"
        print(f"\n  {tag}")
        for label, name in (("depth", "depth-matched random"),
                            ("unif", "uniform random")):
            o = buckets[live][label]
            if not o:
                print(f"    vs {name:22} -- no cells")
                continue
            w, l, t = o.count("real"), o.count("ctrl"), o.count("tie")
            print(f"    vs {name:22} localization finer {w:2}, control finer "
                  f"{l:2}, tied {t:2}  (n={len(o)})  p={sign_test(w, l):.4f}")

    print(f"\n{'=' * 92}")
    print("  READING")
    print("=" * 92)
    live_d = buckets[True]["depth"]
    dead_d = buckets[False]["depth"]
    if live_d:
        lw, ll, lt = (live_d.count("real"), live_d.count("ctrl"),
                      live_d.count("tie"))
        print(f"    Among live cells the depth-matched tie rate is "
              f"{lt}/{len(live_d)}"
              f" ({lt / len(live_d):.0%}).")
    if dead_d:
        dt = dead_d.count("tie")
        print(f"    Among dead cells it is {dt}/{len(dead_d)} "
              f"({dt / len(dead_d):.0%}).")
    print("    If the tie rate is high only among dead cells, 'head choice "
          "earns nothing' is")
    print("    an artefact of cells with no range, not a finding about "
          "localization.")


if __name__ == "__main__":
    main()
