#!/usr/bin/env python
"""
Provenance assay + arm-vs-arm comparison for the persona (extraversion) task.

WHY THIS EXISTS
---------------
CLAUDE.md section 2 defines a mandatory gate. At topk=1.0 every arm selects
*every* attention head, so the atp and random arms apply an identical
intervention and must produce identical accuracy. Any non-zero gap there is not
an effect -- it is a measurement floor, and no arm-vs-arm difference smaller
than that floor is interpretable.

The floor is real for this task because the two sides have different numerical
provenance: umang's atp trees were generated on another machine, while the
random control arms were generated here. Generation is greedy
(eval/generation.py:25-28), so a float-level difference flips a token at a
near-tie and the continuation diverges. Section 2 measured the same effect on
verse at +0.346; persona is far smaller but not zero.

METRIC EQUIVALENCE
------------------
The real arms are scored by umang's pipeline under the 'comb' tag and the
control arms by this repo under 'w_rf'. These are the same quantity:

    comb  = judge >= 5 AND fluency >= 2 AND relevance >= 2   (eval_pipeline_persona.py)
    w_rf  = (judge == 5) AND fluency == 2 AND relevance == 2  (compute_accuracies.py)

Ratings are 1-5 and fluency/relevance are 0-2, so '>= 5' == '== 5' and
'>= 2' == '== 2'. Both also require the same no-prefill judge prompt; the
default '(' prefill shifts ratings down one step and would silently zero w_rf.

TWO EVAL MODES, NEVER MIXED
---------------------------
Free-form eval is judged by the 70B; single-token eval is token matching. Those
numbers are not comparable, so each mode is assayed separately. Single-token
additionally has *shared* provenance -- both arms were generated here -- which is
why it gates clean at 0.000 while free-form does not.

Steering is always matched to the eval mode (CLAUDE.md section 4).

USAGE
    python persona_provenance_assay.py [--eval long|single|both]
"""
from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path

AYUSHI = Path("/home/ubuntu/gcm-interp")
UMANG = Path("/home/ubuntu/gcm-interp-umang")

# The metric tag depends on the EVAL MODE, not on the root: umang's pipeline
# writes 'comb' for free-form and 'w_rf' for single-token. Keying it per root
# instead made every single-token real arm invisible. Same table as figures.py
# MET -- mirror it, do not re-derive it.
MET = {("umang", "long"): "comb", ("umang", "single"): "w_rf",
       ("ayushi", "long"): "w_rf", ("ayushi", "single"): "w_rf"}

# Real arms are searched in priority order. Most persona atp trees are umang's,
# but Falcon's live only in this repo, so ayushi is a real-arm source too.
REAL_ROOTS = [("umang", UMANG / "results_pipeline_with_answers"),
              ("umang", UMANG / "results_pipeline"),
              ("umang", UMANG / "results_with_answers" / "accuracy"),
              ("ayushi", AYUSHI / "judge-evals" / "accuracy")]
CTRL_ROOT = ("ayushi", AYUSHI / "judge-evals" / "accuracy")

MODELS = ["gemma-3-12b-it", "OLMo-2-1124-13B-DPO", "Qwen1.5-14B-Chat",
          "Qwen1.5-32B-Chat", "Falcon3-10B-Instruct"]
LOCS = ["extraversion-long", "extraversion-single"]
SMALL = ["0.01", "0.03", "0.05", "0.07", "0.09", "0.1"]
MODEL_RE = re.compile(r"/([A-Za-z0-9.\-]+(?:Chat|it|DPO|Instruct))/")


def scan(root: Path, metric: str, arms: set[str], eval_mode: str):
    """(model, loc, arm) -> {budget: max accuracy over steering factors}.

    Max over N follows the project convention: every arm is summarised by its
    best steering factor, so no arm wins on sweep width alone.
    """
    out: dict = collections.defaultdict(dict)
    if not root.exists():
        return out
    want_eval = f"extraversion-{eval_mode}_eval"
    want_steer = f"extraversion-{eval_mode}_steer"      # matched steering only
    for p in root.rglob(f"*_gen_accuracy_{metric}.json.accuracy.json"):
        s = str(p)
        if want_eval not in s or want_steer not in s:
            continue
        arm = re.search(r"/(atp|random-s\d+|randomlayer-s\d+)/", s)
        if not arm or arm.group(1) not in arms:
            continue
        mdl = MODEL_RE.search(s)
        loc = re.search(r"from_(extraversion-\w+)_to", s)
        k = re.search(r"topk_([0-9.]+)_gen", s)
        if not (mdl and loc and k):
            continue
        try:
            v = json.load(open(p)).get("q1")
        except Exception:
            continue
        if v is None:
            continue
        d = out[(mdl.group(1), loc.group(1), arm.group(1))]
        d[k.group(1)] = max(d.get(k.group(1), 0.0), float(v))
    return out


def collect(eval_mode: str):
    real: dict = {}
    for repo, root in REAL_ROOTS:                # first root that has a cell wins
        metric = MET[(repo, eval_mode)]
        for key, budgets in scan(root, metric, {"atp"}, eval_mode).items():
            real.setdefault(key, budgets)
    repo, root = CTRL_ROOT
    ctrl = scan(root, MET[(repo, eval_mode)], {"random-s0", "randomlayer-s0"},
                eval_mode)
    return real, ctrl


def report(eval_mode: str) -> None:
    real, ctrl = collect(eval_mode)
    fmt = lambda v: "   -- " if v is None else f"{v:6.3f}"
    mx = lambda d: (max((d.get(k, 0.0) for k in SMALL), default=None)
                    if d else None)

    print(f"\n{'=' * 100}")
    print(f"  persona / {eval_mode}-form eval"
          f"   (real arm = {MET[('umang', eval_mode)]},"
          f" control arm = w_rf)")
    print("=" * 100)
    print(f"  {'model':21} {'loc':7} | {'k=1.0 atp':>9} {'unif':>6} {'depth':>6}"
          f" {'gap':>6} | {'k<=.1 atp':>9} {'unif':>6} {'depth':>6}")

    floor, rows, flagged, saturated = 0.0, 0, [], 0
    for mdl in MODELS:
        for loc in LOCS:
            A = real.get((mdl, loc, "atp"), {})
            U = ctrl.get((mdl, loc, "random-s0"), {})
            D = ctrl.get((mdl, loc, "randomlayer-s0"), {})
            if not U and not D:
                continue
            a1, u1, d1 = A.get("1.0"), U.get("1.0"), D.get("1.0")
            note = ""
            if a1 is None:
                gap = None
                note = "  no real arm"
            else:
                have = [x for x in (a1, u1, d1) if x is not None]
                gap = max(have) - min(have)
                floor = max(floor, gap)
                rows += 1
                # A saturated row agrees trivially, so it is weak evidence that
                # the trees are comparable -- counted, not trusted.
                if all(abs(v - round(v)) < 1e-9 and v in (0.0, 1.0)
                       for v in have):
                    saturated += 1
                    note = "  saturated"
            am, um, dm = mx(A), mx(U), mx(D)
            if am is not None:
                best_ctrl = max([v for v in (um, dm) if v is not None] or [0.0])
                if gap is not None and abs(am - best_ctrl) <= gap:
                    flagged.append(f"{mdl}/{loc.split('-')[1]}")
            print(f"  {mdl:21} {loc.split('-')[1]:7} | {fmt(a1):>9} {fmt(u1)}"
                  f" {fmt(d1)} {'   -- ' if gap is None else f'{gap:6.3f}'}"
                  f" | {fmt(am):>9} {fmt(um)} {fmt(dm)}{note}")

    print(f"\n  comparable rows: {rows}"
          f"   worst gap at k=1.0: {floor:.3f}  <- resolution floor")
    if saturated:
        print(f"  {saturated}/{rows} rows are saturated at 0 or 1, where any two "
              f"arms agree trivially;")
        print("  those rows do not really demonstrate comparable provenance.")
    if floor > 0:
        print(f"\n  Any arm-vs-arm difference below {floor:.3f} is NOT "
              f"interpretable for this task/mode.")
    if flagged:
        print(f"\n  CLAUDE.md 6.3 FLAG -- random within the floor of the real "
              f"localization at k<=0.1:")
        for f in flagged:
            print(f"    {f}")
        print("  Read this as 'not resolvable at this floor', NOT as "
              "'random matches localization'.")


def min_k_report(eval_mode: str) -> None:
    """CLAUDE.md 6.2: min-k to reach 80% of the cell's own ceiling, per arm.

    Peak accuracy cannot separate two arms that both saturate -- persona
    single-token puts atp and depth-matched random at exactly 1.000, which looks
    like agreement but is only a shared ceiling. The budget each arm NEEDS to get
    there is the discriminating quantity, and it is the paper's precision metric.

    The ceiling is COMMON to the cell -- the best accuracy any arm reaches there.
    Scoring each arm against its *own* ceiling looks fairer but silently flatters
    weak arms: a control whose curve is flat and low clears 80% of its own low
    asymptote at the very first budget and scores as maximally precise. Against a
    common ceiling, "precise" means reaching most of what is actually achievable
    in the cell, which is the comparison the paper needs.
    """
    real, ctrl = collect(eval_mode)
    ks = [0.01, 0.03, 0.05, 0.07, 0.09, 0.1, 0.5, 1.0]

    def min_k(d, ceil):
        if not d or ceil <= 0:
            return None
        for k in ks:                      # ascending: first budget clearing 80%
            v = d.get(f"{k:g}")
            if v is not None and v >= 0.8 * ceil:
                return k
        return None                       # never reaches it at any budget

    print(f"\n{'=' * 100}")
    print(f"  min-k to 80% of ceiling -- persona / {eval_mode}-form eval"
          f"   (lower = more precise localization)")
    print("=" * 100)
    print(f"  {'model':21} {'loc':7} {'atp':>7} {'unif':>7} {'depth':>7}"
          f"   verdict vs depth-matched")
    wins = losses = ties = 0
    for mdl in MODELS:
        for loc in LOCS:
            arms = {"atp": real.get((mdl, loc, "atp"), {}),
                    "unif": ctrl.get((mdl, loc, "random-s0"), {}),
                    "depth": ctrl.get((mdl, loc, "randomlayer-s0"), {})}
            if not arms["unif"] and not arms["depth"]:
                continue
            ceil = max((max(d.values()) for d in arms.values() if d),
                       default=0.0)
            A = min_k(arms["atp"], ceil)
            U = min_k(arms["unif"], ceil)
            D = min_k(arms["depth"], ceil)
            f = lambda v: "    -- " if v is None else f"{v:7g}"
            verdict = ""
            if A is not None and D is not None:
                if A < D:
                    verdict, _ = "localization more precise", 0
                    wins += 1
                elif A > D:
                    verdict = "depth-matched more precise"
                    losses += 1
                else:
                    verdict = "tied -- head choice earns nothing"
                    ties += 1
            print(f"  {mdl:21} {loc.split('-')[1]:7} {f(A)} {f(U)} {f(D)}"
                  f"   {verdict}")
    n = wins + losses + ties
    if n:
        print(f"\n  atp vs depth-matched: {wins} finer, {losses} coarser, "
              f"{ties} tied  (n={n})")
        if ties + losses >= wins:
            print("  On THIS task/mode, head-level selection buys no precision "
                  "over matching the")
            print("  layer histogram (CLAUDE.md section 1, arm 2). Scope limits "
                  "before generalising:")
            print("    - one draw seed (s0), so 'within noise' is a point "
                  "estimate, not a spread")
            print("    - the budget grid is coarse, so most 'tied' rows are just "
                  "the same grid point")
            print("    - verse/summarization must be checked separately; run "
                  "random_control_analysis.py")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default="both",
                    choices=["long", "single", "both"])
    a = ap.parse_args()
    for ev in (["long", "single"] if a.eval == "both" else [a.eval]):
        report(ev)
        min_k_report(ev)
    print()


if __name__ == "__main__":
    main()
