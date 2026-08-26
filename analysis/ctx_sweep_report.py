#!/usr/bin/env python
"""k-curves and min-k-to-80%-of-ceiling for the --localization_ctx 2x2.

Metric definitions are IMPORTED from random_control_analysis rather than
reimplemented, so the ctx cells are collapsed exactly the way the random-control
arms are: accuracy per topk with the steering-factor axis collapsed by max
(MODEL_SFS), then min-k to 80% of each curve's OWN ceiling.

Long-form eval uses w_rf (judge + fluency + relevance). Single-token eval is
token-matched; CLAUDE.md is explicit that the two are NOT comparable, so they
are reported separately and never pooled.

Usage: python analysis/ctx_sweep_report.py
"""
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from random_control_analysis import (  # noqa: E402
    TOPKS, collapsed_curve, min_k_to_80, load_accuracy, method_dir, sfs_for,
)


def grid_for(model, task, method, reps, mode, rf):
    """Full (steering-factor x topk) accuracy grid, same file naming as
    random_control_analysis.collapsed_curve."""
    d = method_dir(model, task, method, mode)
    sfs = sfs_for(model)
    g = np.full((len(sfs), len(TOPKS)), np.nan)
    for i, sf in enumerate(sfs):
        for j, topk in enumerate(TOPKS):
            g[i, j] = load_accuracy(
                d / f"{sf}_{reps}_steer_topk_{topk}_gen_accuracy_{rf}.json.accuracy.json")
    return g, sfs

MODEL = "Qwen1.5-14B-Chat"
GRID = os.environ.get("CTX_GRID", "fixed")
PREFIX = {"legacy": "atp-", "fixed": "atp-o_proj_in-respfix-"}[GRID]
CTX = [
    ("br-sq", PREFIX + "brsq",           "base+resp / src prompt   (current)"),
    ("br-sr", PREFIX + "srcresp",        "base+resp / src+resp"),
    ("bq-sq", PREFIX + "baseq",          "base prompt / src prompt"),
    ("bq-sr", PREFIX + "baseq-srcresp",  "base prompt / src+resp"),
]
TASKS = ["from_verse-long_to_prose", "from_verse-single_to_prose"]
# (eval mode, metric tag). Long-form is judged; single-token is token-matched.
MODES = [("verse-long", "w_rf"), ("verse-single", "w_rf")]


def main():
    for task in TASKS:
        loc = task.replace("from_", "").replace("_to_prose", "")
        for mode, rf in MODES:
            print(f"\n{'='*78}\n[{GRID}] {loc} localization   |   {mode}_eval   "
                  f"({'judged w_rf' if mode.endswith('long') else 'token-matched'})\n{'='*78}")
            hdr = "ctx      " + "".join(f"{k:>8}" for k in TOPKS) + "   min-k  ceiling"
            print(hdr)
            for name, d, desc in CTX:
                curve, best, found = collapsed_curve(MODEL, task, d, "targeted", mode, rf)
                if found == 0:
                    print(f"{name:8s} " + " (no accuracy files)")
                    continue
                cells = "".join(
                    f"{v:8.3f}" if v == v else f"{'--':>8}" for v in curve)
                mk = min_k_to_80(curve)
                ceil = np.nanmax(curve) if found else np.nan
                print(f"{name:8s}{cells}   {mk if mk==mk else '--':>5}  {ceil:7.3f}"
                      f"   [{found}/{len(TOPKS)*7} cells]  {desc}")

            # max-over-N is the repo convention but inflates the winner: it takes
            # the best of 7 noisy cells (n=50, SE~0.07). Mean-over-N is shown as a
            # robustness check -- a ranking that survives both is worth more than
            # one that only appears under max.
            print("\n  mean over N (robustness check):")
            for name, d, desc in CTX:
                g, sfs = grid_for(MODEL, task, d, "targeted", mode, rf)
                if np.all(np.isnan(g)):
                    continue
                with np.errstate(all="ignore"):
                    mcurve = np.nanmean(g, axis=0)
                cells = "".join(f"{v:8.3f}" if v == v else f"{'--':>8}" for v in mcurve)
                print(f"  {name:8s}{cells}   peak {np.nanmax(mcurve):.3f}")


if __name__ == "__main__":
    main()
