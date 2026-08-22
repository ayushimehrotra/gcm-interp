#!/usr/bin/env python3
"""Analysis figure: where in the network each localization selects.

One figure, one takeaway (FINDINGS.md 2): long-form localization selects EARLIER
and TIGHTER than single-token. Config, palette and data loading come from
figdata; this file is only the figure.

    python figures_analysis.py [--k 0.1]

WHAT WAS REMOVED, AND WHY IT COULD NOT BE KEPT
----------------------------------------------
This module used to build five figures (depth, shared, dose, contain, forest).
The other four are gone because the analyses behind them are gone: shared read
activation_geometry.csv, dose read position_count*.csv, contain read
containment_asymmetry.csv, and forest pooled those plus layerwise_mechanics.csv,
necessity_long.csv and probe_units.csv. Those CSVs are still on disk, so the
figures would still have *rendered* -- but with no producing script left they
could never be refreshed, and FINDINGS.md 7 records exactly what that failure
mode costs: the dose panel kept showing pre-pull numbers however often the
analysis was rerun, because its consumer read a stale filename. A figure whose
producer no longer exists is worse than no figure, so they were removed rather
than left to rot.

fig_depth survives because it recomputes from the raw attribution fields via
figdata.load_fields -- it depends on data, not on another script's CSV.
"""
import argparse
import statistics as st

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import figdata as D
from figdata import (HERE, INK, LF, ST, HAIR, SHORT, MODELS, TASKS,
                     style, titled, load_fields, top_units)


def bare(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(False)
    ax.set_axisbelow(True)

def save(fig, name):
    fig.savefig(HERE / f"{name}.pdf")
    fig.savefig(HERE / f"{name}.png", dpi=400)
    plt.close(fig)
    print(f"wrote {name}.pdf / .png")

def fig_depth(out="fig_depth_profile", k=0.1):
    """Back-to-back histograms of selected-block depth.

    Drawn with stairs(), not fill_between() on bin centres: interpolating
    between centres turns a histogram into a jagged polygon and invents slopes
    the data does not contain. Budget k=0.1 matches the layer statistics
    reported in FINDINGS.md 2, so figure and text agree.
    """
    bins = np.linspace(0, 1, 21)
    accL, accS, means = [], [], {"long": [], "single": []}
    for model in MODELS:
        fields = load_fields(model)
        for task in TASKS:
            if (task, "long") not in fields or (task, "single") not in fields:
                continue
            for arm, bucket in (("long", accL), ("single", accS)):
                A = fields[(task, arm)]
                sel = top_units(A, k)
                nl = A.shape[0]
                rel = np.array([l / (nl - 1) for l, _ in sel])
                h, _ = np.histogram(rel, bins=bins)
                bucket.append(h / h.sum())
                means[arm].append(rel.mean())
    # each pair normalised to sum 1 BEFORE averaging, so models with deeper
    # stacks (Qwen-32B has 64 layers, OLMo and Falcon 40) do not dominate
    L = np.array(accL).mean(0)
    S = np.array(accS).mean(0)
    npairs = len(accL)

    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    bare(ax)
    ax.stairs(L, bins, fill=True, color=LF, alpha=0.8, lw=0)
    ax.stairs(-S, bins, fill=True, color=ST, alpha=0.8, lw=0)
    ax.axhline(0, color="white", lw=1.0, zorder=4)
    mL, mS = st.mean(means["long"]), st.mean(means["single"])
    top, bot = L.max(), S.max()
    ax.plot([mL, mL], [0, top * 1.10], color=INK, lw=0.9, ls=(0, (2, 2)),
            zorder=5)
    ax.plot([mS, mS], [0, -bot * 1.10], color=INK, lw=0.9, ls=(0, (2, 2)),
            zorder=5)
    ax.annotate(f"mean depth {mL:.2f}", xy=(mL, top * 1.10),
                xytext=(mL - 0.03, top * 1.16), ha="right", fontsize=8.5,
                color=INK, alpha=0.8)
    ax.annotate(f"mean depth {mS:.2f}", xy=(mS, -bot * 1.10),
                xytext=(mS + 0.03, -bot * 1.16), ha="left", va="top",
                fontsize=8.5, color=INK, alpha=0.8)
    ax.set_ylim(-bot * 1.42, top * 1.42)
    ax.text(0.985, 0.93, "long-form", transform=ax.transAxes, ha="right",
            color=LF, fontsize=10)
    ax.text(0.985, 0.07, "single-token", transform=ax.transAxes, ha="right",
            color=ST, fontsize=10)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["input", "¼", "half-way", "¾", "output"])
    ax.set_xlabel("Depth through the network")
    titled(ax, "Long-form localization selects earlier in the network",
           f"bar height: share of selected heads at a top-{k:.0%} budget   ·   "
           f"{npairs} model–task pairs, each weighted equally")
    save(fig, out)
    print(f'   depth profile averaged {npairs} model-task pairs')
    return mL, mS

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--k", type=float, default=0.1,
                    help="budget; 0.1 matches the layer statistics in "
                         "FINDINGS.md 2 so figure and text agree")
    a = ap.parse_args()
    style()
    mL, mS = fig_depth(k=a.k)
    print(f"wrote fig_depth_profile.pdf / .png")
    print(f"   mean depth  long-form {mL:.3f}   single-token {mS:.3f}"
          f"   (long-form earlier by {mS - mL:+.3f})")


if __name__ == "__main__":
    main()
