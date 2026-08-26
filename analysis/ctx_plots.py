#!/usr/bin/env python
"""k-curves: long-form vs single-token LOCALIZATION, within each --localization_ctx cell.

One figure per eval mode (they use different metrics and are not comparable).
Shows max-over-N (the repo convention). Mean-over-N remains in
`analysis/ctx_sweep_report.py` as the noise-robust cross-check.
Columns: the four ctx cells.

Palette: dataviz categorical slots 1 (blue) and 2 (orange). Validated all-pairs
in light and dark: CVD dE 24.7/26.8, normal-vision dE 33.6/31.8, contrast >=3:1.

x is ORDINAL: the swept topk values are unevenly spaced (0.01..0.1 then 0.5, 1.0),
so they are drawn evenly with a rule marking the scale break after 0.1.
"""
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from random_control_analysis import TOPKS, collapsed_curve, load_accuracy, method_dir, sfs_for

MODEL = "Qwen1.5-14B-Chat"

# Which results grid to plot. 'legacy' is the committed atp defaults
# (o_proj_out + legacy span); 'fixed' is o_proj_in + respfix, where block u IS
# head u and the metric scores the first response token -- which for -single
# data is the entire answer.
GRID = os.environ.get("CTX_GRID", "fixed")
PREFIX = {"legacy": "atp-", "fixed": "atp-o_proj_in-respfix-"}[GRID]
GRID_LABEL = {"legacy": "o_proj_out + legacy span",
              "fixed": "o_proj_in + respfix"}[GRID]
CTX = [("br-sq", PREFIX + "brsq", "br-sq  (current)\nbase+resp / src prompt"),
       ("br-sr", PREFIX + "srcresp", "br-sr\nbase+resp / src+resp"),
       ("bq-sq", PREFIX + "baseq", "bq-sq\nbase prompt / src prompt"),
       ("bq-sr", PREFIX + "baseq-srcresp", "bq-sr\nbase prompt / src+resp")]
LOCS = [("from_verse-long_to_prose",  "long-form localization",   "#2a78d6"),
        ("from_verse-single_to_prose","single-token localization","#eb6834")]

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e3e2df"


def mean_curve(task, method, mode, rf):
    d = method_dir(MODEL, task, method, mode)
    sfs = sfs_for(MODEL)
    g = np.full((len(sfs), len(TOPKS)), np.nan)
    for i, sf in enumerate(sfs):
        for j, tk in enumerate(TOPKS):
            g[i, j] = load_accuracy(d / f"{sf}_targeted_steer_topk_{tk}_gen_accuracy_{rf}.json.accuracy.json")
    if np.all(np.isnan(g)):
        return None
    with np.errstate(all="ignore"):
        return np.nanmean(g, axis=0)


def figure(mode, rf, metric_label, out):
    x = np.arange(len(TOPKS))
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.3), sharex=True, facecolor=SURFACE)

    rowvals = []
    for cname, cdir, ctitle in CTX:
        for task, llabel, col in LOCS:
            y = collapsed_curve(MODEL, task, cdir, "targeted", mode, rf)[0]
            if y is not None:
                rowvals.append(np.nanmax(y))
    ymax = max(rowvals) * 1.18 if rowvals else 1.0

    for c, (cname, cdir, ctitle) in enumerate(CTX):
        ax = axes[c]
        ax.set_facecolor(SURFACE)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(GRID); ax.spines[sp].set_linewidth(1)
        ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        # scale-break rule: 0.01..0.1 are near-neighbours, 0.5/1.0 are not
        ax.axvline(5.5, color=GRID, linewidth=1, linestyle=(0, (3, 3)), zorder=1)

        for task, llabel, col in LOCS:
            y = collapsed_curve(MODEL, task, cdir, "targeted", mode, rf)[0]
            if y is None:
                continue
            ax.plot(x, y, color=col, linewidth=2, marker="o", markersize=6,
                    markeredgecolor=SURFACE, markeredgewidth=1.5,
                    label=llabel, zorder=3, clip_on=False)
        ax.set_ylim(0, ymax)
        ax.set_xticks(x)
        ax.set_xticklabels([str(t) for t in TOPKS], fontsize=8, color=INK2, rotation=45)
        ax.tick_params(axis="y", labelsize=8, colors=INK2, length=0)
        ax.tick_params(axis="x", length=0)
        ax.set_title(ctitle, fontsize=9.5, color=INK, pad=10, linespacing=1.5)
        if c == 0:
            ax.set_ylabel(f"{metric_label}\n(max over N)", fontsize=9, color=INK)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False,
               fontsize=10, bbox_to_anchor=(0.5, 0.115), labelcolor=INK)
    fig.text(0.5, 0.018, "top-k (head budget) — ordinal spacing; dashed rule marks the scale break after 0.1",
             ha="center", fontsize=8.5, color=INK2)
    fig.suptitle(f"Localization method x patching context — {mode}_eval "
                 f"({metric_label}, {GRID_LABEL})",
                 fontsize=12.5, color=INK, y=0.985)
    fig.tight_layout(rect=[0, 0.19, 1, 0.92])
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print("wrote", out)


if __name__ == "__main__":
    d = Path("results/Qwen1.5-14B-Chat/ctx_report"); d.mkdir(parents=True, exist_ok=True)
    tag = "" if GRID == "legacy" else "_fixed"
    figure("verse-long", "w_rf", "judged w_rf", d / f"ctx_kcurves_long_eval{tag}.png")
    figure("verse-single", "w_rf", "token-matched acc", d / f"ctx_kcurves_single_eval{tag}.png")
