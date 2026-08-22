#!/usr/bin/env python3
"""Paper figures for the free-form vs single-token localization comparison.

Configuration and data loading live in figdata.py; this file is only the
figures. Run it as:

    python figures.py --eval long   --which levels,bytask,grid
    python figures.py --eval single --which levels

FIGURES
-------
  levels    both arms' mean accuracy against budget, pooled over model-task
            pairs (--controls adds the random arms)
  bytask    the same levels curves, but one panel per task and both evaluation
            modes at once -- the pooled mean is carried by two of five tasks,
            which a single pooled panel cannot show
  grid      per-cell supplement: one panel per (task, model)

DESIGN RULES (from review feedback)
-----------------------------------
1. One figure, one takeaway, stated declaratively in the title.
2. Palatino (TeX Gyre Pagella) so figures match document body text.
3. Minimum ink: no gridlines, no boxes, two spines, direct labels over legends.
4. No layout forcing the reader to cross-reference panels. The grid is the one
   place per-cell detail is the point, so it earns its panel count.

THE GAP FIGURE WAS REMOVED
--------------------------
This file used to lead with the paired difference (free-form minus single-token)
against budget, on the argument that between-cell variance is large and common
to both arms, so differencing within a cell cancels it. The levels figures carry
the comparison now. Note what is lost with it: levels put two overlapping CI
ribbons side by side and leave the reader to eyeball a difference the figure
never states, which is exactly what the gap figure existed to fix. `verdict()`
is kept and still computes the paired difference internally, so levels_figure's
title states who leads where rather than leaving it to the eye.
"""
import argparse
import statistics as st

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

import figdata as D
from figdata import (HERE, ACC_ROOTS, TASKS, KS, MODELS, GRID_MODELS, SHORT,
                     EXCLUDED, UMANG, INK, LF, ST, HAIR, BAND, CTRL,
                     harvest, boot_ci, style, titled, bare, strip_mode)


EVALS = (("long", "Long-form evaluation", "judged accuracy"),
         ("single", "Single-token evaluation", "token/letter-match accuracy"))

MIN_MODELS = 3

NO_ANSWER, WITH_ANSWER = "no answer in context", "answer in context"

NO_ANSWER, WITH_ANSWER = "no answer in context", "answer in context"

TASK_SRC = {"verse": "verse", "summarization": "paragraph", "bias": "female",
            "factual recall": "lying", "persona": "extraversion"}

AYUSHI_TASKS = {"verse", "summarization"}

def condition_of(task, eval_mode):
    """Which umang root actually supplies this task, mirroring harvest's rule.

    Returns None for the tasks that come from the ayushi checkout instead.
    """
    if task in AYUSHI_TASKS:
        return None
    src = TASK_SRC[task]
    # follow the SAME priority order harvest will read, so the tag cannot
    # disagree with the curve it labels
    labels = {"results_pipeline": NO_ANSWER,
              "results_pipeline_with_answers": WITH_ANSWER}
    for root in D.ACC_ROOTS["umang"]:
        label = labels.get(root.name)
        if label is None or not root.exists():
            continue
        # the two roots nest the scored files one level deeper than the ayushi
        # tree (…/<steer>/accuracy/<file>), so search the eval dir rather than
        # pinning the depth
        for evd in root.glob(f"*/from_{src}-*/atp/{src}-{eval_mode}_eval"):
            if next(evd.rglob("*targeted*.accuracy.json"), None) is not None:
                return label
    return None

def curve(acc, models, task, arm):
    """(budgets, mean, lo, hi, n_models) across models at each budget."""
    xs, mu, lo, hi, ns = [], [], [], [], []
    for k in D.KS:
        v = [acc[(m, task, arm)][k] for m in models
             if acc.get((m, task, arm)) and k in acc[(m, task, arm)]]
        if len(v) < MIN_MODELS:
            continue
        l, h = D.boot_ci(v)
        xs.append(k)
        mu.append(st.mean(v))
        lo.append(l)
        hi.append(h)
        ns.append(len(v))
    return xs, mu, lo, hi, (max(ns) if ns else 0)

def verdict(xs, lo, hi):
    """Budgets where each arm's advantage is resolved (CI excludes zero).

    The two evaluation modes give opposite answers, so a hardcoded title would
    mislabel one of them. Every title below is derived from these two lists.
    """
    ahead_long = [xs[z] for z in range(len(xs)) if lo[z] > 0]
    ahead_single = [xs[z] for z in range(len(xs)) if hi[z] < 0]
    return ahead_long, ahead_single

def headline(ahead_long, ahead_single, evalmode):
    where = "free-form" if evalmode == "long" else "single-token"
    if ahead_long and not ahead_single:
        return ("Free-form localization's advantage is confined to "
                "small budgets")
    if ahead_single and not ahead_long:
        return (f"On {where} evaluation, single-token localization is the "
                f"one that leads")
    if ahead_long and ahead_single:
        return "Which localization leads depends on the budget"
    return "The two localizations are indistinguishable at every budget"

CTRL_ARMS = (("random", "uniform random"),
             ("randomlayer_long", "depth-matched (free-form)"),
             ("randomlayer_single", "depth-matched (single-token)"))

def levels_figure(acc, out, evalmode, with_controls=False):
    """Both arms' mean curves. Companion to the gap figure, own takeaway.

    Direct labels on the curves instead of a legend: a legend makes the reader
    look away from the line to decode a colour, which is the same
    cross-referencing cost as a second panel, just smaller.

    with_controls adds the uniform and depth-matched random arms. Each curve is
    averaged over the cells where that arm exists: the localizations over all
    19, the controls over the 12 that have them (verse, summarization, persona
    -- bias and factual recall were never run with controls). The cell counts
    are printed in the subtitle because the curves therefore rest on different
    cell sets, and part of any vertical gap between a localization and a control
    is that difference rather than the arm.
    """
    paired = [(m, t) for m in MODELS for t in TASKS
              if acc.get((m, t, "long")) and acc.get((m, t, "single"))]
    fig, ax = plt.subplots(figsize=(5.4, 3.5))
    bare(ax)
    ax.set_xscale("log")

    curves = {}
    for arm, colour in (("long", LF), ("single", ST)):
        xs, mu, lo, hi = [], [], [], []
        for k in KS:
            v = [acc[(m, t, arm)][k] for m, t in paired if k in acc[(m, t, arm)]]
            if len(v) < 3:
                continue
            xs.append(k)
            mu.append(st.mean(v))
            l, h = boot_ci(v)
            lo.append(l)
            hi.append(h)
        curves[arm] = (xs, mu, lo, hi)
        ax.fill_between(xs, lo, hi, color=colour, alpha=0.13, lw=0, zorder=1)
        ax.plot(xs, mu, color=colour, lw=2.0, zorder=3,
                ls="-" if arm == "long" else (0, (4, 2.2)),
                solid_capstyle="round")
        ax.plot(xs, mu, "o", color=colour, ms=3.6, mew=0, zorder=4)

    n_ctrl = 0
    if with_controls:
        # Controls sit behind the localizations in grey with no CI band: they are
        # a reference level, and three more ribbons would bury the two curves the
        # figure is about.
        for arm, label in CTRL_ARMS:
            cells = [(m, t) for m, t in paired if acc.get((m, t, arm))]
            n_ctrl = max(n_ctrl, len(cells))
            xs, mu = [], []
            for k in KS:
                v = [acc[(m, t, arm)][k] for m, t in cells
                     if k in acc[(m, t, arm)]]
                if len(v) < 3:
                    continue
                xs.append(k)
                mu.append(st.mean(v))
            if not xs:
                continue
            dash = {"random": (0, (1, 1.6)),
                    "randomlayer_long": (0, (5, 1.8)),
                    "randomlayer_single": (0, (2.5, 1.6))}[arm]
            ax.plot(xs, mu, color=CTRL, lw=1.3, ls=dash, zorder=2,
                    solid_capstyle="round")
            curves[arm] = (xs, mu, None, None)
        # label the controls once, on the rightmost point of the topmost one
        top = max((a for a, _ in CTRL_ARMS if a in curves),
                  key=lambda a: max(curves[a][1]), default=None)
        if top is not None:
            xs, mu = curves[top][0], curves[top][1]
            ax.text(xs[-1] * 0.86, max(mu) + 0.02, "random controls",
                    color=CTRL, fontsize=8.8, ha="right", va="bottom")

    xL, muL, _, _ = curves["long"]
    xS, muS, _, _ = curves["single"]
    i = xL.index(0.03) if 0.03 in xL else 1
    ax.text(xL[i] * 0.93, muL[i] + 0.035, "free-form", color=LF, fontsize=9.5,
            ha="center", va="bottom")
    j = xS.index(0.03) if 0.03 in xS else 1
    ax.text(xS[j] * 1.02, muS[j] - 0.035, "single-token", color=ST,
            fontsize=9.5, ha="center", va="top")

    pk = max(range(len(muL)), key=lambda z: muL[z])
    # shade the budgets where free-form actually leads, so the title's claim is
    # something the reader can see rather than something they must take on trust
    # Shade only as far as the last budget actually measured with free-form
    # ahead. The grid jumps 0.1 -> 0.5, so the crossover is unresolved; shading
    # to a midpoint would imply a location the data does not establish.
    # Shade a leading region only when one arm leads over a contiguous run from
    # the smallest budget; on single-token eval the arms trade places and a
    # shaded band would assert a structure the data does not have.
    lead = [z for z in range(len(xL)) if muL[z] > muS[z]]
    contiguous = lead[:1] == [0] and lead == list(range(len(lead)))
    if contiguous and len(lead) < len(xL):
        edge = xL[lead[-1]] * 1.14
        ax.axvspan(0.0085, edge, color=LF, alpha=0.05, lw=0, zorder=0)
        ax.text(edge * 0.94, 0.022,
                "free-form ahead at every\nbudget measured here", fontsize=8.5,
                color=INK, alpha=0.62, ha="right", va="bottom", linespacing=1.35)

    ax.set_xticks([0.01, 0.03, 0.05, 0.1, 0.5, 1.0])
    ax.set_xticklabels(["1%", "3%", "5%", "10%", "50%", "100%"])
    ax.set_xlim(0.0085, 1.25)
    top_y = max(max(curves["long"][3]), max(curves["single"][3]))
    if with_controls:            # a control curve may sit above both CI ribbons
        top_y = max([top_y] + [max(curves[a][1]) for a, _ in CTRL_ARMS
                               if a in curves])
    ax.set_ylim(0, top_y + 0.07)
    ax.set_xlabel("Fraction of attention heads steered")
    ax.set_ylabel("Judged accuracy")
    gx, gmu, glo, ghi = [], [], [], []
    for k in KS:
        d = [acc[(m, t, "long")][k] - acc[(m, t, "single")][k] for m, t in paired
             if k in acc[(m, t, "long")] and k in acc[(m, t, "single")]]
        if len(d) < 3:
            continue
        gx.append(k)
        l, h = boot_ci(d)
        glo.append(l)
        ghi.append(h)
    aheadL, aheadS = verdict(gx, glo, ghi)
    sub = (f"{'free-form' if evalmode == 'long' else 'single-token'} evaluation "
           f"and steering   ·   bands: bootstrap 95% CI of the mean")
    if with_controls:
        # state both counts: the curves are averaged over different cell sets
        sub += (f"\nlocalizations averaged over {len(paired)} model–task pairs, "
                f"controls over the {n_ctrl} that have them")
    titled(ax, headline(aheadL, aheadS, evalmode), sub)
    # "cells" is internal shorthand; readers see the setup, not the tally
    fig.savefig(HERE / f"{out}.pdf")
    fig.savefig(HERE / f"{out}.png", dpi=400)
    plt.close(fig)
    return len(paired), curves

def supplement(acc, out, evalmode):
    """Per-cell grid. Detail is the point here, so the panel count is the design."""
    n_ctrl = set()
    nrow, ncol = len(TASKS), len(GRID_MODELS)
    fig = plt.figure(figsize=(1.6 * ncol + 1.0, 1.42 * nrow + 1.1))
    # left/bottom margins leave room for the shared axis titles; the per-panel
    # ylabels carry task names, so the quantity label goes outside them
    gs = GridSpec(nrow, ncol, figure=fig, top=0.862, bottom=0.10, left=0.155,
                  right=0.985, hspace=0.45, wspace=0.28)
    n = 0
    # rows are tasks, columns are models
    for i, task in enumerate(TASKS):
        # the row's label and y tick labels belong on the leftmost panel that
        # actually has data: set_axis_off() on an empty first column would
        # otherwise take the label and the y axis with it
        first = next((jj for jj, mm in enumerate(GRID_MODELS)
                      if acc.get((mm, task, "long")) and
                      acc.get((mm, task, "single"))), None)
        for j, model in enumerate(GRID_MODELS):
            ax = fig.add_subplot(gs[i, j])
            cL = acc.get((model, task, "long"))
            cS = acc.get((model, task, "single"))
            if not cL or not cS:
                ax.set_axis_off()
                ax.text(0.5, 0.5, "not\nlocalized", ha="center", va="center",
                        fontsize=7, color=HAIR, transform=ax.transAxes)
            else:
                n += 1
                bare(ax)
                ax.set_xscale("log")
                # controls first and faint, so they read as background; each
                # depth-matched control takes its own arm's colour
                for cname, colour in (("random", CTRL),
                                      ("randomlayer_long", LF),
                                      ("randomlayer_single", ST)):
                    cc = acc.get((model, task, cname))
                    if not cc:
                        continue
                    ks = [k for k in KS if k in cc]
                    ax.plot(ks, [cc[k] for k in ks], ls=(0, (1, 1.7)),
                            color=colour, lw=1.0, alpha=0.85, clip_on=False,
                            zorder=1)
                    n_ctrl.add((model, task))
                for curve, colour, ls in ((cL, LF, "-"), (cS, ST, (0, (3, 2)))):
                    ks = [k for k in KS if k in curve]
                    ax.plot(ks, [curve[k] for k in ks], ls=ls, color=colour,
                            lw=1.1, clip_on=False, zorder=3)
                ax.set_xlim(0.0085, 1.25)
                ax.set_ylim(-0.03, 1.03)
                ax.set_xticks([0.01, 0.1, 1.0])
                ax.set_xticklabels(["1%", "10%", "100%"], fontsize=7)
                ax.set_yticks([0, 0.5, 1.0])
                ax.set_yticklabels(["0", ".5", "1"] if j == first else [],
                                   fontsize=7)
                if j == first:
                    ax.set_ylabel(task.capitalize(), fontsize=8.5, labelpad=4)
            if i == 0:
                ax.set_title(SHORT[model], fontsize=9, pad=5)
    where = "free-form" if evalmode == "long" else "single-token"
    fig.text(0.5, 0.972, f"Localization on {where} evaluation, "
             f"per task and model", ha="center", fontsize=10.5)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=LF, lw=1.4, ls="-",
                      label="Free-form localization"),
               Line2D([], [], color=LF, lw=1.0, ls=(0, (1, 1.7)),
                      label="…its depth-matched random"),
               Line2D([], [], color=ST, lw=1.4, ls=(0, (3, 2)),
                      label="Single-token localization"),
               Line2D([], [], color=ST, lw=1.0, ls=(0, (1, 1.7)),
                      label="…its depth-matched random"),
               Line2D([], [], color=CTRL, lw=1.0, ls=(0, (1, 1.7)),
                      label="Uniform random")]
    if not n_ctrl:
        handles = [handles[0], handles[2]]
    # two rows rather than one: five entries on a single line stretched the
    # legend the full figure width and pushed the panels down
    ncol = 3 if len(handles) > 2 else 2
    fig.legend(handles=handles, loc="upper center", ncol=ncol,
               bbox_to_anchor=(0.5, 0.952), fontsize=8.5, handlelength=2.6,
               columnspacing=2.0, labelspacing=0.5, borderpad=0)
    fig.supxlabel("Fraction of attention heads steered", fontsize=10,
                  y=0.028, color=INK)
    fig.supylabel("Judged accuracy", fontsize=10, x=0.026, color=INK)
    fig.savefig(HERE / f"{out}.pdf")
    fig.savefig(HERE / f"{out}.png", dpi=400)
    plt.close(fig)
    return n



# ============================================================================
# bytask -- levels, one panel per task, both evaluation modes
# ============================================================================
# The grid (supplement) shows one levels panel per (task, model); levels_figure
# pools every cell into one panel. This sits between them: the grid's task
# columns, averaged over its model rows. A task whose two arms behave unlike the
# pooled mean stays visible instead of being absorbed into it.
#
# MODEL SET: GRID_MODELS -- all five, Falcon3-10B included. figures.py holds
# Falcon out of the POOLED figures (MODELS) because only 3 of 5 tasks are
# localized for it, so a mean pooled over tasks would rest on a different task
# set for Falcon than for everyone else. That does not apply here: nothing pools
# across tasks, each panel is one task, and each panel's mean is over exactly
# the models that have both arms for it. n is annotated per column because it
# therefore varies: verse / summarization / persona 5, bias 4, factual recall 3.
#
# The two ROWS use different metrics (judged vs token/letter match) and are
# never averaged into one number -- they share a y-range so curve SHAPES can be
# compared, not so a point in one equals a point in the other.


def _bytask_panel(ax, acc, models, task, show_y, show_x):
    bare(ax)
    ax.set_xscale("log")
    n_seen = 0
    for arm, colour, ls in (("long", LF, "-"), ("single", ST, (0, (4, 2.2)))):
        xs, mu, lo, hi, n = curve(acc, models, task, arm)
        if not xs:
            continue
        n_seen = max(n_seen, n)
        ax.fill_between(xs, lo, hi, color=colour, alpha=0.13, lw=0, zorder=1)
        ax.plot(xs, mu, color=colour, lw=2.0, ls=ls, zorder=3,
                solid_capstyle="round")
        ax.plot(xs, mu, "o", color=colour, ms=3.4, mew=0, zorder=4)
    ax.set_xlim(0.0085, 1.25)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xticks([0.01, 0.1, 1.0])
    ax.set_xticklabels(["1%", "10%", "100%"] if show_x else [], fontsize=8)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_yticklabels(["0", ".5", "1"] if show_y else [], fontsize=8)
    return n_seen


def bytask_figure(out="fig_localization_by_task", models=None):
    """Levels per task x evaluation mode, averaged across models."""
    models = models or GRID_MODELS
    if not UMANG.exists():
        print(f"warning: {UMANG} missing -- bias / factual recall / persona "
              f"will be empty. Clone the umang checkout there.")
    nrow, ncol = len(EVALS), len(TASKS)
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.05 * ncol + 0.9,
                                                  2.15 * nrow + 0.95))
    counts, peaks = {}, {}
    sparse = [k for k in KS if k <= 0.1]
    for r, (mode, row_title, row_metric) in enumerate(EVALS):
        acc = harvest(mode)
        for c, task in enumerate(TASKS):
            ax = axes[r][c]
            counts[(mode, task)] = _bytask_panel(
                ax, acc, models, task, show_y=(c == 0), show_x=(r == nrow - 1))
            pk = {}
            for arm in ("long", "single"):
                vals = [max(acc[(m, task, arm)][k] for k in sparse
                            if k in acc[(m, task, arm)])
                        for m in models if acc.get((m, task, arm))]
                pk[arm] = st.mean(vals) if vals else float("nan")
            peaks[(mode, task)] = pk
            if c == 0:
                ax.text(-0.42, 0.5, f"{row_title}\n{row_metric}",
                        transform=ax.transAxes, ha="center", va="center",
                        rotation=90, fontsize=9, color=INK, linespacing=1.5)
    # n and the source condition vary by TASK, not by row, so they belong in the
    # column header -- in-panel they collided with curves that reach 1.0
    for c, task in enumerate(TASKS):
        ns = {counts[(mode, task)] for mode, _, _ in EVALS}
        n_txt = str(ns.pop()) if len(ns) == 1 else "/".join(
            str(counts[(mode, task)]) for mode, _, _ in EVALS)
        conds = {condition_of(task, mode) for mode, _, _ in EVALS}
        cond = conds.pop() if len(conds) == 1 else None
        ax = axes[0][c]
        ax.set_title(task.capitalize(), fontsize=9.5, pad=17)
        ax.text(0.5, 1.035, f"$n$={n_txt}" + (f", {cond}" if cond else ""),
                transform=ax.transAxes, ha="center", va="bottom",
                fontsize=7.4, color=INK, alpha=0.62)

    fig.text(0.5, 0.045, "steering budget (fraction of blocks steered)",
             ha="center", fontsize=9.5, color=INK)
    handles = [Line2D([], [], color=LF, lw=2.0, ls="-",
                      label="Free-form localization"),
               Line2D([], [], color=ST, lw=2.0, ls=(0, (4, 2.2)),
                      label="Single-token localization")]
    fig.legend(handles, [h.get_label() for h in handles], loc="lower center",
               ncol=2, fontsize=9, bbox_to_anchor=(0.5, -0.005),
               handlelength=2.6, columnspacing=2.4)
    fig.suptitle("Free-form localization's advantage is task- and "
                 "evaluation-dependent", fontsize=12, y=1.012)
    fig.text(0.5, 0.952,
             "mean across models at each budget; band is a 10,000-sample "
             "bootstrap 95% CI. Rows use different metrics and are not "
             "comparable point-for-point.",
             ha="center", va="bottom", fontsize=8.2, color=INK, alpha=0.62)
    fig.subplots_adjust(left=0.105, right=0.99, top=0.845, bottom=0.135,
                        hspace=0.22, wspace=0.16)
    for ext in ("png", "pdf"):
        fig.savefig(HERE / f"{out}.{ext}", dpi=200)
    plt.close(fig)
    return counts, peaks




def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--eval", default="long", choices=["long", "single"],
                    help="evaluation mode for levels/grid (bytask does both)")
    ap.add_argument("--which", default="levels,bytask",
                    help="comma list: levels,bytask,grid")
    ap.add_argument("--out", default=None, help="basename for levels/grid")
    ap.add_argument("--controls", action="store_true",
                    help="also write the levels figure with the random arms")
    ap.add_argument("--models", default=None,
                    help="comma-separated model dirs for bytask "
                         "(default: all five)")
    a = ap.parse_args()
    want = [w.strip() for w in a.which.split(",")]
    out = a.out or f"fig_{a.eval}eval_levels"
    fam = style()
    print(f"font: {fam}")

    if "levels" in want:
        acc = harvest(a.eval)
        nl, _ = levels_figure(acc, out, a.eval)
        print(f"wrote {out}.pdf / .png   ({nl} cells averaged)")
        if a.controls:
            acc_ctrl = harvest(a.eval, include_random=True)
            nc, _ = levels_figure(acc_ctrl, out + "_controls", a.eval,
                                  with_controls=True)
            print(f"wrote {out}_controls.pdf / .png   ({nc} cells averaged)")

    if "bytask" in want:
        models = a.models.split(",") if a.models else GRID_MODELS
        counts, peaks = bytask_figure(models=models)
        print("wrote fig_localization_by_task.pdf / .png")
        print(f"\n{'eval':8s}{'task':16s}{'n':>2s}  "
              f"{'free-form':>10s}{'single-tok':>12s}{'diff':>9s}   (peak k<=0.1)")
        for mode, _, _ in EVALS:
            for task in TASKS:
                pk = peaks[(mode, task)]
                print(f"{mode:8s}{task:16s}{counts[(mode, task)]:2d}  "
                      f"{pk['long']:10.3f}{pk['single']:12.3f}"
                      f"{pk['long'] - pk['single']:+9.3f}")

    if "grid" in want:
        acc_ctrl = harvest(a.eval, include_random=True)
        ns = supplement(acc_ctrl, f"fig_{a.eval}eval_grid", a.eval)
        print(f"wrote fig_{a.eval}eval_grid.pdf / .png   "
              f"({ns} localized cells of 25)")


if __name__ == "__main__":
    main()

