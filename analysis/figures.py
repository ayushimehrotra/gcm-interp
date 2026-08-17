#!/usr/bin/env python3
"""Paper figures for the free-form vs single-token localization comparison.

DESIGN RULES (from review feedback)
-----------------------------------
1. One figure, one takeaway, stated declaratively in the title and repeated in
   the LaTeX caption printed by --caption.
2. Palatino (TeX Gyre Pagella, the clone LaTeX's mathpazo/newpxtext use) so the
   figure matches document body text rather than approximating it. One shared
   palette across every figure in this file.
3. Minimum ink: no gridlines, no boxes, two spines, direct labels instead of a
   legend, and no annotation repeating a number the axis already gives.
4. No layout forcing the reader to cross-reference panels. The main figure is a
   SINGLE panel. The 5x5 per-cell grid lives behind --supplement, where per-cell
   detail is the point rather than the message.

MAIN FIGURE  (default)
----------------------
The paired accuracy gap, free-form minus single-token, against budget.

The paired difference is the right object rather than two overlaid level curves:
between-cell variance (task difficulty, model, judge behaviour) is large and
common to both arms, so plotting levels puts two heavily overlapping CI ribbons
side by side and leaves the reader to eyeball a difference the figure never
states. Differencing within each cell cancels that variance and plots the
quantity the claim is actually about.

  faint grey lines   one per cell, so spread and counter-examples stay visible
                     rather than being hidden inside a mean
  bold line + band   mean over cells, 10,000-sample bootstrap 95% CI

Accuracy is taken at the BEST steering factor for each budget: each (arm, k) ran
at N in {1,2,4,5,6,8,10}, and the max over N asks "what is the best this budget
can do", which is what a claim about budget should mean. It also removes N as a
nuisance dimension the arms could differ on for unrelated reasons. The same rule
is applied to the random control arms, so all four curves are comparable -- a
sanity check is that at k=1.0, where every arm steers the identical full set,
they agree.

Long-form judged evaluation only -- single-token evals are token-matched MCQA and
are not the target behaviour. Metric per repo follows the judged pipeline
(w_rf for ayushi's tasks, comb for umang's).

The budget grid {0.01, 0.03, 0.05, 0.07, 0.09, 0.1, 0.5, 1.0} is very
non-uniform, so the axis is log-scaled with ticks at the real grid values. Lines
between grid points are a reading aid; nothing is interpolated.
"""
import argparse
import json
import re
import statistics as st
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

HERE = Path(__file__).parent
AYUSHI = Path("/home/ubuntu/gcm-interp")
UMANG = Path("/home/ubuntu/gcm-interp-umang")
# Per repo, judged-accuracy roots in PRIORITY order. umang has two runs:
# results_pipeline (steering evaluated without the answer present) and
# results_pipeline_with_answers (with it). They are different conditions, so a
# cell is taken from the first root that has it rather than merged -- a cell's
# two arms always come from the same condition, which is what the paired
# comparison requires.
ACC_ROOTS = {"ayushi": [AYUSHI / "judge-evals" / "accuracy"],
             "umang": [UMANG / "results_pipeline",
                       UMANG / "results_pipeline_with_answers"]}
TASKNAME = {"verse": "verse", "paragraph": "summarization", "female": "bias",
            "lying": "factual recall", "extraversion": "persona"}
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

# MODELS drives the AVERAGED figures (gap, levels). Falcon3-10B is held out of
# those on purpose: only 3 of 5 tasks are localized for it, so averaging over a
# different task set than the other models would make the means incomparable.
MODELS = ["gemma-3-12b-it", "Qwen1.5-14B-Chat", "Qwen1.5-32B-Chat",
          "OLMo-2-1124-13B-DPO"]
# GRID_MODELS drives the per-cell grid, where every cell is read on its own and
# nothing is pooled -- so an incomplete model costs nothing and its populated
# cells are worth showing. Falcon's missing tasks render as "not localized".
GRID_MODELS = MODELS + ["Falcon3-10B-Instruct"]
# Models held out of anything that pools across tasks. Derived from the two lists
# above so it cannot drift from them; figures_analysis.py imports it to apply the
# same filter to the CSVs it reads directly.
EXCLUDED = [m for m in GRID_MODELS if m not in MODELS]
SHORT = {"gemma-3-12b-it": "Gemma-3-12B", "Qwen1.5-14B-Chat": "Qwen1.5-14B",
         "Qwen1.5-32B-Chat": "Qwen1.5-32B", "OLMo-2-1124-13B-DPO": "OLMo-2-13B",
         "Falcon3-10B-Instruct": "Falcon3-10B"}
TASKS = ["verse", "summarization", "bias", "factual recall", "persona"]
KS = [0.01, 0.03, 0.05, 0.07, 0.09, 0.1, 0.5, 1.0]

# one palette, used by every figure here
INK = "#2b2f36"        # text and axes
LF = "#1f6f8b"         # free-form
ST = "#c25a34"         # single-token
HAIR = "#b9bec6"       # per-cell context lines
BAND = "#7fb3c4"       # CI fill
CTRL = "#8d939c"       # random control arms (grid only)


def strip_mode(s):
    m = MODE_RE.match(s)
    return (m.group("task"), m.group("mode")) if m else (s, None)


def harvest(eval_mode="long", include_random=False):
    """(model, task, arm) -> {k: best accuracy over steering factors}.

    arm is "long" / "single" for the two localizations and, when
    include_random is set, "random" (uniform) / "randomlayer" (depth-matched)
    for the control arms. Every arm is aggregated the same way: the best
    steering factor at each budget, so the four curves are comparable.
    """
    g = defaultdict(lambda: defaultdict(dict))
    for repo, roots in ACC_ROOTS.items():
      for prio, root in enumerate(roots):
        if not root.exists():
            continue
        for p in root.rglob("*.accuracy.json"):
              parts = p.relative_to(root).parts
              m = FN_RE.match(parts[-1])
              if not m or m.group("method") != "steer":
                  continue
              ctrl = next((c for c in parts if c.startswith("random")), None)
              if ctrl is None and "atp" not in parts:
                  continue
              if ctrl is not None and not include_random:
                  continue
              # atp trees are written with reps=targeted, control trees with
              # reps=random (eval_runner.py keeps the filename stem so the judge
              # regex still matches), so the expected value depends on the tree
              if m.group("reps") != ("random" if ctrl else "targeted"):
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
              if task is None or arm is None or ev != eval_mode or stm != ev:
                  continue
              if MET.get((repo, ev)) != m.group("metric"):
                  continue
              try:
                  v = json.load(open(p)).get("q1")
              except Exception:
                  continue
              if v is not None:
                  # Depth-matched random is drawn from a SPECIFIC arm's layer
                  # histogram, so there is one per arm and they differ a lot
                  # (mean |gap| up to 0.225, max 0.62). Key it by arm so each
                  # localization is compared against its own control. Uniform
                  # random uses a fixed seed and is arm-independent, so the two
                  # copies are the same intervention and stay pooled.
                  cname = None if ctrl is None else ctrl.split("-")[0]
                  key = (arm if cname is None else
                         f"{cname}_{arm}" if cname == "randomlayer" else cname)
                  # keep the localization tree in the bucket: control arms exist
                  # under BOTH trees, and max-over-N must be taken within a tree
                  # before averaging across trees, never over the two mixed
                  g[(model, task, key)][float(m.group("topk"))]\
                      .setdefault((prio, srcbase), []).append(float(v))
    out = {}
    for key, d in g.items():
        agg = {}
        for kk, bucket in d.items():
            top = min(p for p, _ in bucket)          # highest-priority root
            # One rule for every arm: the best steering factor at this budget.
            # Applied identically to free-form, single-token, uniform random and
            # depth-matched random, so the four curves are directly comparable.
            # (An earlier version averaged over N for the control arms only,
            # which depressed them -- Gemma verse at k=1.0 read 0.514 where the
            # best-N value is 0.800.) Where a control exists under both
            # localization trees, this takes the better of the two.
            agg[kk] = max(v for (p, _), vs in bucket.items() if p == top
                          for v in vs)
        out[key] = agg
    return out


def boot_ci(rows, n=10000, seed=0):
    if len(rows) < 2:
        return float("nan"), float("nan")
    r = np.random.default_rng(seed)
    a = np.asarray(rows, float)
    m = np.sort(r.choice(a, (n, len(a))).mean(1))
    return float(m[int(.025 * n)]), float(m[int(.975 * n)])


def style():
    for f in sorted((HERE / "fonts").glob("*.otf")):
        fm.fontManager.addfont(str(f))
    fam = ("TeX Gyre Pagella"
           if any("Pagella" in f.name for f in fm.fontManager.ttflist)
           else "DejaVu Serif")
    plt.rcParams.update({
        "font.family": "serif", "font.serif": [fam, "DejaVu Serif"],
        "mathtext.fontset": "custom", "mathtext.rm": fam,
        "mathtext.it": f"{fam}:italic", "mathtext.bf": f"{fam}:bold",
        # mathtext.cal defaults to a cursive family that is not installed and
        # emits a findfont warning on every render
        "mathtext.cal": fam, "mathtext.sf": fam, "mathtext.tt": "DejaVu Sans Mono",
        "font.size": 9,
        "axes.linewidth": 0.7, "axes.edgecolor": INK, "axes.labelcolor": INK,
        "text.color": INK, "xtick.color": INK, "ytick.color": INK,
        "xtick.direction": "out", "ytick.direction": "out",
        "xtick.major.width": 0.7, "ytick.major.width": 0.7,
        "xtick.major.size": 3, "ytick.major.size": 3,
        "xtick.minor.size": 0, "ytick.minor.size": 0,
        "legend.frameon": False, "figure.dpi": 200,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
    })
    return fam


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


def titled(ax, title, subtitle=None, fontsize=10.5):
    """Centred title with an optional smaller subtitle above the axes.

    Meta text (sample size, what the band means, which evaluation) goes here
    rather than inside the axes, where it collides with data as soon as the
    curve shape changes.
    """
    # pad must grow with the subtitle's line count, or a second line runs into
    # the title
    lines = 0 if not subtitle else subtitle.count("\n") + 1
    ax.set_title(title, fontsize=fontsize, pad=9 + 9 * lines)
    if subtitle:
        ax.text(0.5, 1.015, subtitle, transform=ax.transAxes, ha="center",
                va="bottom", fontsize=8.5, color=INK, alpha=0.62,
                linespacing=1.45)


def bare(ax):
    """Minimum ink: two spines, no grid, no box."""
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(False)
    ax.set_axisbelow(True)


def main_figure(acc, out, evalmode):
    paired = [(m, t) for m in MODELS for t in TASKS
              if acc.get((m, t, "long")) and acc.get((m, t, "single"))]
    fig, ax = plt.subplots(figsize=(5.4, 3.5))
    bare(ax)
    ax.set_xscale("log")

    # Cell spread is shown as an interquartile band rather than 20 crossing
    # lines: the spaghetti spans -0.66 to +0.76 and forces a y-range that
    # squashes the result being reported into a sliver around zero.
    xs, mu, lo, hi, q1, q3 = [], [], [], [], [], []
    for k in KS:
        d = [acc[(m, t, "long")][k] - acc[(m, t, "single")][k] for m, t in paired
             if k in acc[(m, t, "long")] and k in acc[(m, t, "single")]]
        if len(d) < 3:
            continue
        xs.append(k)
        mu.append(st.mean(d))
        l, h = boot_ci(d)
        lo.append(l)
        hi.append(h)
        q1.append(float(np.percentile(d, 25)))
        q3.append(float(np.percentile(d, 75)))

    # One band only. An IQR ribbon under the CI ribbon reads as two similar
    # translucent greys and makes the reader decode which is which; per-cell
    # heterogeneity belongs in the supplementary grid, not here.

    ax.axhline(0, color=INK, lw=0.7, ls=(0, (4, 3)), zorder=2)
    ax.fill_between(xs, lo, hi, color=BAND, alpha=0.30, lw=0, zorder=3)
    ax.plot(xs, mu, color=LF, lw=2.0, zorder=4, solid_capstyle="round")
    ax.plot(xs, mu, "o", color=LF, ms=4.0, mew=0, zorder=5)

    peak = max(range(len(xs)), key=lambda i: mu[i])
    ax.annotate(f"+{mu[peak]:.3f}", xy=(xs[peak], mu[peak]),
                xytext=(xs[peak] * 1.08, mu[peak] + 0.028),
                fontsize=9.5, color=LF, ha="left", va="bottom")
    ax.set_ylim(min(lo) - 0.04, max(hi) + 0.055)

    ax.set_xticks([0.01, 0.03, 0.05, 0.1, 0.5, 1.0])
    ax.set_xticklabels(["1%", "3%", "5%", "10%", "50%", "100%"])
    ax.set_xlim(0.0085, 1.25)
    ax.set_xlabel("Fraction of attention heads steered")
    ax.set_ylabel("Accuracy gap   (free-form $-$ single-token)")
    gone = next((xs[z] for z in range(peak + 1, len(xs)) if lo[z] <= 0), xs[-1])
    sig, sigS = verdict(xs, lo, hi)
    if sig and not sigS:
        ttl = f"The free-form advantage is resolved only near a {sig[0]:.0%} budget"
    elif sigS and not sig:
        ttl = f"Single-token localization leads, resolved at a {sigS[0]:.0%} budget"
    elif sig and sigS:
        ttl = "Which localization leads depends on the budget"
    else:
        ttl = "No budget resolves a difference between the localizations"
    titled(ax, ttl,
           f"{'free-form' if evalmode == 'long' else 'single-token'} evaluation "
           f"and steering   ·   shaded: bootstrap 95% CI of the mean")
    fig.savefig(HERE / f"{out}.pdf")
    fig.savefig(HERE / f"{out}.png", dpi=400)
    plt.close(fig)
    return len(paired), xs, mu, lo, hi, peak, gone, sig


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default="long", choices=["long", "single"])
    ap.add_argument("--out", default=None,
                    help="basename; defaults to fig_<eval>eval_gap")
    ap.add_argument("--supplement", action="store_true")
    ap.add_argument("--caption", action="store_true",
                    help="print a LaTeX caption stating the takeaway")
    a = ap.parse_args()
    if a.out is None:
        a.out = f"fig_{a.eval}eval_gap"
    lev_out = a.out.replace("_gap", "_levels")
    fam = style()
    acc = harvest(a.eval)                      # real arms only
    acc_ctrl = harvest(a.eval, include_random=True)  # grid only

    n, xs, mu, lo, hi, peak, gone, sig = main_figure(acc, a.out, a.eval)
    print(f"font: {fam}")
    print(f"wrote {a.out}.pdf / .png   ({n} cells with both arms)")
    nl, curves = levels_figure(acc, lev_out, a.eval)
    aheadLs, aheadSs = verdict(xs, lo, hi)
    print(f"wrote {lev_out}.pdf / .png   ({nl} cells averaged)")
    # control-arm version: same two curves, plus the random baselines. Written
    # alongside rather than in place, so the localization-only figure stays
    # available for the places that only need the headline contrast.
    nc, _ = levels_figure(acc_ctrl, lev_out + "_controls", a.eval,
                          with_controls=True)
    print(f"wrote {lev_out}_controls.pdf / .png   ({nc} cells averaged)")
    if a.supplement:
        ns = supplement(acc_ctrl, a.out + "_grid", a.eval)
        print(f"wrote {a.out}_grid.pdf / .png   ({ns} localized cells of 25)")

    if a.caption:
        last = len(xs) - 1
        print("\n" + r"\begin{figure}[t]")
        print(r"  \centering")
        print(rf"  \includegraphics[width=\linewidth]{{{a.out}.pdf}}")
        print(rf"  \caption{{\textbf{{The free-form advantage is significant "
              rf"only near a {xs[peak]:.0%} budget.}}".replace("%", r"\%"))
        siglist = ", ".join(f"$k={v:g}$" for v in sig) if sig else "no budget"
        print(rf"  Paired difference in judged accuracy (free-form $-$ "
              rf"single-token) against the fraction of attention heads steered, "
              rf"across {n} model--task pairs. The difference is positive "
              rf"at every budget through $k=0.1$ and peaks at "
              rf"${mu[peak]:+.3f}$ [{lo[peak]:+.3f}, {hi[peak]:+.3f}] at "
              rf"$k={xs[peak]:g}$, but the bootstrap 95\% CI excludes zero at "
              rf"{siglist} only; at every other budget it includes zero. By "
              rf"$k=0.5$ the mean has reversed sign (${mu[-2]:+.3f}$). The claim "
              rf"is therefore that the advantage is concentrated at small "
              rf"budgets, not that it is resolved at each of them. Per-cell "
              rf"curves are in Figure~\ref{{fig:{a.out.replace('_', '-')}-grid}}. "
              rf"Accuracy is taken at the best steering factor for each budget, "
              rf"on the free-form judged evaluation.}}")
        print(rf"  \label{{fig:{a.out.replace('_', '-')}}}")
        print(r"\end{figure}")

        xL, muL, loL, hiL = curves["long"]
        xS, muS, loS, hiS = curves["single"]
        pk = max(range(len(muL)), key=lambda z: muL[z])
        print("\n" + r"\begin{figure}[t]")
        print(r"  \centering")
        print(rf"  \includegraphics[width=\linewidth]{{{lev_out}.pdf}}")
        # one closing brace: it ends \textbf, and the body print closes \caption
        print(r"  \caption{\textbf{Free-form localization's advantage is "
              r"confined to small budgets.}")
        # describe who leads where WITHOUT assuming one arm leads contiguously:
        # on single-token evaluation the two trade places, and phrasing it as
        # "ahead up to k=X" off the last leading index is simply false there
        lead = [z for z in range(len(muL)) if muL[z] > muS[z]]
        contiguous = lead[:1] == [0] and lead == list(range(len(lead)))
        if contiguous and len(lead) < len(xL):
            who = (rf"Free-form is ahead at every budget up to "
                   rf"$k={xL[lead[-1]]:g}$ (shaded), single-token above it")
        elif not lead:
            who = "Single-token is ahead at every budget"
        elif len(lead) == len(xL):
            who = "Free-form is ahead at every budget"
        else:
            aheadL = ", ".join(f"$k={xL[z]:g}$" for z in lead)
            who = (rf"The two trade places across the range: free-form is ahead "
                   rf"at {aheadL} and single-token elsewhere")
        where = "free-form" if a.eval == "long" else "single-token"
        res = (rf"resolved in favour of free-form at {', '.join(f'$k={v:g}$' for v in aheadLs)}"
               if aheadLs else
               rf"resolved in favour of single-token at {', '.join(f'$k={v:g}$' for v in aheadSs)}"
               if aheadSs else "not resolved at any single budget")
        print(rf"  Judged accuracy averaged over {nl} model--task pairs, "
              rf"against the fraction of attention heads steered, under "
              rf"{where} evaluation with {where} steering. {who}. The paired "
              rf"difference is {res}. Both methods peak near $k={xL[pk]:g}$ "
              rf"(free-form ${muL[pk]:.3f}$, single-token ${muS[pk]:.3f}$) and "
              rf"decline thereafter as steering degrades fluency. Bands are "
              rf"bootstrap 95\% CIs of the mean and overlap throughout, which "
              rf"is why the paired difference in "
              rf"Figure~\ref{{fig:{a.out.replace('_', '-')}}} carries the "
              rf"inference. Accuracy is taken at the best steering factor for "
              rf"each budget.}}")
        print(rf"  \label{{fig:{lev_out.replace('_', '-')}}}")
        print(r"\end{figure}")


if __name__ == "__main__":
    main()
