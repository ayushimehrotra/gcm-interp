#!/usr/bin/env python3
"""Alignment as a function of n, swept across every legal rank.

WHY THIS EXISTS
---------------
`svcca_cosmat.py` draws the cosine matrix at a handful of chosen n. But n is a
CHOICE, and picking two or three of them invites the question this script
answers directly: what does the alignment look like at ALL of them?

"All n" in the literal sense -- keep every singular vector -- is not an option,
and the reason is worth stating plainly. X_L is (m x 10240): 10240 features in
an m-dimensional sample space, so its column space is all of R^(m-1). So is
X_S's. Two subspaces that are both "everything" are identical, and every cosine
is exactly 1. Measured on pure noise at m=200: n=199 gives mean rho = 1.000000
and min rho = 1.000000. The heatmap would be solid white on any data at all.

So the sweep runs n = 1 .. rank, and the curve is read against two things:

    the shuffled floor   the same statistic with one arm's rows permuted. It
                         rises with n and reaches 1 at full rank, which is what
                         makes the raw curve uninterpretable on its own.
    the degeneracy bound 2n >= m-1, past which the leading cosines are pinned
                         at 1 by dimension counting. Shaded on every panel.

WHAT IS PLOTTED
---------------
    mean rho        mean of ALL singular values of M at that n
    mean rho^2      = ||M||_F^2 / n, the fraction of the long arm's top-n
                    subspace lying inside the single arm's. Analytic null n/d.
    min rho         the smallest singular value -- the only one of the three
                    that can say whether the subspaces actually COINCIDE, since
                    a mean trades a lost direction against a perfect one.

USAGE
-----
    python cosmat_sweep_n.py                       # needs --save_bases output
    python cosmat_sweep_n.py --indir cosmat_figs_sweep --shaping b_last_h
"""
import argparse
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

SEQ_BLUE = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
            "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281",
            "#0d366b"]

REPO = Path(__file__).resolve().parent
TASKS = [("persona", "Persona"), ("bias", "Bias"), ("verse", "Verse"),
         ("factual-recall", "Factual recall"),
         ("summarization", "Summarization")]
SHAPINGS = ("bs_h", "s_h", "b_last_h")

INK = {"light": dict(surface="#fcfcfb", plane="#f9f9f7", primary="#0b0b0b",
                     secondary="#52514e", muted="#898781", grid="#e1e0d9",
                     axis="#c3c2b7", band="#f0efec"),
       "dark": dict(surface="#1a1a19", plane="#0d0d0d", primary="#ffffff",
                    secondary="#c3c2b7", muted="#898781", grid="#2c2c2a",
                    axis="#383835", band="#2c2c2a")}
# categorical slots 1-3 of the reference palette, light / dark
HUES = {"light": ["#2a78d6", "#eb6834", "#1baf7a"],
        "dark": ["#3987e5", "#d95926", "#199e70"]}


def sweep(QL, QS, perm, ns):
    """(mean rho, mean rho^2, min rho, and the same for the shuffle) per n.

    One SVD per n of an n x n matrix -- cheap. The bases themselves are never
    recomputed: U does not depend on n, so QL[:, :n] is the rank-n basis.
    """
    out = {k: [] for k in ("mean", "sq", "min", "fmean", "fsq", "fmin")}
    QSp = QS[perm]
    for n in ns:
        r = np.linalg.svd(QL[:, :n].T @ QS[:, :n], compute_uv=False)
        f = np.linalg.svd(QL[:, :n].T @ QSp[:, :n], compute_uv=False)
        r = np.clip(r, 0, 1); f = np.clip(f, 0, 1)
        out["mean"].append(r.mean());   out["fmean"].append(f.mean())
        out["sq"].append((r ** 2).mean()); out["fsq"].append((f ** 2).mean())
        out["min"].append(r.min());     out["fmin"].append(f.min())
    return {k: np.array(v) for k, v in out.items()}


STATS = [("mean", "mean \u03c1"),
         ("sq", "mean \u03c1\u00b2"),
         ("min", "min \u03c1")]


def draw(data, shaping, path, mode):
    """Rows = statistic, columns = task, one figure per shaping.

    Stacking the three statistics puts them on a shared x-axis so a single
    column reads as one task's full story: how much overlap there is (mean),
    what fraction of the subspace that is (rho^2), and whether ANY direction is
    unshared (min) -- which the two means structurally cannot show.
    """
    ink = INK[mode]; hues = HUES[mode]
    nr, nc = len(STATS), len(data)
    fig, axes = plt.subplots(nr, nc, figsize=(2.55 * nc + 0.9, 2.35 * nr + 0.9),
                             squeeze=False, sharey=True)
    fig.patch.set_facecolor(ink["plane"])
    for i, (stat, slabel) in enumerate(STATS):
        for j, (name, d) in enumerate(data):
            ax = axes[i][j]
            ax.set_facecolor(ink["surface"])
            ns, m = d["ns"], d["m"]
            bound = (m - 1) / 2.0
            if bound < ns[-1]:
                ax.axvspan(bound, ns[-1], color=ink["band"], zorder=0, lw=0)
                if i == 0:
                    ax.text(bound, 1.03, " degenerate", fontsize=6.8,
                            color=ink["muted"], ha="left", va="bottom")
            ax.plot(ns, d[stat], "-", color=hues[0], lw=1.7, label="observed")
            ax.plot(ns, d["f" + stat], "--", color=hues[1], lw=1.5,
                    label="shuffled")
            if stat == "sq":
                ax.plot(ns, ns / (m - 1), ":", color=ink["muted"], lw=1.3,
                        label="chance  n/d")
            if i == 0:
                ax.set_title(f"{name}\n{m} rows", fontsize=9.5,
                             color=ink["primary"], loc="left", pad=7)
            if i == nr - 1:
                ax.set_xlabel("n  (components kept)", fontsize=8.5,
                              color=ink["secondary"])
            if j == 0:
                ax.set_ylabel(slabel, fontsize=10, color=ink["secondary"])
            ax.set_ylim(0, 1.06); ax.set_xlim(ns[0], ns[-1])
            ax.grid(axis="y", color=ink["grid"], lw=0.8)
            ax.set_axisbelow(True)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            for sp in ("left", "bottom"):
                ax.spines[sp].set_color(ink["axis"])
                ax.spines[sp].set_linewidth(0.8)
            ax.tick_params(labelsize=7.5, colors=ink["muted"], length=0)
            if i == 1 and j == nc - 1:
                leg = ax.legend(fontsize=7.5, frameon=False, loc="upper left")
                for t in leg.get_texts():
                    t.set_color(ink["secondary"])
    fig.suptitle(f"Alignment across every legal rank \u2014 {shaping}",
                 fontsize=12.5, color=ink["primary"], x=0.007, ha="left",
                 y=0.995)
    fig.text(0.007, 0.008,
             "\u03c1 are the singular values of M at that n. min \u03c1 is the only "
             "one that can say whether the subspaces COINCIDE \u2014 a mean trades a "
             "lost direction against a perfect one.\n"
             "Shaded: 2n \u2265 rows\u22121, where the leading cosines are pinned at 1 "
             "on any data. Full rank is solid white on noise, which is why n exists.",
             fontsize=7.2, color=ink["muted"], ha="left", va="bottom",
             linespacing=1.5)
    fig.tight_layout(rect=[0, 0.055, 1, 0.945])
    fig.savefig(path, dpi=170, facecolor=ink["plane"])
    plt.close(fig)
    return path


def draw_fulln(entries, task, path, mode):
    """The cosine matrix itself at full rank, beside the largest legal rank.

    Rows = shaping. Left column is the biggest n that clears 2n < rows-1; right
    column keeps EVERY component. The right column is what "all n" means, and
    it is shown with its degeneracy stated rather than withheld -- past the
    bound the leading cosines are pinned at 1 by dimension counting, which is
    visible as the bright diagonal filling in.
    """
    ink = INK[mode]
    seq = LinearSegmentedColormap.from_list("blue_seq", SEQ_BLUE)
    nr = len(entries)
    fig, axes = plt.subplots(nr, 2, figsize=(8.2, 3.5 * nr + 1.1),
                             squeeze=False)
    fig.patch.set_facecolor(ink["plane"])
    im = None
    for i, (shaping, QL, QS, m) in enumerate(entries):
        top = min(QL.shape[1], QS.shape[1], m - 1)
        safe = max(1, (m - 1) // 2 - 1)
        for j, n in enumerate((min(safe, top), top)):
            ax = axes[i][j]
            ax.set_facecolor(ink["surface"])
            M = np.abs(QL[:, :n].T @ QS[:, :n])
            im = ax.imshow(M, cmap=seq, vmin=0, vmax=1,
                           interpolation="nearest", aspect="equal")
            deg = 2 * n >= m - 1
            for sp in ax.spines.values():
                sp.set_color("#d03b3b" if deg else ink["axis"])
                sp.set_linewidth(1.8 if deg else 0.8)
            head = f"{shaping}   n={n}" + ("   \u26a0 DEGENERATE" if deg else "")
            ax.set_title(head, fontsize=10, color=ink["primary"], loc="left",
                         pad=18)
            ax.text(0.0, 1.012,
                    f"rows={m}   |diag|={np.abs(np.diag(M)).mean():.2f}   "
                    f"mean \u03c1={np.linalg.svd(M, compute_uv=False).mean():.2f}",
                    transform=ax.transAxes, fontsize=7.3, color=ink["muted"],
                    va="bottom")
            step = max(1, n // 8)
            ticks = list(range(0, n, step))
            ax.set_xticks(ticks); ax.set_yticks(ticks)
            ax.set_xticklabels([str(t + 1) for t in ticks], fontsize=7,
                               color=ink["muted"])
            ax.set_yticklabels([str(t + 1) for t in ticks], fontsize=7,
                               color=ink["muted"])
            ax.tick_params(length=0)
            if i == nr - 1:
                ax.set_xlabel("single-arm component", fontsize=8.5,
                              color=ink["secondary"])
            if j == 0:
                ax.set_ylabel("long-arm component", fontsize=8.5,
                              color=ink["secondary"])
    fig.suptitle(f"Cosine similarity matrix at full rank \u2014 {task}",
                 fontsize=12.5, color=ink["primary"], x=0.01, ha="left",
                 y=0.995)
    fig.text(0.01, 0.955,
             "left: the largest n that clears the degeneracy bound   \u00b7   "
             "right: every component",
             fontsize=8.5, color=ink["secondary"], ha="left", va="top")
    fig.tight_layout(rect=[0, 0.035, 0.90, 0.945])
    cax = fig.add_axes([0.915, 0.12, 0.016, 0.6])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("|cos| between singular vectors", fontsize=8.5,
                 color=ink["secondary"])
    cb.ax.tick_params(labelsize=7.5, colors=ink["muted"], length=0)
    cb.outline.set_edgecolor(ink["axis"]); cb.outline.set_linewidth(0.6)
    fig.text(0.01, 0.008,
             "Red border: 2n \u2265 rows\u22121, where dimension counting forces "
             "2n\u2212(rows\u22121) of the cosines to 1 on any data at all.",
             fontsize=7.2, color=ink["muted"], ha="left", va="bottom")
    fig.savefig(path, dpi=170, facecolor=ink["plane"])
    plt.close(fig)
    return path


def draw_ladder(rows_data, shaping, path, mode):
    """The cosine matrix across a ladder of n, up to full rank.

    Rows = task, columns = n. The last column keeps EVERY component, which is
    what "all n" means; because each task's activation matrix has its own rank
    that column's n differs per row and is labelled per panel.

    Past 2n >= rows-1 dimension counting forces 2n-(rows-1) of the cosines to 1
    on any data at all, so those panels get a red border. The bright diagonal
    filling in as you move right IS that effect, not a stronger result.
    """
    ink = INK[mode]
    seq = LinearSegmentedColormap.from_list("blue_seq", SEQ_BLUE)
    caps = [min(QL.shape[1], QS.shape[1], m - 1) for _, QL, QS, m in rows_data]
    ladder = [n for n in (10, 25, 50, 100, 200, 400) if n <= min(caps)]
    ncol = len(ladder) + 1
    nr = len(rows_data)
    fig, axes = plt.subplots(nr, ncol, figsize=(2.35 * ncol + 1.5,
                                                2.35 * nr + 1.5), squeeze=False)
    fig.patch.set_facecolor(ink["plane"])
    im = None
    for i, (name, QL, QS, m) in enumerate(rows_data):
        top = caps[i]
        for j, n in enumerate(ladder + [top]):
            ax = axes[i][j]
            ax.set_facecolor(ink["surface"])
            M = np.abs(QL[:, :n].T @ QS[:, :n])
            im = ax.imshow(M, cmap=seq, vmin=0, vmax=1,
                           interpolation="nearest", aspect="equal")
            deg = 2 * n >= m - 1
            for sp in ax.spines.values():
                sp.set_color("#d03b3b" if deg else ink["axis"])
                sp.set_linewidth(1.6 if deg else 0.8)
            tag = f"n={n}" + ("  (all)" if j == ncol - 1 else "")
            ax.set_title(tag, fontsize=9, color=ink["primary"], loc="left",
                         pad=13)
            ax.text(0.0, 1.01,
                    f"|diag|={np.abs(np.diag(M)).mean():.2f}",
                    transform=ax.transAxes, fontsize=6.8, color=ink["muted"],
                    va="bottom")
            ax.set_xticks([]); ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(f"{name}\n{m} rows", fontsize=9,
                              color=ink["secondary"])
    fig.suptitle(f"Cosine similarity matrix across every n \u2014 {shaping}",
                 fontsize=12.5, color=ink["primary"], x=0.008, ha="left",
                 y=0.995)
    fig.text(0.008, 0.955,
             "columns are the truncation rank n; the last column keeps every "
             "component. Rows are tasks.",
             fontsize=8.5, color=ink["secondary"], ha="left", va="top")
    fig.tight_layout(rect=[0, 0.05, 0.90, 0.945])
    cax = fig.add_axes([0.915, 0.14, 0.014, 0.6])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("|cos| between singular vectors", fontsize=8.5,
                 color=ink["secondary"])
    cb.ax.tick_params(labelsize=7.5, colors=ink["muted"], length=0)
    cb.outline.set_edgecolor(ink["axis"]); cb.outline.set_linewidth(0.6)
    fig.text(0.008, 0.008,
             "Red border: 2n \u2265 rows\u22121. There, dimension counting forces "
             "2n\u2212(rows\u22121) of the cosines to exactly 1 on ANY data \u2014 the "
             "diagonal brightening to the right is that, not a stronger result.",
             fontsize=7.2, color=ink["muted"], ha="left", va="bottom")
    fig.savefig(path, dpi=170, facecolor=ink["plane"])
    plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--indir", default="cosmat_figs_sweep")
    ap.add_argument("--outdir", default="cosmat_figs_sweep")
    ap.add_argument("--model", default="Qwen1.5-14B-Chat")
    ap.add_argument("--shapings", default=",".join(SHAPINGS))
    ap.add_argument("--mode", default="light", choices=("light", "dark"))
    ap.add_argument("--ladder", action="store_true",
                    help="heatmap of M across a ladder of n up to full rank")
    ap.add_argument("--heatmaps", action="store_true",
                    help="draw the cosine matrix itself at full rank, per task")
    a = ap.parse_args()
    ind, outd = REPO / a.indir, REPO / a.outdir
    outd.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    shs = [s for s in a.shapings.split(",") if s in SHAPINGS]
    if a.ladder:
        for shaping in shs:
            rd = []
            for slug, name in TASKS:
                f = ind / f"cosmat_{a.model}_{slug}_prompt.npz"
                if not f.exists():
                    continue
                z = np.load(f)
                if f"{shaping}_QL" not in z.files:
                    continue
                QL, QS = z[f"{shaping}_QL"], z[f"{shaping}_QS"]
                rd.append((name, QL, QS, QL.shape[0]))
            if rd:
                print("  wrote", draw_ladder(
                    rd, shaping, outd / f"ladder_{shaping}.png", a.mode))
        return
    if a.heatmaps:
        for slug, name in TASKS:
            f = ind / f"cosmat_{a.model}_{slug}_prompt.npz"
            if not f.exists():
                continue
            z = np.load(f)
            entries = [(sh, z[f"{sh}_QL"], z[f"{sh}_QS"], z[f"{sh}_QL"].shape[0])
                       for sh in shs if f"{sh}_QL" in z.files]
            if entries:
                print("  wrote", draw_fulln(
                    entries, name, outd / f"fulln_{slug}.png", a.mode))
        return

    for shaping in shs:
        data = []
        for slug, name in TASKS:
            f = ind / f"cosmat_{a.model}_{slug}_prompt.npz"
            if not f.exists():
                continue
            z = np.load(f)
            if f"{shaping}_QL" not in z.files:
                print(f"  {f.name}: no stored bases -- rerun with --save_bases")
                return
            QL, QS = z[f"{shaping}_QL"], z[f"{shaping}_QS"]
            m = QL.shape[0]
            # A centred m-row matrix has rank <= m-1; _basis keeps anything
            # above 1e-8*s[0], which can admit one spurious null direction on
            # the small shapings. Cap so the sweep never plots it.
            top = min(QL.shape[1], QS.shape[1], m - 1)
            # every legal rank, thinned once the curve is smooth so the sweep
            # stays quick on the 10k-row shapings
            ns = list(range(1, min(top, 60) + 1)) + \
                 list(range(65, top + 1, 5))
            ns = [n for n in ns if n <= top]
            d = sweep(QL, QS, rng.permutation(m), ns)
            d["ns"] = np.array(ns); d["m"] = m
            data.append((name, d))
            print(f"  {shaping:<9}{name:<16} rows={m:<6} swept n=1..{ns[-1]}")
        if not data:
            continue
        print("  wrote", draw(data, shaping,
                              outd / f"sweep_{shaping}.png", a.mode))


if __name__ == "__main__":
    main()
