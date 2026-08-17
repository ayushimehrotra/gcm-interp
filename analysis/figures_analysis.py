#!/usr/bin/env python3
"""Analysis figures: how long-form and single-token localization differ.

Five single-panel options, each with one takeaway, sharing the palette and
Palatino setup of figures.py. Every number is read from the analysis CSVs or
recomputed from the attribution fields; nothing is hardcoded.

  depth      where in the network each method selects        (FINDINGS.md 2)
  shared     disjoint units, shared information              (FINDINGS.md 4)
  dose       long-form emerges as supervision accumulates    (FINDINGS.md 6)
  contain    whose ranking subsumes whose                    (FINDINGS.md 5)
  forest     every paired difference at once, standardized   (FINDINGS.md 1-8)

Design rules follow the same review feedback as figures.py: declarative titles,
minimum ink, direct labels over legends, one panel per figure. The forest plot is
the one exception to "no cross-referencing" -- comparing effects IS its content,
and a forest plot is the standard form for that.

STANDARDIZED EFFECTS (forest)
-----------------------------
The measures live in different units (layers, nats, probabilities, AUC), so the
forest plot shows the standardized paired difference

    d = mean(LF - ST) / sd(LF - ST)

over cells, with a bootstrap 95% CI. This is a within-cell effect size: it asks
how large the difference is relative to how much it varies across model/task,
which is the right question when the cells differ enormously in difficulty.
Sign is oriented so positive always means "long-form more so".
"""
import argparse
import csv
import glob
import statistics as st
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import figures as F          # palette, fonts, harvest helpers
from figures import titled, EXCLUDED

HERE = Path(__file__).parent
INK, LF, ST, HAIR = F.INK, F.LF, F.ST, F.HAIR
SHORT = F.SHORT
MODELS, TASKS = F.MODELS, F.TASKS


def read(name):
    """Paper models only -- Falcon3-10B is excluded (see figures.EXCLUDED)."""
    p = HERE / name
    if not p.exists():
        return []
    return [r for r in csv.DictReader(open(p))
            if r.get("model") not in EXCLUDED]


def boot(d, n=10000, seed=0):
    if len(d) < 2:
        return float("nan"), float("nan")
    r = np.random.default_rng(seed)
    a = np.asarray(d, float)
    m = np.sort(r.choice(a, (n, len(a))).mean(1))
    return float(m[int(.025 * n)]), float(m[int(.975 * n)])


def boot_d(d, n=10000, seed=0):
    """Bootstrap CI of the standardized mean difference."""
    a = np.asarray(d, float)
    if len(a) < 3 or a.std(ddof=1) == 0:
        return float("nan"), float("nan"), float("nan")
    r = np.random.default_rng(seed)
    idx = r.integers(0, len(a), (n, len(a)))
    s = a[idx]
    sd = s.std(1, ddof=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        ds = np.where(sd > 0, s.mean(1) / sd, np.nan)
    ds = np.sort(ds[~np.isnan(ds)])
    return (float(a.mean() / a.std(ddof=1)),
            float(ds[int(.025 * len(ds))]), float(ds[int(.975 * len(ds))]))


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


# ------------------------------------------------------------------ 1. depth
def fig_depth(out="fig_depth_profile", k=0.1):
    """Back-to-back histograms of selected-block depth.

    Drawn with stairs(), not fill_between() on bin centres: interpolating
    between centres turns a histogram into a jagged polygon and invents slopes
    the data does not contain. Budget k=0.1 matches the layer statistics
    reported in FINDINGS.md 2, so figure and text agree.
    """
    import position_count_mechanism as P
    bins = np.linspace(0, 1, 21)
    accL, accS, means = [], [], {"long": [], "single": []}
    for model in MODELS:
        fields = P.load_fields(model)
        for task in TASKS:
            if (task, "long") not in fields or (task, "single") not in fields:
                continue
            for arm, bucket in (("long", accL), ("single", accS)):
                A = fields[(task, arm)]
                sel = P.top_units(A, k)
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


# ----------------------------------------------------------------- 2. shared
def fig_shared(out="fig_shared_information"):
    """Two quantities per cell, connected. A scatter of one against the other
    hides that the gap is the point; the connector makes it the subject."""
    rows = read("activation_geometry.csv")
    if not rows:
        return
    jac = [float(r["jaccard"]) for r in rows]
    rho = [float(r["rho_mean"]) for r in rows]
    perm = st.mean(float(r["rho_perm"]) for r in rows)

    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    bare(ax)
    x0, x1 = 0, 1
    for a, b in zip(jac, rho):
        ax.plot([x0, x1], [a, b], color=HAIR, lw=0.7, alpha=0.8, zorder=1)
    ax.plot([x0] * len(jac), jac, "o", color=ST, ms=4, mew=0, zorder=3)
    ax.plot([x1] * len(rho), rho, "o", color=LF, ms=4, mew=0, zorder=3)
    ax.axhline(perm, xmin=0.5, color=INK, lw=0.8, ls=(0, (3, 3)), alpha=0.55)
    ax.text(x1 - 0.02, perm - 0.035, "chance level for shared information",
            ha="right", va="top", fontsize=8, color=INK, alpha=0.6)
    ax.text(x0, st.mean(jac) - 0.075, f"{st.mean(jac):.0%}", ha="center",
            fontsize=11, color=ST)
    ax.text(x1, st.mean(rho) + 0.045, f"{st.mean(rho):.0%}", ha="center",
            fontsize=11, color=LF)
    ax.set_xlim(-0.35, 1.35)
    ax.set_ylim(-0.02, 1.12)
    ax.set_xticks([x0, x1])
    ax.set_xticklabels(["heads the two methods\nselect in common",
                        "information their\nselections share"], fontsize=9.5)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_yticklabels(["0%", "50%", "100%"])
    ax.tick_params(axis="x", length=0, pad=6)
    titled(ax, "The two methods pick different heads that carry the same "
           "information", f"{len(jac)} model–task pairs")
    save(fig, out)
    return st.mean(jac), st.mean(rho), perm


# ------------------------------------------------------------------- 3. dose
def fig_dose(out="fig_supervision_dose"):
    """Every cell's own trajectory, because monotone-in-21/21 is the result;
    a mean line alone would not show that it holds individually."""
    rows = read("position_count.csv") + read("position_count_umang.csv")
    if not rows:
        return
    cells = defaultdict(dict)
    for r in rows:
        cells[(r["model"], r["task"])][r["T"]] = float(r["rho_vs_LF"])
    order = ["1", "4", "16", "all"]
    xs = [1, 4, 16, 50]
    fig, ax = plt.subplots(figsize=(5.4, 3.5))
    bare(ax)
    ax.set_xscale("log")
    keep = [c for c in cells.values() if all(t in c for t in order)]
    for c in keep:
        ax.plot(xs, [c[t] for t in order], color=HAIR, lw=0.7, alpha=0.85,
                zorder=1)
    mu = [st.mean(c[t] for c in keep) for t in order]
    ax.plot(xs, mu, color=LF, lw=2.2, zorder=3, solid_capstyle="round")
    ax.plot(xs, mu, "o", color=LF, ms=4.5, mew=0, zorder=4)
    ax.set_xticks(xs)
    ax.set_xticklabels(["1", "4", "16", "all\n(~50)"])
    ax.set_xlim(0.75, 75)
    ax.set_ylim(-0.15, 1.02)
    ax.set_xlabel("Response tokens the attribution objective sums over")
    ax.set_ylabel("Agreement with free-form localization")
    titled(ax, "Long-form localization is what emerges as supervision "
           "accumulates",
           f"{len(keep)} model–task pairs, every one increasing")
    save(fig, out)
    return mu, len(keep)


# --------------------------------------------------------------- 4. contain
def fig_contain(out="fig_containment"):
    """One row per pair, sorted. The asymmetry is per-pair evidence, so the
    figure shows every pair rather than two summary bars."""
    rows = read("containment_asymmetry.csv")
    if not rows:
        return
    rows.sort(key=lambda r: float(r["auc_S_in_L"]) - float(r["auc_L_in_S"]))
    fig, ax = plt.subplots(figsize=(5.6, 5.4))
    bare(ax)
    ys = range(len(rows))
    for y, r in zip(ys, rows):
        a, b = float(r["auc_L_in_S"]), float(r["auc_S_in_L"])
        ax.plot([a, b], [y, y], color=HAIR, lw=1.0, zorder=1)
        ax.plot(a, y, "o", color=LF, ms=4, mew=0, zorder=3)
        ax.plot(b, y, "o", color=ST, ms=4, mew=0, zorder=3)
    ax.axvline(0.5, color=INK, lw=0.8, ls=(0, (3, 3)), alpha=0.6)
    ax.set_yticks(list(ys))
    ax.set_yticklabels([f"{SHORT[r['model']]}  ·  {r['task']}" for r in rows],
                       fontsize=7.5)
    ax.set_ylim(-0.8, len(rows) - 0.2)
    ax.set_xlabel("How highly the other method ranks these heads")
    mL = st.mean(float(r["auc_L_in_S"]) for r in rows)
    mS = st.mean(float(r["auc_S_in_L"]) for r in rows)
    # a legend below the axis rather than labels inside it: with 21 rows the
    # plotting area has no reliably empty region, and in-axes labels collided
    # with both the subtitle and the chance line
    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([], [], marker="o", ls="none", color=LF, ms=5,
               label="long-form's picks, ranked by single-token"),
        Line2D([], [], marker="o", ls="none", color=ST, ms=5,
               label="single-token's picks, ranked by long-form")],
        loc="upper center", bbox_to_anchor=(0.5, -0.125), ncol=1,
        fontsize=8.5, handletextpad=0.6, borderpad=0)
    titled(ax, "Long-form's ranking already contains single-token's picks;\n"
           "the reverse is not true", f"{len(rows)} model–task pairs   ·   "
           f"dashed line: chance")
    save(fig, out)
    return mL, mS


# ---------------------------------------------------------------- 5. forest
def fig_forest(out="fig_effect_summary"):
    """Standardized paired effects. Comparing effects IS the content here, so a
    multi-row layout is the right form rather than a violation of the one-panel
    rule."""
    eff = []

    lay = defaultdict(dict)
    for r in read("layerwise_mechanics.csv"):
        lay[(r["model"], r["task"])][r["arm"]] = r
    for lbl, key, sign in (("selects earlier", "mean_depth", -1),
                           ("uses fewer layers", "layers_used", -1),
                           ("more concentrated by layer", "top3_mass", +1)):
        d = [sign * (float(v["long"][key]) - float(v["single"][key]))
             for v in lay.values() if "long" in v and "single" in v]
        eff.append((lbl, d, "placement"))

    con = defaultdict(dict)
    for r in read("localization_concentration.csv"):
        con[(r["model"], r["task"])][r["arm"]] = r
    d = [float(v["long"]["cap0.03"]) - float(v["single"]["cap0.03"])
         for v in con.values() if "long" in v and "single" in v]
    eff.append(("captures more effect at a 3% budget", d, "placement"))

    geo = read("activation_geometry.csv")
    eff.append(("heads are concept-denser",
                [float(r["density_L"]) - float(r["density_S"]) for r in geo],
                "geometry"))
    eff.append(("shared subspace carries the concept",
                [float(r["lab_L"]) - float(r["lab_S"]) for r in geo],
                "content"))

    cnt = read("containment_asymmetry.csv")
    eff.append(("its ranking subsumes the other's",
                [float(r["auc_S_in_L"]) - float(r["auc_L_in_S"]) for r in cnt],
                "geometry"))

    # these two globs bypass read(), so apply the same model filter here or
    # the forest mixes a 19-pair grid with rows that still contain Falcon
    # necessity_dose.py writes necessity_<margin>.csv. The dose_<model>.csv files
    # are orphaned output from an earlier naming that no current script produces,
    # so globbing them meant this panel kept showing pre-pull numbers however
    # often the analysis was rerun.
    dose = []
    for f in [str(HERE / "necessity_long.csv")]:
        dose += [r for r in csv.DictReader(open(f))
                 if r.get("model") not in EXCLUDED]
    dd = defaultdict(dict)
    for r in dose:
        dd[(r["model"], r["task"], r["topk"])][r["arm"]] = float(r["d_ablate"])
    eff.append(("ablation hurts more",
                [-(v["long"] - v["single"]) for v in dd.values()
                 if "long" in v and "single" in v], "geometry"))

    pr = defaultdict(dict)
    # probe_units.py writes ONE combined probe_units.csv. Globbing "probe_*.csv"
    # also matches the superseded per-model probe_<model>.csv files left on disk
    # from an earlier run; since both carry the same (model, task, group) keys,
    # whichever the glob yielded last silently won and pre-pull numbers could
    # override current ones. Read the canonical file only.
    for f in [str(HERE / "probe_units.csv")]:
        for r in csv.DictReader(open(f)):
            if r.get("model") in EXCLUDED:
                continue
            pr[(r["model"], r["task"])][r["group"]] = float(r["probe_acc"])
    eff.append(("concept is linearly decodable",
                [v["LF-only"] - v["ST-only"] for v in pr.values()
                 if "LF-only" in v and "ST-only" in v], "content"))

    eff = [(l, d, g) for l, d, g in eff if len(d) >= 3]
    stats = [(l, *boot_d(d), len(d), g) for l, d, g in eff]
    stats.sort(key=lambda z: z[1])

    fig, ax = plt.subplots(figsize=(6.0, 3.9))
    bare(ax)
    for y, (lbl, m, lo, hi, n, grp) in enumerate(stats):
        col = INK if lo <= 0 <= hi else LF
        ax.plot([lo, hi], [y, y], color=col, lw=1.3, alpha=0.85, zorder=2,
                solid_capstyle="round")
        ax.plot(m, y, "o", color=col, ms=5, mew=0, zorder=3)
    ax.axvline(0, color=INK, lw=0.8, ls=(0, (3, 3)), alpha=0.6)
    ax.set_yticks(range(len(stats)))
    ax.set_yticklabels([s[0] for s in stats], fontsize=9)
    ax.set_ylim(-0.7, len(stats) - 0.3)
    ax.set_xlabel("Standardized difference, free-form $-$ single-token")
    titled(ax, "Long-form and single-token localization differ in placement,\n"
           "not in what the units encode",
           "bars: bootstrap 95% CI   ·   grey where the interval includes zero")
    save(fig, out)
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", default="all",
                    help="comma list: depth,shared,dose,contain,forest")
    a = ap.parse_args()
    F.style()
    want = ([w.strip() for w in a.which.split(",")]
            if a.which != "all" else
            ["depth", "shared", "dose", "contain", "forest"])
    out = {}
    if "depth" in want:
        out["depth"] = fig_depth()
    if "shared" in want:
        out["shared"] = fig_shared()
    if "dose" in want:
        out["dose"] = fig_dose()
    if "contain" in want:
        out["contain"] = fig_contain()
    if "forest" in want:
        st_ = fig_forest()
        out["forest"] = st_
        if st_:
            print("\nstandardized effects (largest first):")
            for lbl, m, lo, hi, n, grp in sorted(st_, key=lambda z: -abs(z[1])):
                flag = "" if lo <= 0 <= hi else "  *"
                print(f"  {lbl:<40}{m:+6.2f} [{lo:+.2f},{hi:+.2f}]  n={n}{flag}")


if __name__ == "__main__":
    main()
