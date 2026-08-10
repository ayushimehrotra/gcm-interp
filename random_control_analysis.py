"""Reporting for the random-head control experiment (CLAUDE.md section 6).

Reads judge-evals/accuracy/ and produces, per (model, task, eval mode):

  1. k-curves for real long-form, real single-token, uniform random and
     layer-matched random, with the random arms drawn as across-seed bands
  2. min-k to 80% of each arm's OWN ceiling -- the paper's precision metric
  3. a flag wherever either random arm is within noise of the real
     localizations at k <= 0.1

Conventions follow the existing plotting code:
  - the steering-factor axis is collapsed by max per topk (plots_collapsed.py)
  - long-form evals score on w_rf, single-token evals on wo_rf (plots.py)
All comparisons stay WITHIN an eval mode: long-form is judged and single-token
is token-matched, so the two are not comparable.

Usage:
  python random_control_analysis.py
  python random_control_analysis.py --models Falcon3-10B-Instruct --seeds 0 1 2
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent
ACCURACY_DIR = REPO / "judge-evals" / "accuracy"
RESULTS_DIR = REPO / "results"

TOPKS = [0.01, 0.03, 0.05, 0.07, 0.09, 0.1, 0.5, 1.0]

# Steering factors PER MODEL, matching that model's existing ATP sweep. Falcon3's
# ATP runs also cover N=15 and N=20; collapsing the two arms over different N
# sets would let the wider sweep win on max alone. Models not listed fall back to
# the CLAUDE.md sweep.
DEFAULT_SFS = [1, 2, 4, 5, 6, 8, 10]
MODEL_SFS = {
    "Falcon3-10B-Instruct": [1, 2, 4, 5, 6, 8, 10, 15, 20],
    "Qwen1.5-14B-Chat": [1, 2, 4, 5, 6, 8, 10],
    "gemma-3-12b-it": [1, 2, 4, 5, 6, 8, 10],
    "OLMo-2-1124-13B-DPO": [1, 2, 4, 5, 6, 8, 10],
    "Qwen1.5-32B-Chat": [1, 2, 4, 5, 6, 8, 10],
}


def sfs_for(model):
    return MODEL_SFS.get(model, DEFAULT_SFS)


def out_dir_for(model):
    """Model-level rollup, spanning that model's tasks."""
    return RESULTS_DIR / model / "random_control_report"


def task_out_dir(model, task):
    """Per-task outputs sit inside the task folder, alongside atp/ and the
    random-s*/ and randomlayer-s*/ arm trees, so every task directory is laid
    out the same way."""
    return RESULTS_DIR / model / task / "random_control_report"


def sibling_task(task, families):
    """The other localization in the same task family (long-form <-> single-token)."""
    for _, tasks in families.values():
        if task in tasks:
            others = [t for t in tasks if t != task]
            return others[0] if others else None
    return None

FAMILIES = {
    "verse": ("prose", ["from_verse-long_to_prose", "from_verse-single_to_prose"]),
    "paragraph": ("sentence", ["from_paragraph-long_to_sentence",
                               "from_paragraph-single_to_sentence"]),
}

ARMS = [
    ("real", "atp", "targeted"),
    ("uniform", "random", "random"),
    ("layer-matched", "randomlayer", "random"),
]


def load_accuracy(path):
    """Accuracy from one accuracy.json; nan if absent or malformed."""
    try:
        with open(path) as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return np.nan
    acc = data.get("gen", {}).get("q1", np.nan)
    if acc != acc:
        acc = data.get("q1", np.nan)
    return float(acc) if acc == acc else np.nan


def method_dir(model, task, method, mode):
    return (ACCURACY_DIR / model / task / method /
            f"{mode}_eval" / f"{mode}_steer")


def collapsed_curve(model, task, method, reps, mode, rf):
    """Accuracy per topk, collapsing the steering-factor axis by max.

    Returns (curve, best_sf, n_cells_found).
    """
    d = method_dir(model, task, method, mode)
    sfs = sfs_for(model)
    grid = np.full((len(sfs), len(TOPKS)), np.nan)
    for i, sf in enumerate(sfs):
        for j, topk in enumerate(TOPKS):
            fn = f"{sf}_{reps}_steer_topk_{topk}_gen_accuracy_{rf}.json.accuracy.json"
            grid[i, j] = load_accuracy(d / fn)
    found = int(np.sum(~np.isnan(grid)))
    all_nan = np.all(np.isnan(grid), axis=0)
    with np.errstate(all="ignore"):
        curve = np.where(all_nan, np.nan, np.nanmax(grid, axis=0))
        best = [sfs[int(np.nanargmax(grid[:, j]))] if not all_nan[j] else np.nan
                for j in range(len(TOPKS))]
    return curve, np.array(best, dtype=float), found


def min_k_to_80(curve):
    """Smallest topk reaching 80% of this curve's own ceiling."""
    if np.all(np.isnan(curve)):
        return np.nan
    ceiling = np.nanmax(curve)
    if ceiling <= 0:
        return np.nan
    for k, v in zip(TOPKS, curve):
        if v == v and v >= 0.8 * ceiling:
            return k
    return np.nan


def collect(models, seeds, families, modes):
    """One row per (model, task, mode, arm, seed) holding the collapsed curve."""
    rows = []
    for model in models:
        for fam, (_, tasks) in families.items():
            for task in tasks:
                localization = "long-form" if "-long_to_" in task else "single-token"
                for mode_suffix in modes:
                    mode = f"{fam}-{mode_suffix}"
                    rf = "w_rf" if mode_suffix == "long" else "wo_rf"
                    for arm, prefix, reps in ARMS:
                        arm_seeds = [None] if arm == "real" else seeds
                        for seed in arm_seeds:
                            method = prefix if seed is None else f"{prefix}-s{seed}"
                            curve, best_sf, found = collapsed_curve(
                                model, task, method, reps, mode, rf)
                            if found == 0:
                                continue
                            rows.append({
                                "model": model, "family": fam, "task": task,
                                "localization": localization,
                                "eval_mode": mode_suffix, "metric": rf,
                                "arm": arm, "seed": seed, "method": method,
                                "cells_found": found,
                                "cells_expected": len(sfs_for(model)) * len(TOPKS),
                                "curve": curve, "best_sf": best_sf,
                                "min_k_80": min_k_to_80(curve),
                                "ceiling": np.nanmax(curve) if found else np.nan,
                            })
    return rows


def long_table(rows):
    """Tidy per-(row, topk) frame for CSV export."""
    recs = []
    for r in rows:
        for k, acc, sf in zip(TOPKS, r["curve"], r["best_sf"]):
            recs.append({
                "model": r["model"], "family": r["family"], "task": r["task"],
                "localization": r["localization"], "eval_mode": r["eval_mode"],
                "metric": r["metric"], "arm": r["arm"], "seed": r["seed"],
                "method": r["method"], "topk": k, "accuracy": acc,
                "best_steering_factor": sf,
            })
    return pd.DataFrame(recs)


def summarize_random(rows, model, task, mode, arm):
    """mean / min / max / sd across seeds for one random arm, per topk."""
    sel = [r for r in rows if r["model"] == model and r["task"] == task
           and r["eval_mode"] == mode and r["arm"] == arm]
    if not sel:
        return None
    stack = np.vstack([r["curve"] for r in sel])
    with np.errstate(all="ignore"):
        return {
            "n_seeds": len(sel),
            "seeds": sorted(r["seed"] for r in sel),
            "mean": np.nanmean(stack, axis=0),
            "min": np.nanmin(stack, axis=0),
            "max": np.nanmax(stack, axis=0),
            "sd": np.nanstd(stack, axis=0, ddof=1) if len(sel) > 1
                  else np.full(len(TOPKS), np.nan),
            "min_k_80": [r["min_k_80"] for r in sel],
        }


COLORS = {"long-form": "#1b4965", "single-token": "#bc4749",
          "uniform": "#6c757d", "layer-matched": "#e09f3e"}


def _style_axis(ax, mode, title):
    ax.set_xscale("log")
    ax.set_xticks(TOPKS)
    ax.set_xticklabels([str(k) for k in TOPKS], fontsize=8)
    ax.axvspan(min(TOPKS), 0.1, color="grey", alpha=0.06)
    ax.set_xlabel("top-k (fraction of heads steered)")
    metric = "judge + fluency + relevance" if mode == "long" else "token match"
    ax.set_ylabel(f"accuracy ({metric})")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)


def plot_task(rows, model, task, mode, families, plt):
    """All four arms for one task: this localization's real curve, the sibling
    localization's real curve for reference, and both random bands."""
    loc = "long-form" if "-long_to_" in task else "single-token"
    fig, ax = plt.subplots(figsize=(7, 4.5))
    plotted = False

    real = [r for r in rows if r["model"] == model and r["task"] == task
            and r["eval_mode"] == mode and r["arm"] == "real"]
    if real:
        ax.plot(TOPKS, real[0]["curve"], "o-", color=COLORS[loc], lw=2.2,
                label=f"real {loc} (this task)")
        plotted = True

    sib = sibling_task(task, families)
    if sib:
        sib_loc = "long-form" if "-long_to_" in sib else "single-token"
        sib_real = [r for r in rows if r["model"] == model and r["task"] == sib
                    and r["eval_mode"] == mode and r["arm"] == "real"]
        if sib_real:
            ax.plot(TOPKS, sib_real[0]["curve"], "o:", color=COLORS[sib_loc],
                    lw=1.2, ms=3, alpha=0.75, label=f"real {sib_loc} (reference)")
            plotted = True

    for arm in ("uniform", "layer-matched"):
        s = summarize_random(rows, model, task, mode, arm)
        if s is None:
            continue
        ax.plot(TOPKS, s["mean"], "-", color=COLORS[arm], lw=1.5, marker="s", ms=4,
                label=f"{arm} random (mean of {s['n_seeds']} seeds)")
        ax.fill_between(TOPKS, s["min"], s["max"], color=COLORS[arm], alpha=0.18, lw=0)
        plotted = True

    if not plotted:
        plt.close(fig)
        return None

    _style_axis(ax, mode, f"{model} — {task} — {mode}-form eval\n"
                          f"steering factor collapsed by max; bands span seeds")
    fig.tight_layout()
    out = task_out_dir(model, task)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"kcurve_{model}_{task}_{mode}eval.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_family(rows, model, fam, tasks, mode, plt):
    """Model-level rollup: both localizations and their random arms together."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    plotted = False
    for task in tasks:
        loc = "long-form" if "-long_to_" in task else "single-token"
        real = [r for r in rows if r["model"] == model and r["task"] == task
                and r["eval_mode"] == mode and r["arm"] == "real"]
        if real:
            ax.plot(TOPKS, real[0]["curve"], "o-", color=COLORS[loc], lw=2,
                    label=f"real {loc}")
            plotted = True
        for arm in ("uniform", "layer-matched"):
            s = summarize_random(rows, model, task, mode, arm)
            if s is None:
                continue
            ls = "-" if loc == "long-form" else "--"
            ax.plot(TOPKS, s["mean"], ls, color=COLORS[arm], lw=1.5, marker="s", ms=4,
                    label=f"{arm} random ({loc} profile, n={s['n_seeds']})")
            ax.fill_between(TOPKS, s["min"], s["max"], color=COLORS[arm],
                            alpha=0.18, lw=0)
            plotted = True
    if not plotted:
        plt.close(fig)
        return None
    _style_axis(ax, mode, f"{model} — {fam} — {mode}-form eval\n"
                          f"steering factor collapsed by max; bands span seeds")
    fig.tight_layout()
    out = out_dir_for(model)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"kcurve_{model}_{fam}_{mode}eval.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def make_plots(rows, models, families, modes):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    made = []
    for model in models:
        for fam, (_, tasks) in families.items():
            for mode in modes:
                for task in tasks:
                    p = plot_task(rows, model, task, mode, families, plt)
                    if p:
                        made.append(p)
                p = plot_family(rows, model, fam, tasks, mode, plt)
                if p:
                    made.append(p)
    return made


def report(rows, models, families, modes, only_tasks=None):
    lines = []
    flags = []
    uncheckable = []

    for model in models:
        for fam, (_, tasks) in families.items():
            for mode in modes:
                header = f"\n=== {model} | {fam} | {mode}-form eval " \
                         f"({'w_rf' if mode == 'long' else 'wo_rf'}) ==="
                block = []
                for task in tasks:
                    if only_tasks is not None and task not in only_tasks:
                        continue
                    loc = "long-form" if "-long_to_" in task else "single-token"
                    real = [r for r in rows if r["model"] == model and r["task"] == task
                            and r["eval_mode"] == mode and r["arm"] == "real"]
                    if not real:
                        continue
                    rc = real[0]["curve"]
                    block.append(f"\n  {loc} localization ({task})")
                    block.append(f"    {'k':<7}{'real':>9}{'uniform (mean [min,max])':>30}"
                                 f"{'layer-matched (mean [min,max])':>34}")
                    summaries = {a: summarize_random(rows, model, task, mode, a)
                                 for a in ("uniform", "layer-matched")}
                    for j, k in enumerate(TOPKS):
                        row = f"    {k:<7}{rc[j]:>9.3f}"
                        for a in ("uniform", "layer-matched"):
                            s = summaries[a]
                            if s is None:
                                row += f"{'--':>30}" if a == "uniform" else f"{'--':>34}"
                                continue
                            txt = f"{s['mean'][j]:.3f} [{s['min'][j]:.3f},{s['max'][j]:.3f}]"
                            row += f"{txt:>30}" if a == "uniform" else f"{txt:>34}"
                        block.append(row)

                    block.append(f"    min-k to 80% of own ceiling: "
                                 f"real={real[0]['min_k_80']} (ceiling {real[0]['ceiling']:.3f})")
                    for a in ("uniform", "layer-matched"):
                        s = summaries[a]
                        if s is None:
                            continue
                        mk = [m for m in s["min_k_80"] if m == m]
                        block.append(f"    min-k to 80% of own ceiling: {a}="
                                     f"{sorted(mk)} across seeds {s['seeds']}")

                    # section 6.3: flag random arms within noise of real at k <= 0.1
                    for a in ("uniform", "layer-matched"):
                        s = summaries[a]
                        if s is None:
                            uncheckable.append(f"{model} | {fam} | {mode}-eval | "
                                               f"{loc} localization | {a}: no seeds scored")
                            continue
                        if s["n_seeds"] < 2:
                            uncheckable.append(f"{model} | {fam} | {mode}-eval | "
                                               f"{loc} localization | {a}: only "
                                               f"{s['n_seeds']} seed scored, need >=2 for spread")
                            continue
                        for j, k in enumerate(TOPKS):
                            if k > 0.1:
                                continue
                            r_, m_, sd_ = rc[j], s["mean"][j], s["sd"][j]
                            if any(x != x for x in (r_, m_, sd_)) or sd_ == 0:
                                continue
                            within_range = s["min"][j] <= r_ <= s["max"][j]
                            z = abs(r_ - m_) / sd_
                            if within_range or z < 2:
                                flags.append(
                                    f"{model} | {fam} | {mode}-eval | {loc} localization | "
                                    f"k={k}: real={r_:.3f} vs {a} random "
                                    f"{m_:.3f}+-{sd_:.3f} (|z|={z:.1f}"
                                    f"{', real inside seed range' if within_range else ''})")
                if block:
                    lines.append(header)
                    lines.extend(block)

    lines.append("\n\n=== HEADLINE CHECK (section 6.3) ===")
    if flags:
        lines.append(f"{len(flags)} condition(s) where a random arm is within noise of the "
                     f"real localization at k <= 0.1:")
        lines.extend("  ! " + f for f in flags)
    elif not uncheckable:
        lines.append("No random arm is within noise of the real localizations at k <= 0.1.")
    else:
        lines.append("Nothing flagged among the conditions that COULD be checked.")
    if uncheckable:
        # The check needs an across-seed spread, so a single scored seed cannot
        # produce a verdict. Saying so beats printing a clean bill of health.
        lines.append(f"\nNOT YET CHECKABLE ({len(uncheckable)} arm-conditions lack >=2 "
                     f"scored seeds; no verdict either way):")
        lines.extend("  - " + u for u in uncheckable)
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="*",
                   default=["Falcon3-10B-Instruct", "Qwen1.5-14B-Chat", "gemma-3-12b-it",
                            "OLMo-2-1124-13B-DPO", "Qwen1.5-32B-Chat"])
    p.add_argument("--seeds", nargs="*", type=int, default=[0])
    p.add_argument("--families", nargs="*", default=list(FAMILIES))
    p.add_argument("--modes", nargs="*", default=["long", "single"])
    p.add_argument("--no_plots", action="store_true")
    args = p.parse_args()

    families = {k: v for k, v in FAMILIES.items() if k in args.families}
    rows = collect(args.models, args.seeds, families, args.modes)
    if not rows:
        print("No accuracy files found yet -- run the judge pipeline first.")
        return 1

    df = long_table(rows)
    csv_paths = []
    for model in args.models:
        sub = df[df["model"] == model]
        if sub.empty:
            continue
        # per task, alongside atp/ inside the task folder
        for task in sorted(sub["task"].unique()):
            tsub = sub[sub["task"] == task]
            out = task_out_dir(model, task)
            out.mkdir(parents=True, exist_ok=True)
            cp = out / "random_control_curves.csv"
            tsub.to_csv(cp, index=False)
            csv_paths.append(cp)
        # model-level rollup across that model's tasks
        out = out_dir_for(model)
        out.mkdir(parents=True, exist_ok=True)
        cp = out / "random_control_curves.csv"
        sub.to_csv(cp, index=False)
        csv_paths.append(cp)

    # coverage: partial conditions would silently distort a max-collapse
    partial = [(r["model"], r["task"], r["eval_mode"], r["method"],
                r["cells_found"], r["cells_expected"])
               for r in rows if r["cells_found"] < r["cells_expected"]]
    if partial:
        print(f"WARNING: {len(partial)} arm(s) have incomplete (N x topk) grids; "
              f"their curves collapse over fewer cells:")
        for t in partial[:20]:
            print("  ", t)
        if len(partial) > 20:
            print(f"   ... and {len(partial) - 20} more")

    text = report(rows, args.models, families, args.modes)
    print(text)
    for model in args.models:
        model_rows = [r for r in rows if r["model"] == model]
        if not model_rows:
            continue
        for task in sorted({r["task"] for r in model_rows}):
            per_task = report(rows, [model], families, args.modes, only_tasks={task})
            out = task_out_dir(model, task)
            out.mkdir(parents=True, exist_ok=True)
            (out / "random_control_report.txt").write_text(per_task)
        per_model = report(rows, [model], families, args.modes)
        out = out_dir_for(model)
        out.mkdir(parents=True, exist_ok=True)
        (out / "random_control_report.txt").write_text(per_model)

    if not args.no_plots:
        made = make_plots(rows, args.models, families, args.modes)
        print(f"\nwrote {len(made)} plot(s)")
    for cp in csv_paths:
        print(f"wrote {cp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
