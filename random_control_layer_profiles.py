"""Layer distribution of ATP-selected heads: long-form vs single-token localization.

This is the structure the layer-matched control arm holds fixed, so it explains
what that arm can and cannot rule out. Each localization's profile is read from
its OWN matched eval/steer subdirectory -- the committed ATP selections are not
consistent across subdirectories, so pooling them would blur the comparison.

Writes one figure per (model, family) into that model's rollup folder, plus a
combined figure.

Usage:  python random_control_layer_profiles.py [--topk 0.1]
"""

import argparse
import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent
RESULTS = REPO / "results"

MODELS = ["Falcon3-10B-Instruct", "Qwen1.5-14B-Chat", "gemma-3-12b-it",
          "OLMo-2-1124-13B-DPO", "Qwen1.5-32B-Chat"]
FAMILIES = {"verse": ("prose", "verse"), "paragraph": ("sentence", "paragraph")}
COL = {"long": "#1b4965", "single": "#bc4749"}


def selection(model, task, mode, topk):
    p = RESULTS / model / task / "atp" / f"{mode}_eval" / f"{mode}_steer" / "eval" / \
        f"numerator_1_targeted_{topk}.csv"
    return pd.read_csv(p)[["layer", "neuron"]] if p.exists() else None


def num_layers(model, task):
    fs = glob.glob(str(RESULTS / model / task / "atp" / "*/*/eval/numerator_1_targeted_1.0.csv"))
    return int(pd.read_csv(fs[0])["layer"].max()) + 1 if fs else None


def collect(topk):
    rows = []
    for model in MODELS:
        for fam, (base, pre) in FAMILIES.items():
            for loc in ("long", "single"):
                task = f"from_{pre}-{loc}_to_{base}"
                sel = selection(model, task, f"{pre}-{loc}", topk)
                if sel is None:
                    continue
                L = num_layers(model, task)
                rows.append({"model": model, "family": fam, "loc": loc, "task": task,
                             "L": L, "layers": sel["layer"].values, "n": len(sel)})
    return rows


def draw(ax, rows, model, fam, topk):
    got = False
    for loc in ("long", "single"):
        r = next((r for r in rows if r["model"] == model and r["family"] == fam
                  and r["loc"] == loc), None)
        if r is None:
            continue
        counts = np.bincount(r["layers"], minlength=r["L"])
        depth = np.arange(r["L"]) / (r["L"] - 1)
        ax.fill_between(depth, counts, step="mid", alpha=0.35, color=COL[loc])
        ax.step(depth, counts, where="mid", color=COL[loc], lw=1.4,
                label=f"{loc}-form" if loc == "long" else "single-token")
        ax.axvline(np.mean(r["layers"]) / (r["L"] - 1), color=COL[loc], ls=":", lw=1.2)
        got = True
    ax.set_xlim(0, 1)
    ax.set_title(f"{model} — {fam}", fontsize=9)
    ax.grid(alpha=0.25)
    return got


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topk", default="0.1")
    args = ap.parse_args()
    rows = collect(args.topk)
    if not rows:
        print("no ATP selections found")
        return 1

    # per-model figures, into that model's rollup folder
    for model in MODELS:
        for fam in FAMILIES:
            fig, ax = plt.subplots(figsize=(6, 3))
            if not draw(ax, rows, model, fam, args.topk):
                plt.close(fig)
                continue
            ax.set_xlabel("normalised depth (0 = first layer, 1 = last)")
            ax.set_ylabel("heads selected")
            ax.legend(fontsize=8)
            fig.tight_layout()
            out = RESULTS / model / "random_control_report"
            out.mkdir(parents=True, exist_ok=True)
            fig.savefig(out / f"layer_profile_{fam}_k{args.topk}.png", dpi=150)
            plt.close(fig)

    # combined grid
    fig, axes = plt.subplots(len(MODELS), 2, figsize=(11, 2.3 * len(MODELS)), squeeze=False)
    for i, model in enumerate(MODELS):
        for j, fam in enumerate(FAMILIES):
            drew = draw(axes[i][j], rows, model, fam, args.topk)
            if not drew:
                axes[i][j].text(.5, .5, "no data", ha="center", va="center", fontsize=8)
            if i == len(MODELS) - 1:
                axes[i][j].set_xlabel("normalised depth")
            if j == 0:
                axes[i][j].set_ylabel("heads")
            if i == 0 and j == 0:
                axes[i][j].legend(fontsize=8)
    fig.suptitle(f"ATP-selected head depth, long-form vs single-token localization "
                 f"(k={args.topk}); dotted = mean depth", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    combined = RESULTS / f"_cross_model_layer_profiles_k{args.topk}.png"
    fig.savefig(combined, dpi=150)
    plt.close(fig)

    # NB: name the column "localization", not "loc" -- df.loc is pandas' indexer.
    df = pd.DataFrame([{"model": r["model"], "family": r["family"],
                        "localization": r["loc"], "n_heads": r["n"],
                        "n_layers": len(set(r["layers"])),
                        "mean_depth": np.mean(r["layers"]) / (r["L"] - 1)} for r in rows])
    piv = df.pivot_table(index=["family", "model"], columns="localization",
                         values=["mean_depth", "n_layers"]).round(3)
    print(piv.to_string())
    deeper = 0
    for _, g in df.groupby(["family", "model"]):
        gs = g.set_index("localization")["mean_depth"]
        if "single" in gs and "long" in gs and gs["single"] > gs["long"]:
            deeper += 1
    print(f"\nsingle-token localization is DEEPER than long-form in "
          f"{deeper}/{df.groupby(['family','model']).ngroups} (model, family) pairs")
    print(f"wrote {combined}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
