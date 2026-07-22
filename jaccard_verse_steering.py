"""
Jaccard similarity between an unsteered "respond in verse" generation and the
steered generation produced when the model is asked to "respond in prose" but
patched with a verse-steering vector at a given (steering factor, top-k heads)
setting.

For each of the two ATP localization types (long-form vs single-token, i.e.
`from_verse-long_to_prose` vs `from_verse-single_to_prose`) this loads the
verse-long free-generation eval (`verse-long_eval`, using the steering vector
that matches the localization type: verse-long_steer for the long-form
localization dir, verse-single_steer for the single-token localization dir),
computes word-level Jaccard similarity per test item against the unsteered
verse baseline, averages over the 50 test items, and plots a
steering-factor x top-k heatmap.

Usage: python jaccard_verse_steering.py <model_name> <baseline_json_path>
  e.g. python jaccard_verse_steering.py Qwen1.5-14B-Chat /path/to/verse_baseline.json
"""
import json
import os
import re
import sys

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm

matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

REPO = "/home/ubuntu/gcm-interp"

STEERING_FACTORS = [10, 8, 6, 5, 4, 2, 1]
TOPK_VALUES = [0.01, 0.03, 0.05, 0.07, 0.09, 0.1, 0.5, 1.0]

LOCALIZATIONS = [
    {
        "key": "from_verse-long_to_prose",
        "label": "Long-Form Localization",
        "steer_variant": "verse-long_steer",
    },
    {
        "key": "from_verse-single_to_prose",
        "label": "Single-Token Localization",
        "steer_variant": "verse-single_steer",
    },
]


def tokenize(text: str) -> set:
    """Lowercased word-level token set for Jaccard similarity."""
    return set(re.findall(r"[a-z0-9']+", text.lower()))


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def load_baseline(baseline_path):
    with open(baseline_path) as f:
        data = json.load(f)
    # id -> tokenized unsteered verse generation
    return {d["id"]: tokenize(d["unsteered_verse"]) for d in data}


def load_test_ids(model_name):
    path = os.path.join(REPO, "data", model_name, "verse-long", "prose-test.jsonl")
    with open(path) as f:
        return [json.loads(l)["id"] for l in f]


def build_heatmap(model_name, loc_key, steer_variant, baseline, test_ids):
    eval_dir = os.path.join(
        REPO, "results", model_name, loc_key, "atp",
        "verse-long_eval", steer_variant, "eval",
    )
    heatmap = np.full((len(STEERING_FACTORS), len(TOPK_VALUES)), np.nan)
    csv_rows = []

    for i, sf in enumerate(STEERING_FACTORS):
        for j, topk in enumerate(TOPK_VALUES):
            fname = f"{sf}_targeted_steer_{topk}_verse-long_gen.json"
            fpath = os.path.join(eval_dir, fname)
            if not os.path.exists(fpath):
                print(f"  Missing: {fpath}")
                continue
            with open(fpath) as f:
                gens = json.load(f)
            scores = []
            for item_idx, item in enumerate(gens):
                gid = test_ids[item_idx] if item_idx < len(test_ids) else None
                if gid not in baseline:
                    continue
                steered_tokens = tokenize(item["edit_prose"])
                scores.append(jaccard(baseline[gid], steered_tokens))
            avg = float(np.mean(scores)) if scores else np.nan
            heatmap[i, j] = avg
            csv_rows.append({
                "model": model_name,
                "localization": loc_key,
                "steer_variant": steer_variant,
                "steering_factor": sf,
                "topk": topk,
                "n_items": len(scores),
                "avg_jaccard": avg,
            })
    return heatmap, csv_rows


def draw_heatmap(ax, heatmap, cmap, norm, title, is_first):
    im = ax.imshow(heatmap, aspect="auto", origin="lower", cmap=cmap, norm=norm)
    for i in range(len(STEERING_FACTORS)):
        for j in range(len(TOPK_VALUES)):
            val = heatmap[i, j]
            if not np.isnan(val):
                rgba = cmap(norm(val))
                brightness = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                color = "black" if brightness > 0.5 else "white"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8, color=color)
    ax.set_xticks(range(len(TOPK_VALUES)))
    ax.set_xticklabels(TOPK_VALUES, rotation=45, ha="right")
    ax.set_xlabel("Top-K fraction of heads")
    if is_first:
        ax.set_yticks(range(len(STEERING_FACTORS)))
        ax.set_yticklabels(STEERING_FACTORS)
        ax.set_ylabel("Steering Factor")
    else:
        ax.set_yticks([])
    ax.tick_params(axis="both", which="both", length=0)
    ax.set_title(title, fontweight="bold", pad=8)
    return im


def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "Qwen1.5-14B-Chat"
    baseline_path = sys.argv[2] if len(sys.argv) > 2 else (
        "/tmp/claude-1000/-home-ubuntu/9b93cea2-f73c-49f8-b00c-32437fcbbd12/scratchpad/verse_baseline.json"
    )

    out_dir = os.path.join(REPO, "results", model_name, "jaccard_plots")
    os.makedirs(out_dir, exist_ok=True)

    baseline = load_baseline(baseline_path)
    test_ids = load_test_ids(model_name)
    print(f"[{model_name}] Loaded {len(baseline)} baseline verse generations, {len(test_ids)} test ids")

    all_csv_rows = []
    heatmaps = []
    for loc in LOCALIZATIONS:
        print(f"\n--- {loc['label']} ({loc['key']}, steer={loc['steer_variant']}) ---")
        heatmap, csv_rows = build_heatmap(model_name, loc["key"], loc["steer_variant"], baseline, test_ids)
        heatmaps.append(heatmap)
        all_csv_rows.extend(csv_rows)

    vmax = np.nanmax([np.nanmax(h) for h in heatmaps])
    vmin = 0.0
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.get_cmap("Blues")

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
    for ax, heatmap, loc in zip(axes, heatmaps, LOCALIZATIONS):
        im = draw_heatmap(ax, heatmap, cmap, norm, loc["label"], is_first=(ax is axes[0]))

    cbar = fig.colorbar(im, ax=axes, fraction=0.046, pad=0.02, aspect=25)
    cbar.set_label("Avg. Jaccard similarity (steered vs. unsteered-verse)", fontsize=9)

    plt.suptitle(
        f"{model_name}: Verse-Steering Lexical Overlap\n"
        "(steered prose→verse generation vs. genuine unsteered verse generation)",
        fontsize=13, fontweight="bold", y=1.08,
    )

    for ext in ("png", "pdf"):
        out_path = os.path.join(out_dir, "verse_steering_jaccard_heatmaps.{}".format(ext))
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"Saved {out_path}")
    plt.close()

    df = pd.DataFrame(all_csv_rows)
    csv_path = os.path.join(out_dir, "verse_steering_jaccard.csv")
    df.to_csv(csv_path, index=False)
    print(f"CSV saved to: {csv_path} ({len(df)} rows)")


if __name__ == "__main__":
    main()
