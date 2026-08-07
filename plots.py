import argparse
import csv
import os
import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd

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

RM_INTERP_REPO = os.path.dirname(os.path.abspath(__file__))

# ===== Defaults =====
DEFAULT_STEERING_FACTORS = [20, 15, 10, 8, 6, 5, 4, 2, 1]
DEFAULT_TOPK_VALUES = [0.01, 0.03, 0.05, 0.07, 0.09, 0.1, 0.5, 1.0]

ALL_TASKS = [
    "from_sycophancy-long_to_non-sycophantic",
    "from_sycophancy-single_to_non-sycophantic",
    "from_verse-long_to_prose",
    "from_verse-single_to_prose",
    "from_paragraph-long_to_sentence",
    "from_paragraph-single_to_sentence",
    "from_paragraphMCQA-long_to_sentenceMCQA-long",
    "from_paragraphMCQA-single_to_sentenceMCQA-single",
]
TASK_DICT = {
    "from_sycophancy-long_to_non-sycophantic":  "Sycophancy\n(Long)",
    "from_sycophancy-single_to_non-sycophantic": "Sycophancy\n(Single)",
    "from_verse-long_to_prose":                  "Localization: Verse\n(Long)",
    "from_verse-single_to_prose":                "Localization: Verse\n(Single)",
    "from_paragraph-long_to_sentence":           "Summarization\n(Long)",
    "from_paragraph-single_to_sentence":         "Summarization\n(Single)",
    "from_paragraphMCQA-long_to_sentenceMCQA-long":     "Summarization MCQA\n(Long)",
    "from_paragraphMCQA-single_to_sentenceMCQA-single": "Summarization MCQA\n(Single)",
}


ALL_MODELS = ["Qwen1.5-14B-Chat", "OLMo-2-1124-13B-DPO", "Qwen1.5-32B-Chat", "gemma-3-12b-it", "phi-4",
              "Falcon3-10B-Instruct"]

MODEL_DISPLAY_NAMES = {
    "Qwen1.5-14B-Chat":            "Qwen 1.5-14B",
    "Qwen1.5-32B-Chat":            "Qwen 1.5-32B",
    "Llama-2-13b-chat-hf":         "Llama 2-13B",
    "OLMo-2-1124-13B-DPO":         "OLMo 2-13B",
    "vicuna-13b-v1.5":             "Vicuna 13B",
    "SOLAR-10.7B-Instruct-v1.0":   "SOLAR-10.7B",
    "gemma-3-12b-it":              "Gemma 3-12B",
    "phi-4":                       "Phi-4",
    "Falcon3-10B-Instruct":        "Falcon3-10B",
}

MODEL_COLORMAPS = {
    "Qwen1.5-14B-Chat":            "Blues",
    "Qwen1.5-32B-Chat":            "Purples",
    "Llama-2-13b-chat-hf":         "Oranges",
    "OLMo-2-1124-13B-DPO":         "Greens",
    "vicuna-13b-v1.5":             "Greys",
    "SOLAR-10.7B-Instruct-v1.0":   "Reds",
    "gemma-3-12b-it":              "YlOrBr",
    "phi-4":                       "PuRd",
}

METHOD_DICT = {
    "acp": "Full Vector\nPatching [[GCM]]",
    "atp": "Attribution\nPatching [[GCM]]",
    "atp-zero": "Attention Head\nKnockouts [[GCM]]",
    "probes": "Inference-Time\nInterventions (ITI)",
    "random": "Randomly Selected\nHeads",
}

ABLATION_DICT = {
    "steer": "Difference in Means Steering",
    "pyreft": "Representation Fine-Tuning based steering",
    "mean": "Means Steering",
}


# ---------------------------------------------------------------------------
# Core heatmap builder
# ---------------------------------------------------------------------------

def load_accuracy(filepath: str) -> float:
    """Load accuracy value from a JSON file."""
    with open(filepath, "r") as f:
        data = json.load(f)
    acc = data.get("gen", {}).get("q1", np.nan)
    if acc is np.nan or acc != acc:
        acc = data.get("q1", np.nan)
    return acc


def build_heatmap(
    root_dir: str,
    model_id: str,
    task: str,
    method: str,
    ablation: str,
    eval_variant: str,
    steer_variant: str,
    rf_suffix: str,  # "w_rf" or "wo_rf"
    steering_factors: list,
    topk_values: list,
) -> tuple[np.ndarray, list[dict]]:
    """Load accuracy values and return (heatmap_data, csv_rows)."""
    source = task.split("_to_")[0].split("from_")[1]
    breakup_source = source.split("-")[0]

    heatmap_data = np.zeros((len(steering_factors), len(topk_values)))
    csv_rows = []

    for i, sf in enumerate(steering_factors):
        for j, topk in enumerate(topk_values):
            acp_dir = os.path.join(root_dir, task, "acp")
            load_method = "acp" if (topk == 1 and os.path.isdir(acp_dir)) else method
            method_dir = os.path.join(
                root_dir, task, load_method,
                f"{breakup_source}-{eval_variant}_eval/",
                f"{breakup_source}-{steer_variant}_steer/",
            )
            if load_method != "random":
                filename = f"{sf}_targeted_{ablation}_topk_{topk}_gen_accuracy_{rf_suffix}.json.accuracy.json"
            else:
                filename = f"{sf}_random_{ablation}_topk_{topk}_gen_accuracy_{rf_suffix}.json.accuracy.json"

            filepath = os.path.join(method_dir, filename)
            try:
                accuracy = load_accuracy(filepath)
                heatmap_data[i, j] = accuracy
                csv_rows.append({
                    "model_id": model_id,
                    "method": method,
                    "ablation": ablation,
                    "task": task,
                    "eval_variant": eval_variant,
                    "steer_variant": steer_variant,
                    "rf_mode": rf_suffix,
                    "steering_factor": sf,
                    "topk": topk,
                    "accuracy": accuracy,
                })
            except FileNotFoundError:
                print(f"  Missing: {filepath}")
                heatmap_data[i, j] = np.nan

    return heatmap_data, csv_rows


def draw_heatmap(ax, heatmap_data, topk_values, steering_factors, cmap, norm,
                 row_idx, col_idx, n_rows, n_cols, task_label, model_id, fig,
                 show_yticks=None):
    """Draw one heatmap cell with annotations and axis labels."""
    im = ax.imshow(heatmap_data, aspect="auto", origin="lower", cmap=cmap, norm=norm)

    for i in range(len(steering_factors)):
        for j in range(len(topk_values)):
            val = heatmap_data[i, j]
            if not np.isnan(val):
                rgba = cmap(norm(val))
                brightness = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                color = "black" if brightness > 0.5 else "white"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=6, color=color)

    ax.tick_params(axis="both", which="both", length=0)

    if row_idx == n_rows - 1:
        ax.set_xticks(range(len(topk_values)))
        ax.set_xticklabels(topk_values, rotation=45, ha="right")
    else:
        ax.set_xticks([])

    # Every model gets its own steering-factor axis: the columns can be swept
    # over different factors, so a shared axis would mislabel them.
    if show_yticks is None:
        show_yticks = True
    if show_yticks:
        ax.set_yticks(range(len(steering_factors)))
        ax.set_yticklabels(steering_factors)
        ax.set_ylabel("Steering Factor")
    else:
        ax.set_yticks([])

    if row_idx == 0:
        display_name = MODEL_DISPLAY_NAMES.get(model_id, model_id)
        ax.set_title(display_name, fontweight="bold", pad=6)

    return im


def make_grid_plot(
    models: list[str],
    task: str,
    method: str,
    ablation: str,
    eval_variant: str,
    steer_variant: str,
    rf_suffix: str,
    steering_factors: list,
    topk_values: list,
    accuracy_dir: str,
    save_dir: str,
    sf_by_model: dict | None = None,
) -> list[dict]:
    """Build and save one heatmap grid for a single task (one file per task).

    `sf_by_model` overrides the steering factors for individual models, so a
    model evaluated over a different sweep than the rest keeps its own y-axis
    instead of showing empty rows for factors it was never run at.
    """
    sf_by_model = sf_by_model or {}
    n_rows = 1
    n_cols = len(models)
    fig, axes = plt.subplots(
        nrows=n_rows, ncols=n_cols,
        figsize=(4.5 * n_cols + 1, 4.5), constrained_layout=True, squeeze=False,
    )

    all_csv_rows = []
    norm = plt.Normalize(vmin=0, vmax=1)

    for col_idx, model_id in enumerate(models):
        root_dir = os.path.join(accuracy_dir, model_id)
        model_sfs = sf_by_model.get(model_id, steering_factors)
        print(f"  {model_id} | {task} | eval={eval_variant} steer={steer_variant} rf={rf_suffix}")

        cmap = cm.get_cmap(MODEL_COLORMAPS.get(model_id, "Reds"))
        heatmap_data, csv_rows = build_heatmap(
            root_dir, model_id, task, method, ablation,
            eval_variant, steer_variant, rf_suffix,
            model_sfs, topk_values,
        )
        all_csv_rows.extend(csv_rows)

        ax = axes[0, col_idx]
        task_label = TASK_DICT.get(task, task)
        im = draw_heatmap(ax, heatmap_data, topk_values, model_sfs,
                          cmap, norm, 0, col_idx, n_rows, n_cols,
                          task_label, model_id, fig)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, aspect=18)
        cbar.set_label("Steering success rate", fontsize=9)
        cbar.ax.tick_params(labelsize=8)

    is_single = eval_variant == "single"
    eval_str  = "Single-Token" if eval_variant == "single" else "Long-Form"
    steer_str = "Single-Token" if steer_variant == "single" else "Long-Form"
    rf_label  = "Token Matching" if is_single else (
        "w/ R+F Filter" if rf_suffix == "w_rf" else "w/o R+F Filter"
    )

    task_label_title = TASK_DICT.get(task, task).replace("\n", " ")
    plt.suptitle(
        f"{task_label_title}  ·  {eval_str} Eval, {steer_str} Steer  ·  {rf_label}",
        fontsize=15, y=1.05, fontweight="bold",
    )
    plt.figtext(0.5, -0.04, "Top-K fraction of concept-sensitive attention heads",
                ha="center", fontsize=10)

    # Derive a short task slug for the filename (e.g. "sycophancy-long", "verse-single")
    task_slug = task.split("from_")[1].split("_to_")[0]
    rf_file_suffix = "token_matching" if is_single else rf_suffix
    fname = f"{ablation}_{task_slug}_{eval_variant}-eval_{steer_variant}-steer_heatmap_{rf_file_suffix}"
    for ext in ("png", "pdf"):
        out_path = os.path.join(save_dir, f"{fname}.{ext}")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"  Saved {out_path}")
    plt.close()

    return all_csv_rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Generate steering evaluation heatmap plots")
    p.add_argument("--models", nargs="*", default=None,
                   help="Models to include (default: all)")
    p.add_argument("--tasks", nargs="*", default=None,
                   help="Task names as short keys (e.g. sycophancy-long sycophancy-single) "
                        "or full names (from_sycophancy-long_to_non-sycophantic). Default: all")
    p.add_argument("--eval_variants", nargs="*", default=["single", "long"],
                   help="Eval variants: long single (default: both)")
    p.add_argument("--steer_variants", nargs="*", default=["single", "long"],
                   help="Steer variants: long single (default: both)")
    p.add_argument("--rf_mode", choices=["with", "without", "both"], default="both",
                   help="Which accuracy files to plot: with/without relevance+fluency filter, or both (default)")
    p.add_argument("--ablations", nargs="*", default=["steer"],
                   help="Ablation types (default: steer)")
    p.add_argument("--methods", nargs="*", default=["atp"],
                   help="Methods to plot (default: atp)")
    p.add_argument("--accuracy_dir", default=None,
                   help="Path to accuracy directory (default: judge-evals/accuracy/)")
    p.add_argument("--save_dir", default=None,
                   help="Where to save plots (default: judge-evals/accuracy/plots/)")
    p.add_argument("--steering_factors", nargs="*", type=int, default=None,
                   help=f"Steering factors (N) to plot, high to low "
                        f"(default: {DEFAULT_STEERING_FACTORS})")
    p.add_argument("--model_steering_factors", nargs="*", default=None,
                   metavar="MODEL=N,N,...",
                   help="Per-model steering factor override, e.g. "
                        "'Falcon3-10B-Instruct=20,15,10,8,6,5,4,2,1'. Models not "
                        "listed use --steering_factors. Use this when one model was "
                        "swept over different factors than the rest, so it keeps its "
                        "own y-axis instead of showing empty rows.")
    p.add_argument("--wordcount", action="store_true",
                   help="Generate deterministic token-count heatmaps for the "
                        "summarization task (long-form eval only) instead of "
                        "judge-accuracy heatmaps. Reads post-intervention responses "
                        "from workdir eval_output.csv files and plots average token "
                        "count per cell (using each model's own tokenizer).")
    return p.parse_args()


def parse_model_steering_factors(specs: list[str] | None) -> dict[str, list[int]]:
    """Parse 'MODEL=N,N,...' specs into {model: [factors]}."""
    parsed = {}
    for spec in specs or []:
        if "=" not in spec:
            raise SystemExit(f"--model_steering_factors expects MODEL=N,N,... (got {spec!r})")
        model, _, factors = spec.partition("=")
        try:
            parsed[model] = [int(f) for f in factors.split(",") if f.strip()]
        except ValueError:
            raise SystemExit(f"--model_steering_factors: non-integer factor in {spec!r}")
        if not parsed[model]:
            raise SystemExit(f"--model_steering_factors: no factors given in {spec!r}")
    return parsed


def resolve_tasks(task_args: list[str] | None) -> list[str]:
    """Resolve short task names to full task keys."""
    if task_args is None:
        return ALL_TASKS
    resolved = []
    for t in task_args:
        if t in TASK_DICT:
            resolved.append(t)
        else:
            # Try matching by short name (e.g. "sycophancy-long" → full key)
            matches = [k for k in TASK_DICT if f"from_{t}_to_" in k or t in k]
            if matches:
                resolved.extend(matches)
            else:
                print(f"Warning: unknown task '{t}', skipping")
    return resolved


# ---------------------------------------------------------------------------
# Word-count heatmaps (deterministic, no judge)
# ---------------------------------------------------------------------------

WORKDIR_ROOT = os.path.join(RM_INTERP_REPO, "judge-evals", "workdirs")

MODEL_HF_IDS = {
    "Qwen1.5-14B-Chat":       "Qwen/Qwen1.5-14B-Chat",
    "Qwen1.5-32B-Chat":       "Qwen/Qwen1.5-32B-Chat",
    "OLMo-2-1124-13B-DPO":    "allenai/OLMo-2-1124-13B-DPO",
    "gemma-3-12b-it":          "google/gemma-3-12b-it",
    "Falcon3-10B-Instruct":    "tiiuae/Falcon3-10B-Instruct",
}

WORDCOUNT_TASKS = [
    # (task_dir, loc_label, eval_variant, steer_variant)
    ("from_paragraph-long_to_sentence",   "Long",   "long", "long"),
    ("from_paragraph-long_to_sentence",   "Long",   "long", "single"),
    ("from_paragraph-single_to_sentence", "Single", "long", "long"),
    ("from_paragraph-single_to_sentence", "Single", "long", "single"),
]


def _avg_token_count(csv_path: str, tokenizer) -> float:
    """Return average token count of post-intervention-response across all rows."""
    try:
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            counts = [len(tokenizer.encode(row.get("post-intervention-response", "")))
                      for row in reader]
        return np.mean(counts) if counts else np.nan
    except FileNotFoundError:
        return np.nan


def _build_token_count_matrix(
    model: str,
    task_dir: str,
    eval_variant: str,
    steer_variant: str,
    steering_factors: list,
    topk_values: list,
    tokenizer,
) -> np.ndarray:
    """Build SF × topk matrix of average token counts from workdir CSVs."""
    source = task_dir.split("from_")[1].split("_to_")[0]  # paragraph-long or paragraph-single
    eval_dir = f"paragraph-{eval_variant}_eval"
    steer_dir = f"paragraph-{steer_variant}_steer"
    eval_suffix = f"paragraph-{eval_variant}"

    base = os.path.join(WORKDIR_ROOT, model, task_dir, "atp",
                        eval_dir, steer_dir, "eval")

    data = np.full((len(steering_factors), len(topk_values)), np.nan)
    for i, sf in enumerate(steering_factors):
        for j, topk in enumerate(topk_values):
            folder = f"{sf}_targeted_steer_{topk}_{eval_suffix}"
            csv_path = os.path.join(base, folder, "eval_output.csv")
            data[i, j] = _avg_token_count(csv_path, tokenizer)
    return data


def _draw_wordcount_heatmap(ax, data, topk_values, steering_factors,
                            cmap, norm, model_name):
    """Draw a single word-count heatmap cell."""
    im = ax.imshow(data, aspect="auto", origin="lower", cmap=cmap, norm=norm)
    for i in range(len(steering_factors)):
        for j in range(len(topk_values)):
            val = data[i, j]
            if np.isnan(val):
                continue
            rgba = cmap(norm(val))
            brightness = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            color = "black" if brightness > 0.5 else "white"
            ax.text(j, i, f"{val:.0f}", ha="center", va="center",
                    fontsize=6, color=color)
    ax.set_yticks(range(len(steering_factors)))
    ax.set_yticklabels(steering_factors)
    ax.set_ylabel("Steering Factor")
    ax.set_xticks(range(len(topk_values)))
    ax.set_xticklabels(topk_values, rotation=45, ha="right")
    ax.set_title(model_name, fontweight="bold", pad=6)
    ax.tick_params(axis="both", which="both", length=0)
    return im


def _draw_wordcount_collapsed(ax, max_vals, best_sf, topk_values,
                              cmap, norm, model_name):
    """Draw a single collapsed (1-row) word-count heatmap."""
    im = ax.imshow(max_vals.reshape(1, -1), aspect="auto", cmap=cmap, norm=norm)
    for j in range(len(topk_values)):
        val = max_vals[j]
        sf = best_sf[j]
        if np.isnan(val):
            continue
        rgba = cmap(norm(val))
        brightness = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
        text_color = "black" if brightness > 0.5 else "white"
        sf_str = f"sf={int(sf)}" if not np.isnan(sf) else "sf=?"
        ax.text(j, 0, f"{val:.0f}\n({sf_str})",
                ha="center", va="center",
                fontsize=9, color=text_color, fontweight="bold", linespacing=1.4)
    ax.set_xticks(range(len(topk_values)))
    ax.set_xticklabels(topk_values, rotation=45, ha="right")
    ax.set_yticks([])
    ax.set_title(model_name, fontweight="bold", pad=6)
    ax.tick_params(axis="both", which="both", length=0)
    return im


def make_wordcount_plots(
    models: list[str],
    task_dir: str,
    loc_label: str,
    eval_variant: str,
    steer_variant: str,
    steering_factors: list,
    topk_values: list,
    save_dir: str,
    sf_by_model: dict | None = None,
    tokenizers: dict | None = None,
):
    """Build and save full + collapsed word-count heatmaps for one config."""
    sf_by_model = sf_by_model or {}
    steer_label = "Long-Form" if steer_variant == "long" else "Single-Token"
    title = (f"Summarization ({loc_label} Loc)  ·  Long-Form Eval, "
             f"{steer_label} Steer  ·  Avg Token Count")

    all_data = {}
    all_sfs = {}
    global_max = 0
    for model in models:
        model_sfs = sf_by_model.get(model, steering_factors)
        all_sfs[model] = model_sfs
        tokenizer = tokenizers[model]
        data = _build_token_count_matrix(model, task_dir, eval_variant,
                                         steer_variant, model_sfs,
                                         topk_values, tokenizer)
        all_data[model] = data
        if not np.all(np.isnan(data)):
            global_max = max(global_max, np.nanmax(data))

    if global_max == 0:
        print(f"  No data for {task_dir} eval={eval_variant} steer={steer_variant}")
        return

    norm = plt.Normalize(vmin=0, vmax=global_max * 1.05)
    n_cols = len(models)

    # ── Full heatmap ──
    fig, axes = plt.subplots(1, n_cols, figsize=(4.5 * n_cols + 1, 4.5),
                             constrained_layout=True, squeeze=False)
    for col, model in enumerate(models):
        model_sfs = all_sfs[model]
        cmap = plt.get_cmap(MODEL_COLORMAPS.get(model, "Reds"))
        display = MODEL_DISPLAY_NAMES.get(model, model)
        im = _draw_wordcount_heatmap(axes[0, col], all_data[model],
                                     topk_values, model_sfs,
                                     cmap, norm, display)
        cbar = fig.colorbar(im, ax=axes[0, col], fraction=0.046, pad=0.02,
                            aspect=18)
        cbar.set_label("Avg tokens", fontsize=9)
        cbar.ax.tick_params(labelsize=8)

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.03)
    fig.supxlabel("Top-K fraction of concept-sensitive attention heads",
                  fontsize=11, y=-0.02)

    slug = task_dir.split("from_")[1].split("_to_")[0].replace("paragraph-", "")
    fname = f"tokencount_paragraph-{slug}_{eval_variant}-eval_{steer_variant}-steer"
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(save_dir, f"{fname}.{ext}"),
                    dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fname}.png/pdf")

    # ── Collapsed heatmap ──
    fig2, axes2 = plt.subplots(1, n_cols, figsize=(4.5 * n_cols + 1, 3.0),
                               constrained_layout=True, squeeze=False)
    for col, model in enumerate(models):
        model_sfs = all_sfs[model]
        data = all_data[model]
        all_nan = np.all(np.isnan(data), axis=0)
        max_vals = np.where(all_nan, np.nan, np.nanmax(data, axis=0))
        best_idx = np.array([
            int(np.nanargmax(data[:, j])) if not all_nan[j] else 0
            for j in range(data.shape[1])
        ])
        best_sf = np.where(
            all_nan, np.nan,
            np.array([model_sfs[k] for k in best_idx], dtype=float),
        )
        cmap = plt.get_cmap(MODEL_COLORMAPS.get(model, "Reds"))
        display = MODEL_DISPLAY_NAMES.get(model, model)
        im = _draw_wordcount_collapsed(axes2[0, col], max_vals, best_sf,
                                       topk_values, cmap, norm, display)

    cbar2 = fig2.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap="Greys"),
        ax=axes2[0, -1], fraction=0.046, pad=0.02, aspect=10)
    cbar2.set_label("Avg tokens", fontsize=9)
    cbar2.ax.tick_params(labelsize=8)

    fig2.suptitle(title + " (max across SF)", fontsize=14,
                  fontweight="bold", y=1.05)
    fig2.supxlabel("Top-K fraction of concept-sensitive attention heads",
                   fontsize=11, y=-0.02)

    fname2 = f"collapsed_tokencount_paragraph-{slug}_{eval_variant}-eval_{steer_variant}-steer"
    for ext in ("png", "pdf"):
        fig2.savefig(os.path.join(save_dir, f"{fname2}.{ext}"),
                     dpi=300, bbox_inches="tight")
    plt.close(fig2)
    print(f"  Saved {fname2}.png/pdf")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    accuracy_dir = args.accuracy_dir or os.path.join(RM_INTERP_REPO, "judge-evals", "accuracy")
    save_dir = args.save_dir or os.path.join(accuracy_dir, "plots")
    os.makedirs(save_dir, exist_ok=True)

    models = args.models or ALL_MODELS
    steering_factors = args.steering_factors or DEFAULT_STEERING_FACTORS
    sf_by_model = parse_model_steering_factors(args.model_steering_factors)
    topk_values = DEFAULT_TOPK_VALUES

    # ── Word-count heatmaps (--wordcount) ──
    if args.wordcount:
        # Default per-model SF overrides: drop 15/20 for all models except Falcon3
        wc_sf_by_model = dict(sf_by_model)  # start from any CLI overrides
        sfs_no_high = [sf for sf in steering_factors if sf not in (15, 20)]
        for m in models:
            if m not in wc_sf_by_model and m != "Falcon3-10B-Instruct":
                wc_sf_by_model[m] = sfs_no_high

        from transformers import AutoTokenizer
        # Only load tokenizers for models that have HF IDs defined
        wc_models = [m for m in models if m in MODEL_HF_IDS]
        print("\nLoading tokenizers …")
        tokenizers = {}
        for m in wc_models:
            hf_id = MODEL_HF_IDS[m]
            print(f"  {m} → {hf_id}")
            tokenizers[m] = AutoTokenizer.from_pretrained(hf_id)

        print("\n=== Token-count heatmaps (summarization, long-form eval) ===")
        for task_dir, loc_label, eval_variant, steer_variant in WORDCOUNT_TASKS:
            print(f"\n{task_dir}  eval={eval_variant}  steer={steer_variant}")
            make_wordcount_plots(
                models=wc_models,
                task_dir=task_dir,
                loc_label=loc_label,
                eval_variant=eval_variant,
                steer_variant=steer_variant,
                steering_factors=steering_factors,
                topk_values=topk_values,
                save_dir=save_dir,
                sf_by_model=wc_sf_by_model,
                tokenizers=tokenizers,
            )
        return

    # ── Standard accuracy heatmaps ──
    tasks = resolve_tasks(args.tasks)
    methods = args.methods
    evals = args.eval_variants
    steering = args.steer_variants

    print(f"Models:        {models}")
    print(f"Tasks:         {tasks}")
    print(f"Ablations:     {args.ablations}")
    print(f"Methods:       {methods}")
    print(f"Accuracy dir:  {accuracy_dir}")
    print(f"Save dir:      {save_dir}")
    print(f"Steering factors: {steering_factors}")
    for model, factors in sf_by_model.items():
        print(f"  override {model}: {factors}")

    all_csv_rows = []

    for ablation in args.ablations:
        for method in methods:
            for task in tasks:
                for evalu in evals:
                    for steer in steering:
                        is_single = evalu == "single"
                        variant = "single" if is_single else "long"
                        rf_suffixes = ["wo_rf"] if is_single else ["w_rf", "wo_rf"]

                        for rf_suffix in rf_suffixes:
                            print(f"\n--- ablation={ablation} method={method} "
                                  f"task={task} variant={variant} rf={rf_suffix} ---")
                            csv_rows = make_grid_plot(
                                models=models,
                                task=task,
                                method=method,
                                ablation=ablation,
                                eval_variant=variant,
                                steer_variant=steer,
                                rf_suffix=rf_suffix,
                                steering_factors=steering_factors,
                                topk_values=topk_values,
                                accuracy_dir=accuracy_dir,
                                save_dir=save_dir,
                                sf_by_model=sf_by_model,
                            )
                            all_csv_rows.extend(csv_rows)

    # Save aggregated CSV
    df = pd.DataFrame(all_csv_rows)
    csv_path = os.path.join(save_dir, "steering_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nCSV saved to: {csv_path}  ({len(df)} rows)")


if __name__ == "__main__":
    main()
