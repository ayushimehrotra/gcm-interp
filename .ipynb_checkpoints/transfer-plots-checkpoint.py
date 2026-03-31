import os
import json
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

model_labels = {
    "Qwen1.5-14B-Chat": "Qwen-14B",
    "OLMo-2-1124-13B-DPO": "OLMo-13B",
    "SOLAR-10.7B-Instruct-v1.0": "SOLAR-10B"
}
def collect_results(root_dir, models, task_patterns, local=False):
    """
    Collect results for each model and task from JSON files.

    Args:
        root_dir (str): Root directory containing model subdirectories.
        models (list): List of model directory names.
        task_patterns (dict): Mapping {task_name: glob_pattern}.

    Returns:
        dict: results[task][model] = list of values.
    """
    results = {task: {model: [] for model in models} for task in task_patterns}

    for model in models:
        model_dir = os.path.join(root_dir, model)
        for task, pattern in task_patterns.items():
            search_path = os.path.join(model_dir, "**", pattern)
            # print(search_path)
            for fpath in glob.glob(search_path, recursive=True):
                print(f"Loading {fpath}")
                if local and "topk_1_" in fpath:
                    continue
                with open(fpath, "r") as f:
                    data = json.load(f)
                    val = list(data.values())[0]
                    results[task][model].append(val)

    print(results)
    assert all(
        len(results[task][model]) > 0 for task in task_patterns for model in models
    ), "Some tasks/models have no results!"
    return results


def plot_results(results, tasks, models, output_dir="plots"):
    """
    Make grouped violin plots: x-axis = tasks, violins = models.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Colors inspired by uploaded figure
    colors = ["#f7b3bf", "#e4f0a7", "#A3ECF7"] # pink, light green, light blue
    model_colors = {m: colors[i % len(colors)] for i, m in enumerate(models)}

    x = np.arange(len(tasks))  # positions for tasks
    width = 0.25  # spacing for violins

    plt.figure(figsize=(8, 5))
    for i, model in enumerate(models):
        data = [results[task][model] for task in tasks]
        positions = x + i * width - (width * (len(models) - 1) / 2)

        parts = plt.violinplot(
            data,
            positions=positions,
            widths=width * 0.9,
            showmeans=True,
            showmedians=False,
            showextrema=False,
        )

        # Set violin colors
        for pc in parts["bodies"]:
            pc.set_facecolor(model_colors[model])
            pc.set_edgecolor("black")
            pc.set_alpha(0.7)
        if "cmeans" in parts:  # mean line
            parts["cmeans"].set_color("black")

    # X ticks centered
    plt.xticks(x, tasks, rotation=0, ha="center")
    plt.ylabel("Change in Behavior Expression Rate\npre- to post-steering")
    plt.ylim(-0.02, 1.02)
    plt.title("Transfer Performance Across Tasks")
    plt.grid(axis="y", linestyle="--", alpha=0.6)

    # Fix legend using manual patches
    legend_handles = [Patch(facecolor=model_colors[m], edgecolor="black", label=model_labels[m]) for m in models]
    plt.legend(handles=legend_handles, loc="upper left")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "transfer_violinplot.png"))
    plt.close()

def plot_results_dual(results_global, results_local, tasks, models, output_dir="plots"):
    """
    Side-by-side violin plots:
      - Left side: non-globalized (hatched, legend top-left)
      - Right side: globalized (solid, legend top-right)
    for each task.
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    os.makedirs(output_dir, exist_ok=True)

    # Colors
    colors = ["#f7b3bf", "#e4f0a7", "#A3ECF7"] # pink, light green, light blue
    model_colors = {m: colors[i % len(colors)] for i, m in enumerate(models)}

    n_tasks = len(tasks)
    n_models = len(models)

    # x positions: first n_tasks = local, next n_tasks = global
    x_local = np.arange(n_tasks)
    x_global = np.arange(n_tasks, 2 * n_tasks)
    x_all = np.concatenate([x_local, x_global])
    tick_labels = tasks * 2
    tick_positions = x_all

    model_sep = 0.15  # spacing between models in each slot

    plt.figure(figsize=(8, 5))

    for i, model in enumerate(models):
        offset = (i - (n_models - 1) / 2) * model_sep

        # --- Local (hatched) ---
        data_local = [results_local[t][model] for t in tasks]
        positions_local = x_local + offset
        parts = plt.violinplot(
            data_local,
            positions=positions_local,
            widths=model_sep * 0.9,
            showmeans=True,
            showmedians=False,
            showextrema=False,
        )
        for pc in parts["bodies"]:
            pc.set_facecolor(model_colors[model])
            pc.set_edgecolor("black")
            pc.set_alpha(1.0)
            pc.set_hatch(".")
        if "cmeans" in parts:
            parts["cmeans"].set_color("black")

        # --- Global (solid) ---
        data_global = [results_global[t][model] for t in tasks]
        positions_global = x_global + offset
        parts = plt.violinplot(
            data_global,
            positions=positions_global,
            widths=model_sep * 0.9,
            showmeans=True,
            showmedians=False,
            showextrema=False,
        )
        for pc in parts["bodies"]:
            pc.set_facecolor(model_colors[model])
            pc.set_edgecolor("black")
            pc.set_alpha(1.0)
        if "cmeans" in parts:
            parts["cmeans"].set_color("black")

    # Formatting
    plt.xticks(tick_positions, tick_labels)
    plt.ylabel("Steering Transfer Rate")
    plt.ylim(-0.02, 1.02)
    plt.title("Transfer Performance Across Tasks")
    plt.grid(axis="y", linestyle="--", alpha=0.6)

    # Rotate ticks differently for local vs global
    ax = plt.gca()
    for idx, label in enumerate(ax.get_xticklabels()):
        if idx < n_tasks:  # local side
            label.set_rotation(0)
            label.set_ha("center")
        else:  # global side
            label.set_rotation(0)
            label.set_ha("center")

    # Vertical line separating local vs global
    plt.axvline(x=n_tasks - 0.5, color="gray", linestyle="--")

    # --- Legends ---
    local_handles = [
        Patch(facecolor=model_colors[m], edgecolor="black", hatch=".", label=model_labels[m])
        for m in models
    ]
    global_handles = [
        Patch(facecolor=model_colors[m], edgecolor="black", label=model_labels[m])
        for m in models
    ]

    leg1 = plt.legend(handles=local_handles, title="Local Steering", loc="lower left")
    leg2 = plt.legend(handles=global_handles, title="Global Steering", loc="lower right")
    ax.add_artist(leg1)  # keep both legends

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "transfer_violinplot_dual.png"))
    plt.close()


if __name__ == "__main__":
    root_dir = "/mnt/align4_drive/arunas/rm-interp-minimal/normalized-results/"
    models = ["Qwen1.5-14B-Chat", "OLMo-2-1124-13B-DPO", "SOLAR-10.7B-Instruct-v1.0"]

    task_patterns_global = {
        "Sycophancy\nReduction": "1_syc-on-nlp_*_topk_1_gen*accuracy.json",
        "Refusal\nInduction": "1_alpaca_*_topk_1_gen*accuracy.json",
        "Verse\nStyle Transfer": "1_wp_targeted_*_topk_1_gen*accuracy.json",
    }

    task_patterns_local = {
        "Sycophancy\nReduction": "*_syc-on-nlp_*_topk_*_gen*accuracy.json",
        "Refusal\nInduction": "*_alpaca_*_topk_*_gen*accuracy.json",
        "Verse\nStyle Transfer": "*_wp_targeted_*_topk_*_gen*accuracy.json",
    }

    results_global = collect_results(root_dir, models, task_patterns_global)
    results_local = collect_results(root_dir, models, task_patterns_local, local=True)

    plot_results_dual(results_global, results_local, list(task_patterns_global.keys()), models,
                      output_dir=root_dir + "/transfer-plots-2")
    print("Plot saved in transfer-plots-2/transfer_violinplot_dual.png")
