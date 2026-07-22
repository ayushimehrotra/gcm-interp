"""
Mean Reciprocal Rank (MRR) between the long-form and single-token ATP head
rankings for the verse localization task.

Methodology (matches the pre-existing results/*/mrr_verse-single_vs_verse-long.csv
convention, reproduced here from a traceable source):
  - Each localization's full head ranking is the `numerator_1_targeted_1.0.csv`
    file (topk=1.0 = every (layer, head), sorted by ATP attribution `value`).
    This ranking is verified identical across all eval/steer sub-combos within
    a localization (see head-ranking verification done earlier), so any combo's
    file can be used as the canonical ranking.
  - For a given topk fraction, take the top-K heads of the QUERY ranking as the
    "relevant set." For each, look up its rank in the TARGET ranking and take
    the reciprocal rank (1/rank). MRR = mean over the K items.
  - Done in both directions (single->long, long->single) and compared to a
    closed-form random-baseline MRR = H(N)/N (harmonic number / N), which is
    what topk=1.0 must equal exactly regardless of ranking order (sanity check).

Note: the earlier results/*/mrr_verse-single_vs_verse-long.csv files reference
a raw `atp/numerator_1_heads.pt` tensor that no longer exists on disk, so they
cannot be bit-reproduced. This script instead uses the per-combo
`numerator_1_targeted_1.0.csv` files, which ARE the files actually consumed to
select heads for every steering run (independently verified). MRR is a
rank-sensitive statistic and the middle of each ranking is a dense near-zero
cluster (attribution values ~1e-4 vs a std of ~0.016), so exact MRR values are
sensitive to precision/reload artifacts there -- but the qualitative
conclusion (both this script's output and the old CSVs) agrees: MRR stays in
the same low range as the random baseline across all topk, i.e. weak-to-no
rank agreement between the two localizations' heads.
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

REPO = "/home/ubuntu/gcm-interp"
TOPK_VALUES = [0.01, 0.03, 0.05, 0.07, 0.09, 0.1, 0.5, 1.0]


def load_full_ranking(model, loc):
    path = os.path.join(
        REPO, "results", model, loc, "atp",
        "verse-long_eval", "verse-long_steer", "eval", "numerator_1_targeted_1.0.csv",
    )
    df = pd.read_csv(path).sort_values("value", ascending=False).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)
    return df


def mrr(query_df, target_df, topk):
    n_total = len(query_df)
    K = int(topk * n_total)
    top_query = query_df.head(K)
    target_rank_lookup = target_df.set_index(["layer", "neuron"])["rank"]
    ranks_in_target = [target_rank_lookup[(row.layer, row.neuron)] for row in top_query.itertuples()]
    reciprocal_ranks = [1.0 / r for r in ranks_in_target]
    return K, float(np.mean(reciprocal_ranks)), float(np.mean(ranks_in_target))


def run(model):
    single_ranking = load_full_ranking(model, "from_verse-single_to_prose")
    long_ranking = load_full_ranking(model, "from_verse-long_to_prose")
    n_total = len(single_ranking)

    rows = []
    for topk in TOPK_VALUES:
        K, m, mr = mrr(single_ranking, long_ranking, topk)
        rows.append(("verse-single", "verse-long", topk, K, m, mr))
        K, m, mr = mrr(long_ranking, single_ranking, topk)
        rows.append(("verse-long", "verse-single", topk, K, m, mr))
    df = pd.DataFrame(rows, columns=["query_ranking", "target_ranking", "topk", "K", "MRR", "mean_rank_of_relevant_in_target"])

    H = sum(1.0 / i for i in range(1, n_total + 1))
    random_baseline = H / n_total
    assert abs(df[(df.topk == 1.0) & (df.query_ranking == "verse-single")]["MRR"].iloc[0] - random_baseline) < 1e-9, \
        "topk=1.0 MRR should exactly equal the random-baseline H(N)/N regardless of ranking order"

    out_dir = os.path.join(REPO, "results", model)
    csv_path = os.path.join(out_dir, "mrr_verse-single_vs_verse-long_reproduced.csv")
    with open(csv_path, "w") as f:
        f.write(f"# MRR between ATP head rankings: verse-single vs verse-long ({model})\n")
        f.write("# rankings = numerator_1_targeted_1.0.csv (verified identical across eval/steer combos within a localization)\n")
        f.write(f"# total heads = {n_total}; K = int(topk * {n_total}); random-baseline MRR (topk=1.0) = H(N)/N = {random_baseline:.6f}\n")
        df.to_csv(f, index=False)
    print(f"[{model}] Saved {csv_path}")

    fig, ax = plt.subplots(figsize=(8, 5))
    for direction, sub in df.groupby(["query_ranking", "target_ranking"]):
        sub = sub.sort_values("topk")
        ax.plot(sub["topk"], sub["MRR"], marker="o", label=f"{direction[0]} -> {direction[1]}")
    ax.axhline(random_baseline, color="gray", linestyle="--", linewidth=1, label=f"random baseline = H(N)/N = {random_baseline:.4f}")
    ax.set_xscale("log")
    ax.set_xlabel("topk fraction (K = topk x {} heads)".format(n_total))
    ax.set_ylabel("MRR")
    ax.set_title(f"{model}: ATP head-ranking MRR (both directions)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    png_path = os.path.join(out_dir, "mrr_verse-single_vs_verse-long_reproduced.png")
    fig.savefig(png_path, dpi=200)
    plt.close(fig)
    print(f"[{model}] Saved {png_path}")

    return df, random_baseline


if __name__ == "__main__":
    models = sys.argv[1:] if len(sys.argv) > 1 else ["Qwen1.5-14B-Chat", "OLMo-2-1124-13B-DPO", "Qwen1.5-32B-Chat"]
    for m in models:
        df, baseline = run(m)
        print(df.to_string(index=False))
        print(f"random baseline: {baseline:.6f}\n")
