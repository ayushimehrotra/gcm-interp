"""
Step 4: Compute per-condition accuracies from judge outputs.

Loads the three judge output files (behavioral judge, fluency, relevance),
merges them on row-level metadata, and writes per-condition accuracy JSONs
into an accuracy/ directory tree.

Usage:
    python compute_accuracies.py [--jp_path FILE] [--flu_path FILE] [--rel_path FILE]
"""

import argparse
import json
import os
import re

import pandas as pd
from tqdm import tqdm

from config import ROW_KEY_COLS, GROUP_COLS, load_jsonl_or_json


# ---------------------------------------------------------------------------
# Rating extraction
# ---------------------------------------------------------------------------

def extract_first_int(text) -> int | None:
    """Extract the first integer from judge output text."""
    if not isinstance(text, str):
        return None
    match = re.search(r"\b(\d+)\b", text)
    return int(match.group(1)) if match else None


# ---------------------------------------------------------------------------
# Data loading with caching
# ---------------------------------------------------------------------------

def load_or_cache(json_path: str, cache_name: str) -> pd.DataFrame:
    """Load from JSON, or use CSV cache if available.
    Cache file is placed next to the JSON source.
    """
    parent = os.path.dirname(json_path) or "."
    csv_cache = os.path.join(parent, cache_name)

    if os.path.exists(csv_cache):
        print(f"Loading cached {csv_cache}")
        return pd.read_csv(csv_cache)

    if not os.path.exists(json_path):
        print(f"Not found: {json_path}, returning empty DataFrame")
        return pd.DataFrame()

    print(f"Loading {json_path}")
    items = load_jsonl_or_json(json_path)
    df = pd.DataFrame(items)

    before = len(df)
    df = df.drop_duplicates()
    print(f"Loaded {before} rows, {len(df)} after deduplication")

    df.to_csv(csv_cache, index=False)
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Compute accuracies from judge outputs")
    p.add_argument("--jp_path",
                    default="judge_prompts.judge_prompt.judge_outputs.json",
                    help="Path to behavioral judge outputs (JSONL)")
    p.add_argument("--flu_path",
                    default="relevance_fluency_prompts.fluency_prompt.judge_outputs.json",
                    help="Path to fluency judge outputs (JSONL)")
    p.add_argument("--rel_path",
                    default="relevance_fluency_prompts.relevance_prompt.judge_outputs.json",
                    help="Path to relevance judge outputs (JSONL)")
    p.add_argument("--output_dir", default="accuracy",
                    help="Root directory for accuracy JSON outputs")
    return p.parse_args()


def main():
    args = parse_args()

    # Load (or cache) each judge output
    jp_df = load_or_cache(args.jp_path, "jp_ratings.csv")
    rf_flu_df = load_or_cache(args.flu_path, "rf_fluency_ratings.csv")
    rf_rel_df = load_or_cache(args.rel_path, "rf_relevance_ratings.csv")

    # Bail out if nothing was loaded at all
    if jp_df.empty and rf_flu_df.empty and rf_rel_df.empty:
        print("No judge outputs found. Nothing to compute.")
        return

    # Extract / rename ratings
    if not jp_df.empty:
        jp_df["jp_rating"] = jp_df["judge_output"].apply(extract_first_int)

    if not rf_flu_df.empty and "judge_rating" in rf_flu_df.columns:
        rf_flu_df = rf_flu_df.rename(columns={"judge_rating": "fluency_rating"})

    if not rf_rel_df.empty and "judge_rating" in rf_rel_df.columns:
        rf_rel_df = rf_rel_df.rename(columns={"judge_rating": "relevance_rating"})

    # Pick the base dataframe for merging.
    # If behavioral judge was run, use it as the base.
    # Otherwise, start from fluency or relevance.
    print("Merging dataframes on metadata columns...")

    if not jp_df.empty:
        available_keys = [c for c in ROW_KEY_COLS if c in jp_df.columns]
        merged = jp_df[available_keys + ["jp_rating"]].copy()
    elif not rf_flu_df.empty:
        available_keys = [c for c in ROW_KEY_COLS if c in rf_flu_df.columns]
        merged = rf_flu_df[available_keys + ["fluency_rating"]].copy()
        merged["jp_rating"] = float("nan")
    else:
        available_keys = [c for c in ROW_KEY_COLS if c in rf_rel_df.columns]
        merged = rf_rel_df[available_keys + ["relevance_rating"]].copy()
        merged["jp_rating"] = float("nan")

    # Left-join fluency if not already the base
    if "fluency_rating" not in merged.columns:
        if not rf_flu_df.empty and "fluency_rating" in rf_flu_df.columns:
            flu_keys = [c for c in ROW_KEY_COLS if c in rf_flu_df.columns]
            merged = merged.merge(
                rf_flu_df[flu_keys + ["fluency_rating"]],
                on=flu_keys, how="left",
            )
        else:
            merged["fluency_rating"] = float("nan")

    # Left-join relevance if not already the base
    if "relevance_rating" not in merged.columns:
        if not rf_rel_df.empty and "relevance_rating" in rf_rel_df.columns:
            rel_keys = [c for c in ROW_KEY_COLS if c in rf_rel_df.columns]
            merged = merged.merge(
                rf_rel_df[rel_keys + ["relevance_rating"]],
                on=rel_keys, how="left",
            )
        else:
            merged["relevance_rating"] = float("nan")

    os.makedirs(args.output_dir, exist_ok=True)
    ratings_path = os.path.join(args.output_dir, "merged_ratings.csv")
    merged.to_csv(ratings_path, index=False)
    print(f"Saved {ratings_path} ({len(merged)} rows)")

    # Group and compute accuracies
    print("Computing per-condition accuracies...")
    available_group = [c for c in GROUP_COLS if c in merged.columns]
    grouped = merged.groupby(available_group)

    has_jp = "jp_rating" in merged.columns and merged["jp_rating"].notna().any()

    for _, group in tqdm(grouped):
        row = group.iloc[0]

        # Build output directory
        base_dir = os.path.join(
            args.output_dir,
            str(row["MODEL_ID"]),
            f"from_{row['SOURCE']}_to_{row['BASE']}",
            str(row["METHOD"]),
            str(row["EVAL_SUB_DIR"]),
            str(row["STEER_SUB_DIR"]),
        )
        os.makedirs(base_dir, exist_ok=True)

        fn_base = (
            f"{row['N']}_{row['REPS']}_{row['STEERING_METHOD']}"
            f"_topk_{row['topk']}"
        )

        if has_jp:
            # Accuracy without relevance/fluency filtering
            acc_without = float((group["jp_rating"] == 5).mean())
            path_wo = os.path.join(
                base_dir, f"{fn_base}_gen_accuracy_wo_rf.json.accuracy.json"
            )
            with open(path_wo, "w") as f:
                json.dump({"q1": acc_without}, f, indent=2)

            # Accuracy with relevance/fluency filtering
            acc_with = float(
                (
                    (group["jp_rating"] == 5)
                    & (group["fluency_rating"] == 2)
                    & (group["relevance_rating"] == 2)
                ).mean()
            )
            path_w = os.path.join(
                base_dir, f"{fn_base}_gen_accuracy_w_rf.json.accuracy.json"
            )
            with open(path_w, "w") as f:
                json.dump({"q1": acc_with}, f, indent=2)
        else:
            # No behavioral judge: just save fluency/relevance pass rate
            acc_rf = float(
                (
                    (group["fluency_rating"] == 2)
                    & (group["relevance_rating"] == 2)
                ).mean()
            )
            path_rf = os.path.join(
                base_dir, f"{fn_base}_gen_rf_pass_rate.json"
            )
            with open(path_rf, "w") as f:
                json.dump({"rf_pass_rate": acc_rf}, f, indent=2)

    print("Done.")


if __name__ == "__main__":
    main()
