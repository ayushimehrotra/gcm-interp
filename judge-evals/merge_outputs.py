"""
Step 1: Merge all generation output files into a single CSV.

Scans RUNS_DIR for *_gen.json files, extracts path metadata, loads
the matching test queries, and writes merged_eval_outputs.csv.

Usage:
    python merge_outputs.py [--runs_dir DIR] [--data_dir DIR] [--output FILE]
"""

import argparse
import json
import math
import glob
from pathlib import Path

import numpy as np
import pandas as pd

from config import BASE_DIR, RUNS_DIR, DATA_DIR, GEN_RE


def extract_path_metadata(path: str) -> dict:
    """Pull model/task/method metadata from the directory structure."""
    parts = Path(path).parts
    runs_idx = parts.index("results")

    model_id = parts[runs_idx + 1]
    from_to = parts[runs_idx + 2]
    _, source, _, base = from_to.split("_")

    method = parts[runs_idx + 3]
    valid_methods = {"acp", "atp", "atp-zero", "probes", "random"}
    if method not in valid_methods:
        raise ValueError(f"Unexpected METHOD: {method} in path: {path}")

    eval_sub_dir = parts[runs_idx + 4]
    steer_sub_dir = parts[runs_idx + 5]
    sub_dir = parts[runs_idx + 6]
    if sub_dir != "eval":
        raise ValueError(f"Unexpected SUB_DIR: {sub_dir} in path: {path}")

    filename = parts[runs_idx + 7]
    m = GEN_RE.match(filename)
    if not m:
        raise ValueError(f"Filename does not match expected pattern: {filename}")

    return {
        "MODEL_ID": model_id,
        "SOURCE": source,
        "BASE": base,
        "METHOD": method,
        "EVAL_SUB_DIR": eval_sub_dir,
        "STEER_SUB_DIR": steer_sub_dir,
        **m.groupdict(),
        "filename": filename,
    }


def load_test_queries(data_dir: str, model_id: str, source: str, base: str) -> list[str]:
    """Load the user-turn text from the test JSONL."""
    logits_path = f"{data_dir}/{model_id}/{source}/{base}-test.jsonl"
    queries = []
    with open(logits_path) as f:
        for line in f:
            obj = json.loads(line)
            queries.append(obj["prompt"][-1]["content"])
    return queries


def validate_record(record: dict):
    """Raise on NaN or non-primitive values."""
    for key, value in record.items():
        if isinstance(value, float) and (math.isnan(value) or np.isnan(value)):
            raise ValueError(f"NaN value for '{key}'")
        if not isinstance(value, (str, int, float)):
            raise TypeError(
                f"Invalid type for '{key}': {type(value).__name__} ({value})"
            )


def discover_gen_files(
    runs_dir: str,
    model_name: str | None = None,
    source: str | None = None,
    base: str | None = None,
    algos: list[str] | None = None,
) -> list[str]:
    """
    Glob for *_gen.json files, optionally filtered by model/task/algo.

    When filters are provided the glob is scoped to the matching subtree,
    which is much faster than scanning the entire results directory.
    """
    # Build a scoped glob pattern when filters are given
    model_part = model_name or "**"
    task_part = f"from_{source}_to_{base}" if (source and base) else "**"

    if algos:
        gen_files = []
        for algo in algos:
            pattern = f"{runs_dir}/{model_part}/{task_part}/{algo}/**/*_gen.json"
            gen_files.extend(sorted(glob.glob(pattern, recursive=True)))
    else:
        pattern = f"{runs_dir}/{model_part}/{task_part}/**/*_gen.json"
        gen_files = sorted(glob.glob(pattern, recursive=True))

    gen_files = [f for f in gen_files if GEN_RE.search(Path(f).name)]
    return gen_files


def main():
    parser = argparse.ArgumentParser(description="Merge gen.json files into one CSV")
    parser.add_argument("--runs_dir", default=str(RUNS_DIR))
    parser.add_argument("--data_dir", default=str(DATA_DIR))
    parser.add_argument("--output", default="merged_eval_outputs.csv")
    parser.add_argument("--model_name", default=None,
                        help="Filter to a specific model (e.g. Qwen1.5-14B-Chat)")
    parser.add_argument("--source", default=None,
                        help="Filter to a specific source (e.g. sycophancy-long)")
    parser.add_argument("--base", default=None,
                        help="Filter to a specific base (e.g. non-sycophantic)")
    parser.add_argument("--algos", nargs="*", default=None,
                        help="Filter to specific algorithms (e.g. atp acp)")
    args = parser.parse_args()

    runs_dir = args.runs_dir
    data_dir = args.data_dir

    # Discover generation files (scoped if filters provided)
    gen_files = discover_gen_files(
        runs_dir, args.model_name, args.source, args.base, args.algos
    )
    print(f"Found {len(gen_files)} gen files matching pattern")

    query_cache: dict[tuple, list[str]] = {}
    output_records = []

    for gpath in gen_files:
        try:
            meta = extract_path_metadata(gpath)
        except ValueError as e:
            print(f"Skipping {gpath}: {e}")
            continue

        model_id = meta["MODEL_ID"]
        source = meta["SOURCE"]
        base = meta["BASE"]

        with open(gpath) as f:
            items = json.load(f)

        # Cache test queries per (model, source, base)
        cache_key = (model_id, source, base)
        if cache_key not in query_cache:
            query_cache[cache_key] = load_test_queries(data_dir, model_id, source, base)
        test_queries = query_cache[cache_key]

        old_key = f"old_{base}"
        edit_key = f"edit_{base}"

        for i, item in enumerate(items):
            record = {
                "query": item["query"].strip().replace("\r", "\n"),
                "post-intervention-response": item[edit_key].strip().replace("\r", "\n"),
                "original-response": item[old_key].strip().replace("\r", "\n"),
                "filename": meta["filename"].strip().replace("\r", "\n"),
                "data_path_query": test_queries[i].strip().replace("\r", "\n"),
                "MODEL_ID": meta["MODEL_ID"],
                "SOURCE": source,
                "BASE": base,
                "METHOD": meta["METHOD"],
                "EVAL_SUB_DIR": meta["EVAL_SUB_DIR"],
                "STEER_SUB_DIR": meta["STEER_SUB_DIR"],
                "N": meta["N"],
                "REPS": meta["REPS"],
                "STEERING_METHOD": meta["STEERING_METHOD"],
                "topk": meta["topk"],
                "TEST_FILE": meta["TEST_FILE"],
            }
            validate_record(record)
            output_records.append(record)

    df = pd.DataFrame(output_records)

    if df.isna().any().any():
        nan_rows = df[df.isna().any(axis=1)]
        raise ValueError(f"NaNs detected in dataframe!\n{nan_rows.to_string(index=False)}")

    df.to_csv(args.output, index=False)
    print(f"Saved {args.output}  (shape: {df.shape})")


if __name__ == "__main__":
    main()
