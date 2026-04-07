"""
Run the full judge evaluation pipeline.

Strategy:
  Phase 1 (per file, fast): convert each gen.json → CSV → build prompt CSVs
  Phase 2 (batch, GPU):     concatenate all prompt CSVs, run vLLM judge ONCE
                             for fluency, relevance, and behavioral judge
  Phase 3 (aggregate):      compute per-condition accuracies from batch outputs

This avoids re-loading the 40GB judge model for every file.

Usage examples:

  # All sycophancy single-eval for both models
  python run_judge.py --eval_subdir sycophancy-single_eval --algos atp

  # Specific model + task
  python run_judge.py \\
      --model_name Qwen1.5-14B-Chat \\
      --source sycophancy-long \\
      --base non-sycophantic \\
      --eval_subdir sycophancy-single_eval \\
      --algos atp

  # Everything for a model
  python run_judge.py --model_name Qwen1.5-14B-Chat

  # All results
  python run_judge.py --all

  # Skip behavioral judge (only fluency + relevance)
  python run_judge.py --eval_subdir sycophancy-single_eval --skip_judge

  # Generate plots after evaluation
  python run_judge.py --eval_subdir sycophancy-single_eval --plots
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

from config import BASE_DIR, RUNS_DIR, DATA_DIR, TOKENIZER_MODEL_NAME
from merge_outputs import (
    discover_gen_files,
    extract_path_metadata,
    gen_to_csv,
)
from build_prompts import (
    load_tokenizer,
    build_fluency_prompts,
    build_relevance_prompts,
    build_judge_prompts,
)

ACCURACY_DIR = BASE_DIR / "judge-evals" / "accuracy"
WORKDIRS_ROOT = BASE_DIR / "judge-evals" / "workdirs"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run_step(description: str, cmd: list[str], cwd: str | None = None):
    print(f"\n{'=' * 60}")
    print(f"  {description}")
    print(f"{'=' * 60}")
    print(f"  cmd: {' '.join(cmd)}\n")
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        print(f"\nFAILED: {description} (exit code {result.returncode})")
        sys.exit(result.returncode)


def count_lines(path: str) -> int:
    if not os.path.exists(path):
        return 0
    with open(path) as f:
        return sum(1 for _ in f)


def gen_workdir(gen_path: str) -> Path:
    """Derive a per-file workdir under judge-evals/workdirs/."""
    p = Path(gen_path)
    stem = p.stem
    if stem.endswith("_gen"):
        stem = stem[:-4]
    try:
        rel = p.parent.relative_to(RUNS_DIR)
    except ValueError:
        rel = Path(p.parent.name)
    return WORKDIRS_ROOT / rel / stem


def accuracy_paths(meta: dict) -> tuple[Path, Path]:
    base_dir = (
        ACCURACY_DIR
        / meta["MODEL_ID"]
        / f"from_{meta['SOURCE']}_to_{meta['BASE']}"
        / meta["METHOD"]
        / meta["EVAL_SUB_DIR"]
        / meta["STEER_SUB_DIR"]
    )
    fn_base = f"{meta['N']}_{meta['REPS']}_{meta['STEERING_METHOD']}_topk_{meta['topk']}"
    wo_rf = base_dir / f"{fn_base}_gen_accuracy_wo_rf.json.accuracy.json"
    w_rf  = base_dir / f"{fn_base}_gen_accuracy_w_rf.json.accuracy.json"
    return wo_rf, w_rf


def accuracy_exists(meta: dict) -> bool:
    wo_rf, w_rf = accuracy_paths(meta)
    return wo_rf.exists() and w_rf.exists()


# ---------------------------------------------------------------------------
# Phase 1: per-file CSV + prompt building
# ---------------------------------------------------------------------------

def phase1_prepare(gen_files: list[str], data_dir: str, skip_judge: bool, force: bool):
    """
    For each gen file: convert to CSV and build prompt CSVs.
    Tokenizer is loaded ONCE and reused for all files.
    Returns list of workdirs that were prepared.
    """
    # Determine which files actually need processing
    to_process = []
    for gen_path in gen_files:
        try:
            meta = extract_path_metadata(gen_path)
        except ValueError as e:
            print(f"Skipping {gen_path}: {e}")
            continue
        if not force and accuracy_exists(meta):
            print(f"  Skip (done): {Path(gen_path).name}")
            continue
        to_process.append(gen_path)

    if not to_process:
        return []

    print(f"\nPhase 1: {len(to_process)} files to prepare. Loading tokenizer once...")
    tokenizer = load_tokenizer(TOKENIZER_MODEL_NAME)
    print("Tokenizer loaded.\n")

    prepared = []
    errors = 0

    for gen_path in to_process:
        workdir = gen_workdir(gen_path)
        workdir.mkdir(parents=True, exist_ok=True)
        name = Path(gen_path).stem

        # Step 1: gen.json → eval_output.csv
        eval_csv = workdir / "eval_output.csv"
        if not eval_csv.exists():
            try:
                gen_to_csv(gen_path, data_dir, str(eval_csv))
            except (ValueError, FileNotFoundError) as e:
                print(f"  ERROR converting {name}: {e}")
                errors += 1
                continue

        # Load CSV (force string dtypes on text columns)
        df = pd.read_csv(eval_csv, keep_default_na=False,
                         dtype={"post-intervention-response": str,
                                "original-response": str,
                                "query": str,
                                "data_path_query": str})

        # Step 2: fluency + relevance prompts
        rf_csv = workdir / "relevance_fluency_prompts.csv"
        if not rf_csv.exists():
            print(f"  Building fluency+relevance prompts: {name}")
            rf_df = build_fluency_prompts(df.copy(), tokenizer)
            rf_df = build_relevance_prompts(rf_df, tokenizer)
            rf_df.to_csv(rf_csv, index=False)

        # Step 2b: behavioral judge prompts
        if not skip_judge:
            jp_csv = workdir / "judge_prompts.csv"
            if not jp_csv.exists():
                print(f"  Building judge prompts: {name}")
                try:
                    jp_df = build_judge_prompts(df.copy(), tokenizer)
                    jp_df.to_csv(jp_csv, index=False)
                except (ValueError, AssertionError) as e:
                    print(f"  ERROR building judge prompts for {name}: {e}")
                    errors += 1
                    continue

        prepared.append(workdir)

    print(f"\nPhase 1 done: {len(prepared)} files prepared, {errors} errors")
    return prepared


# ---------------------------------------------------------------------------
# Phase 2: batch evaluation (load model once)
# ---------------------------------------------------------------------------

def phase2_evaluate(workdirs: list[Path], batch_dir: Path,
                    batch_size: int, skip_judge: bool):
    """
    Concatenate all per-file prompt CSVs, run evaluator once per prompt type.
    """
    batch_dir.mkdir(parents=True, exist_ok=True)

    # Gather and concatenate
    rf_frames, jp_frames = [], []
    for wd in workdirs:
        rf_csv = wd / "relevance_fluency_prompts.csv"
        jp_csv = wd / "judge_prompts.csv"
        if rf_csv.exists():
            rf_frames.append(pd.read_csv(rf_csv, keep_default_na=False))
        if not skip_judge and jp_csv.exists():
            jp_frames.append(pd.read_csv(jp_csv, keep_default_na=False))

    if not rf_frames:
        print("No prompt CSVs found — nothing to evaluate.")
        return

    combined_rf = pd.concat(rf_frames, ignore_index=True)
    combined_rf_path = batch_dir / "combined_rf_prompts.csv"
    combined_rf.to_csv(combined_rf_path, index=False)
    print(f"Combined RF CSV: {len(combined_rf)} rows → {combined_rf_path}")

    flu_out = batch_dir / "fluency_judge_outputs.json"
    rel_out = batch_dir / "relevance_judge_outputs.json"
    flu_acc = batch_dir / "fluency_judge_accuracy.json"
    rel_acc = batch_dir / "relevance_judge_accuracy.json"

    # Fluency
    if not flu_acc.exists():
        skip = count_lines(str(flu_out))
        run_step("Batch fluency evaluation", [
            sys.executable, "evaluator.py",
            "--input_csv", str(combined_rf_path),
            "--fluency",
            "--batch_size", str(batch_size),
            "--skip_rows", str(skip),
            "--output_json", str(flu_out),
        ])
    else:
        print("Skipping fluency eval (already done)")

    # Relevance
    if not rel_acc.exists():
        skip = count_lines(str(rel_out))
        run_step("Batch relevance evaluation", [
            sys.executable, "evaluator.py",
            "--input_csv", str(combined_rf_path),
            "--relevance",
            "--batch_size", str(batch_size),
            "--skip_rows", str(skip),
            "--output_json", str(rel_out),
        ])
    else:
        print("Skipping relevance eval (already done)")

    # Behavioral judge
    if not skip_judge and jp_frames:
        combined_jp = pd.concat(jp_frames, ignore_index=True)
        combined_jp_path = batch_dir / "combined_judge_prompts.csv"
        combined_jp.to_csv(combined_jp_path, index=False)
        print(f"Combined judge CSV: {len(combined_jp)} rows → {combined_jp_path}")

        jp_out = batch_dir / "judge_outputs.json"
        jp_acc = batch_dir / "judge_accuracy.json"
        if not jp_acc.exists():
            skip = count_lines(str(jp_out))
            run_step("Batch behavioral judge evaluation", [
                sys.executable, "evaluator.py",
                "--input_csv", str(combined_jp_path),
                "--judge",
                "--batch_size", str(batch_size),
                "--skip_rows", str(skip),
                "--output_json", str(jp_out),
            ])
        else:
            print("Skipping behavioral judge eval (already done)")


# ---------------------------------------------------------------------------
# Phase 3: compute accuracies from batch outputs
# ---------------------------------------------------------------------------

def phase3_accuracies(batch_dir: Path, skip_judge: bool):
    """Compute per-condition accuracies from the batch judge output files."""
    jp_out  = batch_dir / "judge_outputs.json"
    flu_out = batch_dir / "fluency_judge_outputs.json"
    rel_out = batch_dir / "relevance_judge_outputs.json"

    cmd = [
        sys.executable, "compute_accuracies.py",
        "--output_dir", str(ACCURACY_DIR),
    ]
    if not skip_judge and jp_out.exists():
        cmd += ["--jp_path", str(jp_out)]
    if flu_out.exists():
        cmd += ["--flu_path", str(flu_out)]
    if rel_out.exists():
        cmd += ["--rel_path", str(rel_out)]

    run_step("Computing per-condition accuracies", cmd)


# ---------------------------------------------------------------------------
# Optional: plots
# ---------------------------------------------------------------------------

def step_plots(plots_args: list[str] | None = None):
    plots_script = str(BASE_DIR / "plots.py")
    if not os.path.exists(plots_script):
        print(f"Skipping plots ({plots_script} not found)")
        return
    cmd = [sys.executable, plots_script] + (plots_args or [])
    run_step("Generating plots", cmd, cwd=str(BASE_DIR))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Run judge evaluation pipeline (batch GPU, per-file CSVs)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--model_name",   default=None)
    p.add_argument("--source",       default=None)
    p.add_argument("--base",         default=None)
    p.add_argument("--algos",        nargs="*", default=None)
    p.add_argument("--eval_subdir",  default=None,
                   help="e.g. sycophancy-single_eval")
    p.add_argument("--steer_subdir", default=None,
                   help="e.g. sycophancy-long_steer")
    p.add_argument("--all",          action="store_true")
    p.add_argument("--skip_judge",   action="store_true",
                   help="Skip behavioral judge (fluency + relevance only)")
    p.add_argument("--force",        action="store_true",
                   help="Re-process even if accuracy files exist")
    p.add_argument("--batch_size",   type=int, default=16)
    p.add_argument("--plots",        action="store_true")
    p.add_argument("--plots_args",   nargs="*", default=None)
    p.add_argument("--runs_dir",     default=str(RUNS_DIR))
    p.add_argument("--data_dir",     default=str(DATA_DIR))
    p.add_argument("--batch_dir",    default=None,
                   help="Where to store concatenated prompts + batch outputs "
                        "(default: judge-evals/workdirs/batch_{eval_subdir}/)")
    return p.parse_args()


def main():
    args = parse_args()

    if not args.all and not any([
        args.model_name, args.source, args.base,
        args.algos, args.eval_subdir, args.steer_subdir,
    ]):
        print("Error: specify at least one filter or --all. Run --help for examples.")
        sys.exit(1)

    gen_files = discover_gen_files(
        args.runs_dir, args.model_name, args.source, args.base, args.algos,
        args.eval_subdir, args.steer_subdir,
    )
    print(f"Found {len(gen_files)} gen files")

    # Batch dir: shared across all files processed in this run
    if args.batch_dir:
        batch_dir = Path(args.batch_dir)
    else:
        tag = args.eval_subdir or "all"
        batch_dir = WORKDIRS_ROOT / f"batch_{tag}"

    print(f"Batch dir: {batch_dir}")
    print(f"Accuracy dir: {ACCURACY_DIR}\n")

    # Phase 1: convert + build prompts (fast, no GPU)
    print("=" * 60)
    print("  PHASE 1: Convert gen files + build prompt CSVs")
    print("=" * 60)
    prepared_dirs = phase1_prepare(gen_files, args.data_dir, args.skip_judge, args.force)

    if not prepared_dirs:
        print("Nothing to evaluate.")
    else:
        # Phase 2: batch GPU evaluation (model loaded once)
        print("\n" + "=" * 60)
        print("  PHASE 2: Batch GPU evaluation")
        print("=" * 60)
        phase2_evaluate(prepared_dirs, batch_dir, args.batch_size, args.skip_judge)

        # Phase 3: compute per-condition accuracies
        print("\n" + "=" * 60)
        print("  PHASE 3: Compute accuracies")
        print("=" * 60)
        phase3_accuracies(batch_dir, args.skip_judge)

    print(f"\nPipeline complete. Accuracy files: {ACCURACY_DIR}")

    if args.plots:
        step_plots(args.plots_args)


if __name__ == "__main__":
    main()
