"""
Run the full judge evaluation pipeline for a specific model/task.

Given a model, source, base, and algorithm(s), this script:
  1. Merges all matching *_gen.json files into a single CSV
  2. Builds behavioral judge, fluency, and relevance prompt columns
  3. Runs the vLLM judge for each prompt type (resumable)
  4. Computes per-condition accuracies
  5. (Optionally) generates plots

This is the script you run AFTER patching/steering has produced gen.json
files under results/.

Usage examples:

  # Evaluate sycophancy steering on Qwen with ATP
  python run_judge.py \\
      --model_name Qwen1.5-14B-Chat \\
      --source sycophancy-long \\
      --base non-sycophantic \\
      --algos atp

  # Evaluate multiple algorithms at once
  python run_judge.py \\
      --model_name Qwen1.5-14B-Chat \\
      --source sycophancy-long \\
      --base non-sycophantic \\
      --algos atp acp probes

  # Evaluate everything for a model (all tasks, all algos)
  python run_judge.py --model_name Qwen1.5-14B-Chat

  # Evaluate ALL results in the results/ directory
  python run_judge.py --all

  # Skip the behavioral judge (only fluency + relevance)
  python run_judge.py \\
      --model_name Qwen1.5-14B-Chat \\
      --source sycophancy-long \\
      --base non-sycophantic \\
      --algos atp \\
      --skip_judge

  # Also generate plots after evaluation
  python run_judge.py \\
      --model_name Qwen1.5-14B-Chat \\
      --source sycophancy-long \\
      --base non-sycophantic \\
      --algos atp \\
      --plots
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

from config import BASE_DIR, RUNS_DIR, DATA_DIR, TOKENIZER_MODEL_NAME


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run_step(description: str, cmd: list[str], cwd: str | None = None):
    """Run a subprocess, printing the step name and failing loudly."""
    print(f"\n{'=' * 60}")
    print(f"  {description}")
    print(f"{'=' * 60}")
    print(f"  cmd: {' '.join(cmd)}\n")
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        print(f"\nFAILED: {description} (exit code {result.returncode})")
        sys.exit(result.returncode)


def count_lines(path: str) -> int:
    """Count lines in a file (for resume support)."""
    if not os.path.exists(path):
        return 0
    with open(path) as f:
        return sum(1 for _ in f)


def make_workdir_name(model_name, source, base, algos):
    """Build a descriptive working directory name."""
    parts = []
    if model_name:
        parts.append(model_name)
    if source and base:
        parts.append(f"from_{source}_to_{base}")
    if algos:
        parts.append("_".join(algos))
    if not parts:
        parts.append("all")
    return "_".join(parts)


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def step_merge(workdir, runs_dir, data_dir, model_name, source, base, algos):
    """Step 1: Merge gen.json files into merged_eval_outputs.csv."""
    output = os.path.join(workdir, "merged_eval_outputs.csv")
    if os.path.exists(output):
        print(f"[1/5] Skipping merge ({output} exists)")
        return

    cmd = [
        sys.executable, "merge_outputs.py",
        "--runs_dir", str(runs_dir),
        "--data_dir", str(data_dir),
        "--output", output,
    ]
    if model_name:
        cmd += ["--model_name", model_name]
    if source:
        cmd += ["--source", source]
    if base:
        cmd += ["--base", base]
    if algos:
        cmd += ["--algos"] + algos

    run_step("[1/5] Merging generation outputs", cmd)


def step_build_prompts(workdir, skip_judge):
    """Step 2: Build prompt columns (judge + fluency + relevance)."""
    merged = os.path.join(workdir, "merged_eval_outputs.csv")

    # Build fluency + relevance prompts
    rf_csv = os.path.join(workdir, "relevance_fluency_prompts.csv")
    if os.path.exists(rf_csv):
        print("[2/5] Skipping relevance/fluency prompt build (file exists)")
    else:
        # Build both fluency and relevance in one pass using "all" mode,
        # but we only need fluency+relevance columns in this file.
        # We do two passes since build_prompts.py subcommands handle one at a time.
        cmd_flu = [
            sys.executable, "build_prompts.py", "fluency",
            "--input", merged,
            "--output", rf_csv,
            "--tokenizer", TOKENIZER_MODEL_NAME,
        ]
        run_step("[2/5] Building fluency prompts", cmd_flu)

        # Now add relevance column to the same CSV
        cmd_rel = [
            sys.executable, "build_prompts.py", "relevance",
            "--input", rf_csv,  # read from the fluency output
            "--output", rf_csv,
            "--tokenizer", TOKENIZER_MODEL_NAME,
        ]
        run_step("[2/5] Building relevance prompts", cmd_rel)

    # Build behavioral judge prompts (separate CSV)
    if not skip_judge:
        jp_csv = os.path.join(workdir, "judge_prompts.csv")
        if os.path.exists(jp_csv):
            print("[2/5] Skipping judge prompt build (file exists)")
        else:
            cmd_jp = [
                sys.executable, "build_prompts.py", "judge",
                "--input", merged,
                "--output", jp_csv,
                "--tokenizer", TOKENIZER_MODEL_NAME,
            ]
            run_step("[2/5] Building behavioral judge prompts", cmd_jp)


def step_evaluate(workdir, batch_size, skip_judge):
    """Step 3: Run vLLM judge for fluency, relevance, and (optionally) judge."""
    eval_jobs = []

    # Fluency
    rf_csv = os.path.join(workdir, "relevance_fluency_prompts.csv")
    flu_out = os.path.join(workdir, "relevance_fluency_prompts.fluency_prompt.judge_outputs.json")
    flu_acc = os.path.join(workdir, "relevance_fluency_prompts.fluency_prompt.judge_accuracy.json")
    eval_jobs.append(("fluency", rf_csv, flu_out, flu_acc, "--fluency"))

    # Relevance
    rel_out = os.path.join(workdir, "relevance_fluency_prompts.relevance_prompt.judge_outputs.json")
    rel_acc = os.path.join(workdir, "relevance_fluency_prompts.relevance_prompt.judge_accuracy.json")
    eval_jobs.append(("relevance", rf_csv, rel_out, rel_acc, "--relevance"))

    # Behavioral judge
    if not skip_judge:
        jp_csv = os.path.join(workdir, "judge_prompts.csv")
        jp_out = os.path.join(workdir, "judge_prompts.judge_prompt.judge_outputs.json")
        jp_acc = os.path.join(workdir, "judge_prompts.judge_prompt.judge_accuracy.json")
        eval_jobs.append(("judge", jp_csv, jp_out, jp_acc, "--judge"))

    for name, input_csv, out_json, acc_json, flag in eval_jobs:
        if os.path.exists(acc_json):
            print(f"[3/5] Skipping {name} eval (accuracy file exists)")
            continue

        skip = count_lines(out_json)
        cmd = [
            sys.executable, "evaluator.py",
            "--input_csv", input_csv,
            flag,
            "--batch_size", str(batch_size),
            "--skip_rows", str(skip),
            "--output_json", out_json,
        ]
        run_step(f"[3/5] Running {name} evaluation (skip_rows={skip})", cmd)


def step_accuracies(workdir):
    """Step 4: Compute per-condition accuracies."""
    jp_path = os.path.join(workdir, "judge_prompts.judge_prompt.judge_outputs.json")
    flu_path = os.path.join(workdir, "relevance_fluency_prompts.fluency_prompt.judge_outputs.json")
    rel_path = os.path.join(workdir, "relevance_fluency_prompts.relevance_prompt.judge_outputs.json")
    acc_dir = os.path.join(workdir, "accuracy")

    cmd = [
        sys.executable, "compute_accuracies.py",
        "--output_dir", acc_dir,
    ]

    # Only pass paths that exist
    if os.path.exists(jp_path):
        cmd += ["--jp_path", jp_path]
    if os.path.exists(flu_path):
        cmd += ["--flu_path", flu_path]
    if os.path.exists(rel_path):
        cmd += ["--rel_path", rel_path]

    run_step("[4/5] Computing accuracies", cmd)


def step_plots():
    """Step 5: Generate plots (runs from the project root)."""
    plots_script = str(BASE_DIR / "plots.py")
    if not os.path.exists(plots_script):
        print(f"[5/5] Skipping plots ({plots_script} not found)")
        return
    run_step("[5/5] Generating plots", [sys.executable, plots_script], cwd=str(BASE_DIR))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Run the full judge evaluation pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

  # Evaluate a specific task
  python run_judge.py \\
      --model_name Qwen1.5-14B-Chat \\
      --source sycophancy-long \\
      --base non-sycophantic \\
      --algos atp

  # Evaluate everything for a model
  python run_judge.py --model_name SOLAR-10.7B-Instruct-v1.0

  # Evaluate all results
  python run_judge.py --all
        """,
    )

    # Scoping filters
    p.add_argument("--model_name", type=str, default=None,
                    help="Model name as it appears in results/ "
                         "(e.g. Qwen1.5-14B-Chat)")
    p.add_argument("--source", type=str, default=None,
                    help="Source behavior (e.g. sycophancy-long, harmful, verse)")
    p.add_argument("--base", type=str, default=None,
                    help="Base/target behavior (e.g. non-sycophantic, harmless, prose)")
    p.add_argument("--algos", nargs="*", default=None,
                    help="Patching algorithms to include (e.g. atp acp probes random)")
    p.add_argument("--all", action="store_true",
                    help="Process ALL results in the results directory")

    # Pipeline control
    p.add_argument("--skip_judge", action="store_true",
                    help="Skip behavioral judge (only run fluency + relevance)")
    p.add_argument("--plots", action="store_true",
                    help="Generate plots after computing accuracies")
    p.add_argument("--batch_size", type=int, default=16,
                    help="Batch size for vLLM judge inference")

    # Paths
    p.add_argument("--runs_dir", default=str(RUNS_DIR),
                    help="Root directory containing patching results")
    p.add_argument("--data_dir", default=str(DATA_DIR),
                    help="Root directory containing test data")
    p.add_argument("--workdir", default=None,
                    help="Working directory for intermediate files "
                         "(auto-generated if not specified)")

    return p.parse_args()


def main():
    args = parse_args()

    # Validate: must specify at least --model_name or --all
    if not args.all and not args.model_name:
        print("Error: specify --model_name (and optionally --source, --base, --algos)")
        print("       or use --all to process everything.")
        print("       Run with --help for examples.")
        sys.exit(1)

    # Build working directory
    if args.workdir:
        workdir = args.workdir
    else:
        name = make_workdir_name(args.model_name, args.source, args.base, args.algos)
        workdir = os.path.join("workdirs", name)

    os.makedirs(workdir, exist_ok=True)
    print(f"Working directory: {workdir}")

    # Print what we're evaluating
    print(f"\nScope:")
    print(f"  Model:  {args.model_name or '(all)'}")
    print(f"  Source: {args.source or '(all)'}")
    print(f"  Base:   {args.base or '(all)'}")
    print(f"  Algos:  {', '.join(args.algos) if args.algos else '(all)'}")
    print(f"  Judge:  {'skip' if args.skip_judge else 'yes'}")
    print()

    # Run pipeline
    step_merge(workdir, args.runs_dir, args.data_dir,
               args.model_name, args.source, args.base, args.algos)

    step_build_prompts(workdir, args.skip_judge)

    step_evaluate(workdir, args.batch_size, args.skip_judge)

    step_accuracies(workdir)

    if args.plots:
        step_plots()

    print(f"\nPipeline complete. Results in: {workdir}/")


if __name__ == "__main__":
    main()