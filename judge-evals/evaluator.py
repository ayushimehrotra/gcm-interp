"""
Step 3: Run the vLLM judge model over a prompt CSV.

Reads a CSV with a prompt column (judge_prompt, fluency_prompt, or
relevance_prompt), feeds each prompt through a quantized Llama-70B judge,
and writes per-row results as JSONL plus an accuracy summary.

Usage:
    python evaluator.py --input_csv judge_prompts.csv --judge --batch_size 16
    python evaluator.py --input_csv relevance_fluency_prompts.csv --fluency
    python evaluator.py --input_csv relevance_fluency_prompts.csv --relevance
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from vllm import LLM, SamplingParams

from config import JUDGE_MODEL_NAME, PASSTHROUGH_COLS, extract_rating


# ---------------------------------------------------------------------------
# Model setup
# ---------------------------------------------------------------------------

SEED = 42


def make_llm(model_name: str = JUDGE_MODEL_NAME) -> LLM:
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("No GPUs detected!")
    print(f"Detected {num_gpus} GPU(s). Loading judge model: {model_name}")

    return LLM(
        model=model_name,
        quantization="bitsandbytes",
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        dtype="auto",
        max_num_seqs=64,
        max_model_len=4096,
        seed=SEED,
    )


def get_sampling_params() -> SamplingParams:
    return SamplingParams(
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        max_tokens=3,
        seed=SEED,
    )


# ---------------------------------------------------------------------------
# Batched generation
# ---------------------------------------------------------------------------

def generate_in_batches(llm, prompts, sampling_params, batch_size):
    """Yield decoded output strings batch by batch."""
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        results = llm.generate(batch, sampling_params)
        yield [r.outputs[0].text for r in results]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Run vLLM judge evaluation")
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--output_json", type=str, default=None)
    parser.add_argument("--skip_rows", type=int, default=0,
                        help="Skip first N rows (for resuming interrupted runs)")

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--relevance", action="store_true")
    mode.add_argument("--fluency", action="store_true")
    mode.add_argument("--judge", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()

    # Determine which prompt column to use
    if args.relevance:
        prompt_col = "relevance_prompt"
    elif args.fluency:
        prompt_col = "fluency_prompt"
    else:
        prompt_col = "judge_prompt"

    # Load data
    df = pd.read_csv(args.input_csv)
    assert prompt_col in df.columns, (
        f"CSV must contain a '{prompt_col}' column. Found: {list(df.columns)}"
    )

    # Resume support
    if args.skip_rows > 0:
        print(f"Skipping first {args.skip_rows} rows (resuming from row {args.skip_rows})")
        df = df.iloc[args.skip_rows :].reset_index(drop=True)

    prompts = df[prompt_col].tolist()

    # Output paths
    input_path = Path(args.input_csv)
    if args.output_json is None:
        args.output_json = str(
            input_path.with_suffix(f".{prompt_col}.judge_outputs.json")
        )
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    accuracy_path = str(input_path.with_suffix(f".{prompt_col}.judge_accuracy.json"))

    # Load model
    llm = make_llm()
    sampling_params = get_sampling_params()

    # Run evaluation
    buffer = []
    correct = 0
    total = 0
    batch_counter = 0
    flush_interval = 4  # flush to disk every N batches

    for batch_outputs in generate_in_batches(llm, prompts, sampling_params, args.batch_size):
        rows = df.iloc[total : total + len(batch_outputs)]

        for (_, row), output in zip(rows.iterrows(), batch_outputs):
            judge_rating = extract_rating(output)

            item = {
                **{col: row[col] for col in PASSTHROUGH_COLS if col in row},
                prompt_col: row[prompt_col],
                "judge_output": output,
                "judge_rating": judge_rating,
            }
            buffer.append(item)

            if judge_rating == 2:
                correct += 1
            total += 1

        batch_counter += 1

        if batch_counter % flush_interval == 0:
            _flush(buffer, out_path)
            buffer = []

    # Final flush
    if buffer:
        _flush(buffer, out_path)

    # Save accuracy
    with open(accuracy_path, "w", encoding="utf-8") as f:
        json.dump({"accuracy": correct / total if total else 0}, f, indent=2)

    print(f"Saved judge results -> {out_path}")
    print(f"Saved accuracy ({correct}/{total}) -> {accuracy_path}")


def _flush(items: list[dict], path: Path):
    """Append items as JSONL."""
    with open(path, "a", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
