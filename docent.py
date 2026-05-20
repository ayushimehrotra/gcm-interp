"""
Ingest post-intervention steering results into Docent for any task and model.

Discovers condition directories automatically from the results tree.

Directory structure:
  results/{model}/
    from_{source}_to_{target}/atp/{eval_task}_eval/{steer_variant}_steer/eval/
      {topk}_targeted_steer_{factor}_{eval_task}_gen.json   (50 entries each)

All 50 entries per file are processed (no sampling).

Target steering generations for comparison (Phase 2) come from a JSONL where
each line has: {"id": ..., "prompt": [{"role": "user", ...}, {"role": "assistant", ...}]}
For the verse task this is data/{model}/verse-long/prose-undesired-all.jsonl.

Usage:
  /home/ubuntu/docent-env/bin/python3 docent.py \\
    --results_dir results/Qwen1.5-14B-Chat \\
    --model Qwen1.5-14B-Chat \\
    --eval_task verse-long \\
    --target_file data/Qwen1.5-14B-Chat/verse-long/prose-undesired-all.jsonl \\
    --collection_name verse_qwen14b_steering

  /home/ubuntu/docent-env/bin/python3 docent.py \\
    --results_dir results/Qwen1.5-32B-Chat \\
    --model Qwen1.5-32B-Chat \\
    --eval_task verse-long \\
    --target_file data/Qwen1.5-32B-Chat/verse-long/prose-undesired-all.jsonl \\
    --collection_name verse_qwen32b_steering

  /home/ubuntu/docent-env/bin/python3 docent.py \\
    --results_dir results/OLMo-2-1124-13B-DPO \\
    --model OLMo-2-1124-13B-DPO \\
    --eval_task verse-long \\
    --target_file data/OLMo-2-1124-13B-DPO/verse-long/prose-undesired-all.jsonl \\
    --collection_name verse_olmo13b_steering
"""

import sys
import os
sys.path = [p for p in sys.path if os.path.abspath(p) != os.path.abspath(os.path.dirname(__file__))]

import argparse
import json
import re
from pathlib import Path

from docent.sdk.client import Docent
from docent.data_models import AgentRun, Transcript
from docent.data_models.chat import SystemMessage, UserMessage, AssistantMessage


# ---------------------------------------------------------------------------
# Path parsers
# ---------------------------------------------------------------------------

def parse_conditions_from_path(filepath: Path) -> dict | None:
    """Extract experimental conditions from directory path."""
    from_segment = None
    steer_segment = None
    for part in filepath.parts:
        if re.match(r"from_.+_to_.+", part):
            from_segment = part
        if re.match(r".+_steer$", part):
            steer_segment = part

    if from_segment is None or steer_segment is None:
        print(f"  [WARN] Could not parse conditions from path: {filepath}")
        return None

    m = re.match(r"from_(.+)_to_(.+)", from_segment)
    if not m:
        print(f"  [WARN] Unexpected from_segment format: {from_segment}")
        return None

    source = m.group(1)
    target = m.group(2)
    steer_variant = re.sub(r"_steer$", "", steer_segment)

    return {
        "from_segment": from_segment,
        "steer_segment": steer_segment,
        "source": source,
        "target": target,
        "steer_variant": steer_variant,
        "condition": f"from_{source}__steer_{steer_variant}",
    }


def parse_filename(fname: str) -> dict | None:
    """Parse topk and steering_factor from filenames like 4_targeted_steer_0.01_verse-long_gen.json."""
    stem = Path(fname).stem
    m = re.match(r"^(\d+)_targeted_steer_([\d.]+)_", stem)
    if not m:
        print(f"  [WARN] Could not parse filename: {fname}")
        return None

    topk = int(m.group(1))
    try:
        steering_factor = float(m.group(2))
    except ValueError:
        print(f"  [WARN] Could not convert factor '{m.group(2)}' to float in {fname}")
        return None

    return {"topk": topk, "steering_factor": steering_factor, "filename": fname}


# ---------------------------------------------------------------------------
# Transcript builder
# ---------------------------------------------------------------------------

def parse_query(query_raw: str) -> tuple[str, str]:
    """Split raw query string into (system_text, user_text)."""
    system_text = ""
    user_text = query_raw.strip()
    if "\nuser\n" in query_raw:
        parts = query_raw.split("\nuser\n", 1)
        system_text = parts[0].replace("system\n", "").strip()
        user_text = parts[1].replace("\nassistant\n", "").strip()
    return system_text, user_text


def make_transcript(system_text: str, user_text: str, response: str, role: str) -> Transcript:
    messages = []
    if system_text:
        messages.append(SystemMessage(content=system_text))
    messages.append(UserMessage(content=user_text))
    messages.append(AssistantMessage(content=response))
    return Transcript(messages=messages, metadata={"role": role})


# ---------------------------------------------------------------------------
# AgentRun builders
# ---------------------------------------------------------------------------

def build_result_runs(
    filepath: Path,
    conditions: dict,
    file_meta: dict,
    model: str,
) -> list[AgentRun]:
    """Convert one results JSON file into AgentRuns (all entries, no sampling).

    Handles any task by detecting the old_*/edit_* field names dynamically.
    """
    with open(filepath) as f:
        entries = json.load(f)

    runs = []
    for idx, entry in enumerate(entries):
        system_text, user_text = parse_query(entry["query"])

        # Detect the response field names: old_{target} and edit_{target}
        old_key = next((k for k in entry if k.startswith("old_")), None)
        edit_key = next((k for k in entry if k.startswith("edit_")), None)
        if old_key is None or edit_key is None:
            print(f"  [WARN] Entry {idx} in {filepath.name} missing old_/edit_ keys: {list(entry.keys())}")
            continue

        baseline: str = entry[old_key]
        steered: str = entry[edit_key]
        changed = baseline.strip() != steered.strip()

        run = AgentRun(
            transcripts=[
                make_transcript(system_text, user_text, steered, "steered"),
                make_transcript(system_text, user_text, baseline, "baseline"),
            ],
            metadata={
                "from_segment": conditions["from_segment"],
                "steer_segment": conditions["steer_segment"],
                "source": conditions["source"],
                "target": conditions["target"],
                "steer_variant": conditions["steer_variant"],
                "condition": conditions["condition"],
                "topk": file_meta["topk"],
                "steering_factor": file_meta["steering_factor"],
                "prompt_id": idx,
                "prompt_text": user_text,
                "model": model,
                "steering_changed_output": changed,
                "run_type": "intervention",
                "source_file": str(filepath),
            },
        )
        runs.append(run)

    return runs


def build_target_runs(target_file: Path, model: str) -> list[AgentRun]:
    """Load target steering generation JSONL and build AgentRuns for comparison (Phase 2).

    Each entry shows what the model produces on the target-behavior prompts,
    serving as a qualitative reference for the steered outputs.
    """
    runs = []
    with open(target_file) as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)

            prompt = entry.get("prompt", [])
            user_msg = next((m for m in prompt if m["role"] == "user"), None)
            asst_msg = next((m for m in prompt if m["role"] == "assistant"), None)

            if user_msg is None or asst_msg is None:
                print(f"  [WARN] Skipping target entry {idx}: missing user or assistant turn")
                continue

            run = AgentRun(
                transcripts=[
                    make_transcript("", user_msg["content"], asst_msg["content"], "target"),
                ],
                metadata={
                    "prompt_id": entry.get("id", idx),
                    "prompt_text": user_msg["content"],
                    "model": model,
                    "run_type": "target",
                    "source_file": str(target_file),
                },
            )
            runs.append(run)

    return runs


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def discover_eval_files(results_root: Path, eval_task: str) -> list[Path]:
    """Walk results_root to find all gen JSON files under the eval_task eval dirs."""
    return sorted(results_root.glob(f"*/atp/{eval_task}_eval/*_steer/eval/*.json"))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Ingest post-intervention steering results into Docent (any task, any model)."
    )
    parser.add_argument(
        "--results_dir",
        required=True,
        help="Path to results/{model}/ directory",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model name for metadata (e.g. Qwen1.5-14B-Chat, OLMo-2-1124-13B-DPO)",
    )
    parser.add_argument(
        "--eval_task",
        required=True,
        help="Eval task directory name (e.g. verse-long, sycophancy-long)",
    )
    parser.add_argument(
        "--target_file",
        default=None,
        help="Path to target steering generation JSONL for Phase 2 comparison",
    )
    parser.add_argument(
        "--collection_name",
        required=True,
        help="Name for the Docent collection",
    )
    parser.add_argument(
        "--collection_description",
        default="",
        help="Optional description for the Docent collection",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Parse and convert without uploading to Docent",
    )
    args = parser.parse_args()

    client = Docent()
    collection_id = None

    if not args.dry_run:
        description = args.collection_description or (
            f"Post-intervention steering results — model: {args.model}, "
            f"eval task: {args.eval_task}."
        )
        collection_id = client.create_collection(
            name=args.collection_name,
            description=description,
        )
        print(f"Created collection : {collection_id}")
        print(f"View at            : https://docent.transluce.org/dashboard/{collection_id}\n")

    results_root = Path(args.results_dir)
    total_runs = 0
    total_files = 0

    # ------------------------------------------------------------------
    # Phase 1: post-intervention results (steered vs baseline)
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"Phase 1: Post-intervention results")
    print(f"  results_dir : {results_root}")
    print(f"  eval_task   : {args.eval_task}")

    json_files = discover_eval_files(results_root, args.eval_task)
    print(f"  JSON files  : {len(json_files)}")

    batch: list[AgentRun] = []
    for fpath in json_files:
        conditions = parse_conditions_from_path(fpath)
        file_meta = parse_filename(fpath.name)
        if conditions is None or file_meta is None:
            continue

        print(
            f"    topk={file_meta['topk']:2d}  "
            f"factor={file_meta['steering_factor']:.4f}  "
            f"-> {conditions['condition']}"
        )

        runs = build_result_runs(fpath, conditions, file_meta, args.model)
        batch.extend(runs)
        total_runs += len(runs)
        total_files += 1

    if batch and not args.dry_run:
        client.add_agent_runs(collection_id, batch)
        print(f"  Uploaded {len(batch)} intervention runs")

    # ------------------------------------------------------------------
    # Phase 2: target steering generation comparison
    # ------------------------------------------------------------------
    if args.target_file:
        target_path = Path(args.target_file)
        print(f"\n{'='*60}")
        print(f"Phase 2: Target steering generation comparison")
        print(f"  target_file : {target_path}")

        if not target_path.exists():
            print(f"  [WARN] Target file not found: {target_path}")
        else:
            target_runs = build_target_runs(target_path, args.model)
            print(f"  Target entries : {len(target_runs)}")
            total_runs += len(target_runs)

            if target_runs and not args.dry_run:
                client.add_agent_runs(collection_id, target_runs)
                print(f"  Uploaded {len(target_runs)} target runs")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"{'DRY RUN - ' if args.dry_run else ''}Done.")
    print(f"  Files processed : {total_files}")
    print(f"  Runs converted  : {total_runs}")
    if not args.dry_run:
        print(f"  Collection ID   : {collection_id}")
        print(f"  View at         : https://docent.transluce.org/dashboard/{collection_id}")


if __name__ == "__main__":
    main()
