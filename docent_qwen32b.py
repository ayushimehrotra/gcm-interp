"""
Ingest verse/prose steering experiment results into Docent — Qwen1.5-32B-Chat.

Usage:
  /home/ubuntu/docent-env/bin/python3 docent_qwen32b.py --results_dir results/Qwen1.5-32B-Chat
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


CONDITION_DIRS = [
    ("from_verse-long_to_prose",   "verse-long_steer"),
    ("from_verse-long_to_prose",   "verse-single_steer"),
    ("from_verse-single_to_prose", "verse-long_steer"),
    ("from_verse-single_to_prose", "verse-single_steer"),
]


def parse_condition_from_path(filepath: Path) -> dict | None:
    parts = filepath.parts
    from_segment = None
    steer_segment = None
    for p in parts:
        if re.match(r"from_verse-(long|single)_to_prose", p):
            from_segment = p
        if re.match(r"verse-(long|single)_steer", p):
            steer_segment = p

    if from_segment is None or steer_segment is None:
        print(f"  [WARN] Could not parse conditions from path: {filepath}")
        return None

    loc_match = re.search(r"verse-(long|single)", from_segment)
    steer_match = re.search(r"verse-(long|single)", steer_segment)
    localization = "longform" if loc_match.group(1) == "long" else "single_token"
    steering = "longform" if steer_match.group(1) == "long" else "single_token"

    return {
        "localization": localization,
        "steering": steering,
        "localization_raw": f"verse-{loc_match.group(1)}",
        "steering_raw": f"verse-{steer_match.group(1)}",
        "condition": f"{localization}_loc__{steering}_steer",
    }


def parse_filename(fname: str) -> dict | None:
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


def parse_query(query_raw: str) -> tuple[str, str]:
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


def build_agent_runs(filepath: Path, conditions: dict, file_meta: dict) -> list[AgentRun]:
    with open(filepath) as f:
        entries = json.load(f)
    runs = []
    for idx, entry in enumerate(entries):
        system_text, user_text = parse_query(entry["query"])
        old_prose: str = entry["old_prose"]
        edit_prose: str = entry["edit_prose"]
        changed = old_prose.strip() != edit_prose.strip()
        run = AgentRun(
            transcripts=[
                make_transcript(system_text, user_text, edit_prose, "steered"),
                make_transcript(system_text, user_text, old_prose, "baseline"),
            ],
            metadata={
                "localization": conditions["localization"],
                "steering": conditions["steering"],
                "condition": conditions["condition"],
                "localization_raw": conditions["localization_raw"],
                "steering_raw": conditions["steering_raw"],
                "topk": file_meta["topk"],
                "steering_factor": file_meta["steering_factor"],
                "prompt_id": idx,
                "prompt_text": user_text,
                "model": "Qwen1.5-32B-Chat",
                "source_behavior": "verse",
                "base_behavior": "prose",
                "steering_changed_output": changed,
                "source_file": str(filepath),
            },
        )
        runs.append(run)
    return runs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--collection_name", default="verse_prose_qwen32b_steering")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    client = Docent()

    if not args.dry_run:
        collection_id = client.create_collection(
            name=args.collection_name,
            description=(
                "Verse/prose steering on Qwen1.5-32B-Chat. "
                "2x2: localization (longform/single_token) x steering (longform/single_token). "
                "topk ∈ {1,2,4,5,6,8,10}, steering_factor ∈ {0.01,0.03,0.05,0.07,0.09,0.1,0.5,1.0}."
            ),
        )
        print(f"Created collection: {collection_id}")
        print(f"View at: https://docent.transluce.org/dashboard/{collection_id}\n")

    results_root = Path(args.results_dir)
    total_runs = 0
    total_files = 0

    for from_seg, steer_seg in CONDITION_DIRS:
        eval_dir = results_root / from_seg / "atp" / "verse-long_eval" / steer_seg / "eval"
        if not eval_dir.exists():
            print(f"\n[SKIP] Not found: {eval_dir}")
            continue

        json_files = sorted(eval_dir.glob("*.json"))
        print(f"\n{'='*60}")
        print(f"  {from_seg} / {steer_seg}  ({len(json_files)} files)")

        batch = []
        for fpath in json_files:
            conditions = parse_condition_from_path(fpath)
            file_meta = parse_filename(fpath.name)
            if conditions is None or file_meta is None:
                continue
            runs = build_agent_runs(fpath, conditions, file_meta)
            batch.extend(runs)
            total_runs += len(runs)
            total_files += 1

        if batch and not args.dry_run:
            client.add_agent_runs(collection_id, batch)
            print(f"  Uploaded {len(batch)} runs for this condition")

    print(f"\n{'='*60}")
    print(f"{'DRY RUN - ' if args.dry_run else ''}Done.")
    print(f"  Files processed : {total_files}")
    print(f"  Runs converted  : {total_runs}")
    if not args.dry_run:
        print(f"  Collection ID   : {collection_id}")
        print(f"  View at         : https://docent.transluce.org/dashboard/{collection_id}")


if __name__ == "__main__":
    main()
