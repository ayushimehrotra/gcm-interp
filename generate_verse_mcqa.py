"""
Rewrite the verse-single prompts as multiple-choice (MCQA) prompts.

Each question is turned into a 4-option MCQA prompt of the form:

    Question: {question}
    Which of the following responses correctly answer the question in prose?
    (A) ...
    (B) ...
    (C) ...
    (D) ...
    Answer: (

The four options are (shuffled, see below):
  - a prose response that DOES respond to the question   <- the correct answer
  - a verse response that DOES respond to the question
  - a prose response that does NOT respond (off-topic)
  - a verse response that does NOT respond (off-topic)

Design decisions (chosen by the dataset author):
  * Option text is GENERATED with the model (via vLLM, batched). For each question
    we generate a concise prose response and a concise verse response. The two
    "does not respond" distractors are model-generated responses to a *different*
    question in the same pool (deterministically chosen per id), so they are
    fluent and on-format but off-topic.
  * Options are shuffled per question (seeded by id, independent of the file) so the
    correct letter is not always the same, and the SAME A/B/C/D layout is reused
    across every file for a given id. The prose-* and verse-single-* prompts are then
    byte-identical except for the medium word ('prose'/'verse') in the stem.
  * Stem + assistant answer label, with P = prose-responds letter, V = verse-responds:
        prose-desired-all.jsonl           stem 'prose'  -> P
        prose-undesired-all.jsonl         stem 'prose'  -> V
        verse-single-desired-all.jsonl    stem 'verse'  -> V
        verse-single-undesired-all.jsonl  stem 'verse'  -> P
        prose-test.jsonl                  stem 'prose'  -> P (correct answer forced)

Generated responses are cached to <folder>/mcqa_response_cache.json so reruns are
cheap and the generations are inspectable. Original files are backed up to
<folder>/_pre_mcqa_backup/ before being overwritten (disable with --no-backup).

Usage:
    python generate_verse_mcqa.py --model_id Qwen/Qwen1.5-14B-Chat
    HF_TOKEN=... python generate_verse_mcqa.py --model_id Qwen/Qwen1.5-14B-Chat
    python generate_verse_mcqa.py --model_id Qwen/Qwen1.5-14B-Chat --dry-run   # structure only
"""

import argparse
import json
import os
import random
import re
import shutil
import sys

DATA_DIR = "./data"

MODEL_CONFIGS = {
    "Qwen/Qwen1.5-14B-Chat":             {"name": "Qwen1.5-14B-Chat"},
    "Qwen/Qwen1.5-32B-Chat":             {"name": "Qwen1.5-32B-Chat"},
    "upstage/SOLAR-10.7B-Instruct-v1.0": {"name": "SOLAR-10.7B-Instruct-v1.0"},
    "allenai/OLMo-2-1124-13B-DPO":       {"name": "OLMo-2-1124-13B-DPO"},
}

# The subfolder whose prompts we rewrite, and the seed offset for per-id shuffles.
FOLDER = "verse-single"
SHUFFLE_SEED = 42

# Text glue for the MCQA prompt. Edit these to change the prompt wording.
#   Question: {question}
#   {MCQA_STEM}
#   (A) ... (B) ... (C) ... (D) ...
#   Answer: (
QUESTION_PREFIX = "Question: "
MCQA_STEM = "Which of the following responses is written in {medium} and correctly answers the question?"
ANSWER_CUE = "Answer: ("
LETTERS = ["A", "B", "C", "D"]


# ---------------------------------------------------------------------------
# Reading the existing files
# ---------------------------------------------------------------------------

def role_of(filename):
    """Classify a file: 'test', 'undesired', or 'desired'. (check undesired first)"""
    if filename.endswith("-test.jsonl"):
        return "test"
    if "undesired" in filename:
        return "undesired"
    if "desired" in filename:
        return "desired"
    return None


def medium_of(filename):
    """The medium the file's stem asks for: 'verse' for verse-single-*, else 'prose'."""
    return "verse" if filename.startswith("verse-single") else "prose"


def extract_question(content):
    """Pull the bare question out of an existing verse/prose prompt."""
    m = re.search(r"\n\n(.+?)\n\nResponse:", content, re.DOTALL)
    return m.group(1).strip() if m else None


def read_file(fpath):
    """Return list of (id, question) for a jsonl prompt file, preserving order."""
    rows = []
    with open(fpath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            q = extract_question(obj["prompt"][0]["content"])
            if q is None:
                raise ValueError(f"Could not extract a question from: {obj['prompt'][0]['content']!r}")
            rows.append((obj["id"], q))
    return rows


# ---------------------------------------------------------------------------
# Response cache  (key: "<framing>\t<question>")
# ---------------------------------------------------------------------------

def cache_key(question, framing):
    return f"{framing}\t{question}"


def load_cache(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def save_cache(path, cache):
    with open(path, "w") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Generation with vLLM (batched, greedy)
# ---------------------------------------------------------------------------

def generate_with_vllm(model_id, pairs, max_new_tokens, gpu_mem_util, hf_token):
    """
    pairs: list of (question, framing). Returns dict cache_key -> response text.
    One batched vLLM call (greedy decoding) for all pairs.
    """
    from vllm import LLM, SamplingParams

    if hf_token:
        os.environ.setdefault("HF_TOKEN", hf_token)
        os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", hf_token)

    print(f"Loading {model_id} in vLLM (bf16)...")
    llm = LLM(
        model=model_id,
        dtype="bfloat16",
        trust_remote_code=True,
        gpu_memory_utilization=gpu_mem_util,
    )
    sampling = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)

    conversations = [
        [{"role": "user", "content": f"Respond in {framing} and concisely. {question}"}]
        for (question, framing) in pairs
    ]
    print(f"Generating {len(conversations)} responses...")
    outputs = llm.chat(conversations, sampling)

    result = {}
    for (question, framing), out in zip(pairs, outputs):
        result[cache_key(question, framing)] = out.outputs[0].text.strip()
    return result


# ---------------------------------------------------------------------------
# Building one MCQA prompt
# ---------------------------------------------------------------------------

def pick_offtopic_id(qid, pool_ids):
    """Deterministically pick a different id from the pool to source distractors."""
    rng = random.Random(SHUFFLE_SEED + qid)
    others = [i for i in pool_ids if i != qid]
    return rng.choice(others)


def build_mcqa(qid, question, responses, medium):
    """
    Build the MCQA user prompt and return (prompt_str, prose_letter, verse_letter).

    `responses` maps each of the four roles to its text:
        'prose_on', 'verse_on', 'prose_off', 'verse_off'
    The four options are shuffled deterministically by id (independent of `medium`),
    so for a given id every file shares the same A/B/C/D layout and the prompts differ
    only in the `medium` word ('prose'/'verse') in the stem.
    """
    options = [
        ("prose_on",  responses["prose_on"]),    # prose, responds to the question
        ("verse_on",  responses["verse_on"]),    # verse, responds to the question
        ("prose_off", responses["prose_off"]),   # prose, off-topic
        ("verse_off", responses["verse_off"]),   # verse, off-topic
    ]
    rng = random.Random(SHUFFLE_SEED + qid)
    rng.shuffle(options)

    lines = [QUESTION_PREFIX + question, MCQA_STEM.format(medium=medium)]
    prose_letter = verse_letter = None
    for letter, (tag, text) in zip(LETTERS, options):
        lines.append(f"({letter}) {text}")
        if tag == "prose_on":
            prose_letter = letter
        elif tag == "verse_on":
            verse_letter = letter
    prompt = "\n".join(lines) + "\n" + ANSWER_CUE
    return prompt, prose_letter, verse_letter


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Rewrite verse-single prompts as MCQA prompts")
    parser.add_argument("--model_id", required=True, choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--hf_token", default=os.environ.get("HF_TOKEN", ""))
    parser.add_argument("--max_new_tokens", type=int, default=128,
                        help="Cap on each generated option's length (kept short for MCQA).")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    parser.add_argument("--no-backup", action="store_true",
                        help="Overwrite files in place without saving a backup copy.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Build prompts from cached responses only; do not load the model "
                             "or write files. Prints one example per file.")
    args = parser.parse_args()

    random.seed(SHUFFLE_SEED)

    model_name = MODEL_CONFIGS[args.model_id]["name"]
    folder = os.path.join(DATA_DIR, model_name, FOLDER)
    if not os.path.isdir(folder):
        sys.exit(f"Folder not found: {folder}")

    files = sorted(f for f in os.listdir(folder) if f.endswith(".jsonl"))
    files = [f for f in files if role_of(f) is not None]
    print(f"Folder: {folder}")
    print(f"Files to rewrite: {files}")

    # Read questions from the original prompts. If this script has already run and
    # overwritten the files, the un-rewritten originals live in the backup dir;
    # prefer those so re-runs stay idempotent (the rewritten MCQA prompts no longer
    # match the question-extraction regex).
    backup_dir = os.path.join(folder, "_pre_mcqa_backup")

    def source_path(fname):
        bak = os.path.join(backup_dir, fname)
        return bak if os.path.exists(bak) else os.path.join(folder, fname)

    # Read every file once; collect each file's (id, question) rows and a per-file id pool.
    file_rows = {f: read_file(source_path(f)) for f in files}
    # Pool of ids each file's distractors may draw from = the ids within that same file.
    file_pools = {f: [qid for qid, _ in rows] for f, rows in file_rows.items()}
    # id -> question (ids are consistent across the all-files, disjoint for test).
    id_to_question = {}
    for rows in file_rows.values():
        for qid, q in rows:
            id_to_question.setdefault(qid, q)

    cache_path = os.path.join(folder, "mcqa_response_cache.json")
    cache = load_cache(cache_path)

    # We need a prose AND verse response for every unique question (every question
    # serves both as a main question and as a potential off-topic distractor).
    needed = []
    for q in dict.fromkeys(id_to_question.values()):
        for framing in ("prose", "verse"):
            if cache_key(q, framing) not in cache:
                needed.append((q, framing))

    print(f"Unique questions: {len(set(id_to_question.values()))}")
    print(f"Responses needed (not cached): {len(needed)}")

    if needed and not args.dry_run:
        fresh = generate_with_vllm(
            args.model_id, needed, args.max_new_tokens,
            args.gpu_memory_utilization, args.hf_token,
        )
        cache.update(fresh)
        save_cache(cache_path, cache)
        print(f"Cache now holds {len(cache)} responses -> {cache_path}")
    elif needed and args.dry_run:
        print("(dry-run: missing responses will appear as placeholders)")

    def responses_for(qid, pool_ids):
        question = id_to_question[qid]
        off_question = id_to_question[pick_offtopic_id(qid, pool_ids)]
        def g(q, fr):
            return cache.get(cache_key(q, fr), f"<{fr} response for: {q}>")
        return {
            "prose_on":  g(question, "prose"),
            "verse_on":  g(question, "verse"),
            "prose_off": g(off_question, "prose"),
            "verse_off": g(off_question, "verse"),
        }

    # Rewrite each file.
    for fname in files:
        role = role_of(fname)
        medium = medium_of(fname)
        rows = file_rows[fname]
        pool = file_pools[fname]
        out_lines = []
        example = None
        for qid, question in rows:
            resp = responses_for(qid, pool)
            prompt, prose_letter, verse_letter = build_mcqa(qid, question, resp, medium)

            # Correct = the option matching this file's medium; the undesired files
            # carry the *other* medium's letter (i.e. prose-undesired -> verse letter,
            # verse-single-undesired -> prose letter).
            correct = prose_letter if medium == "prose" else verse_letter
            other = verse_letter if medium == "prose" else prose_letter

            messages = [{"role": "user", "content": prompt}]
            if role == "undesired":
                messages.append({"role": "assistant", "content": other})
            else:
                # desired and test both force the correct answer (the option matching
                # the file's medium). The eval path strips the assistant message anyway
                # (only_q=True), so this just supplies ground-truth labels for test.
                messages.append({"role": "assistant", "content": correct})

            entry = {"id": qid, "prompt": messages}
            out_lines.append(json.dumps(entry, ensure_ascii=False))
            if example is None:
                example = entry

        if args.dry_run:
            print(f"\n===== {fname} ({role}, {len(out_lines)} rows) — example =====")
            print(json.dumps(example, ensure_ascii=False, indent=2)[:2500])
            continue

        fpath = os.path.join(folder, fname)
        if not args.no_backup:
            os.makedirs(backup_dir, exist_ok=True)
            if not os.path.exists(os.path.join(backup_dir, fname)):
                shutil.copy2(fpath, os.path.join(backup_dir, fname))
        with open(fpath, "w") as f:
            f.write("\n".join(out_lines) + "\n")
        print(f"Wrote {fname}: {len(out_lines)} rows ({role}).")

    print("\nDone.")


if __name__ == "__main__":
    main()
