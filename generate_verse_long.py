"""
Generate verse/prose responses for new questions and append them to verse-long files.

For each question, generates:
  - A verse response  → appended to verse-long-desired-all.jsonl and prose-undesired-all.jsonl
  - A prose response  → appended to prose-desired-all.jsonl and verse-long-undesired-all.jsonl

Run once per model after adding new questions to the verse-single files:

  python generate_verse_long.py --model_id Qwen/Qwen1.5-14B-Chat
  python generate_verse_long.py --model_id upstage/SOLAR-10.7B-Instruct-v1.0
  python generate_verse_long.py --model_id allenai/OLMo-2-1124-13B-DPO

The script is resumable: questions already present in the verse-long files are skipped.
"""

import argparse
import json
import os
import re
import sys

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

sys.path.insert(0, os.path.dirname(__file__))
from eval.setup import set_seed

DATA_DIR = "./data"

MODEL_CONFIGS = {
    "Qwen/Qwen1.5-14B-Chat":             {"name": "Qwen1.5-14B-Chat"},
    "upstage/SOLAR-10.7B-Instruct-v1.0": {"name": "SOLAR-10.7B-Instruct-v1.0"},
    "allenai/OLMo-2-1124-13B-DPO":       {"name": "OLMo-2-1124-13B-DPO"},
}

LONG_PREFIXES = ("Respond in verse and concisely. ", "Respond in prose and concisely. ")


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(model_id, hf_token):
    print(f"Loading tokenizer for {model_id}...")
    if "qwen" in model_id.lower():
        tokenizer = AutoTokenizer.from_pretrained(
            model_id, token=hf_token, pad_token="<|pad|>", eos_token="<|endoftext|>"
        )
        tokenizer.add_special_tokens({"pad_token": "<|endoftext|>"})
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    print(f"Loading model {model_id} in 4-bit...")
    nf4 = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        token=hf_token,
        quantization_config=nf4,
        device_map="cuda:0",
        dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_response(model, tokenizer, question, framing, max_new_tokens=256):
    """Generate a response for a question with the given framing ('verse' or 'prose')."""
    messages = [{"role": "user", "content": f"Respond in {framing} and concisely. {question}"}]
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------------

def load_existing_questions(fpath):
    """Return the set of pure questions already present in a verse-long file."""
    questions = set()
    if not os.path.exists(fpath):
        return questions
    with open(fpath) as f:
        for line in f:
            obj = json.loads(line)
            content = obj["prompt"][0]["content"]
            for prefix in LONG_PREFIXES:
                if content.startswith(prefix):
                    questions.add(content[len(prefix):])
                    break
    return questions


def get_line_count(fpath):
    if not os.path.exists(fpath):
        return 0
    with open(fpath) as f:
        return sum(1 for _ in f)


def append_entry(fpath, question, framing, response, entry_id):
    entry = {
        "id": entry_id,
        "prompt": [
            {"role": "user",      "content": f"Respond in {framing} and concisely. {question}"},
            {"role": "assistant", "content": response},
        ],
    }
    with open(fpath, "a") as f:
        f.write(json.dumps(entry) + "\n")


# ---------------------------------------------------------------------------
# Questions source: read from verse-single-desired-all.jsonl
# ---------------------------------------------------------------------------

def get_verse_single_questions(model_name):
    """Return the ordered list of pure questions from the verse-single all-file."""
    fpath = os.path.join(DATA_DIR, model_name, "verse-single", "verse-single-desired-all.jsonl")
    questions = []
    with open(fpath) as f:
        for line in f:
            obj = json.loads(line)
            content = obj["prompt"][0]["content"]
            m = re.search(r"\n\n(.+?)\n\nResponse:", content, re.DOTALL)
            if m:
                questions.append(m.group(1).strip())
    return questions


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate verse-long responses for new questions")
    parser.add_argument("--model_id", required=True, choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--hf_token", default=os.environ.get("HF_TOKEN", ""))
    parser.add_argument("--max_new_tokens", type=int, default=256)
    args = parser.parse_args()

    # Seed everything before touching CUDA
    set_seed(42)

    model_name = MODEL_CONFIGS[args.model_id]["name"]
    model_dir = os.path.join(DATA_DIR, model_name, "verse-long")

    files = {
        "verse_desired":   os.path.join(model_dir, "verse-long-desired-all.jsonl"),
        "verse_undesired": os.path.join(model_dir, "verse-long-undesired-all.jsonl"),
        "prose_desired":   os.path.join(model_dir, "prose-desired-all.jsonl"),
        "prose_undesired": os.path.join(model_dir, "prose-undesired-all.jsonl"),
    }

    # Questions already present (for resume support)
    already_done = load_existing_questions(files["verse_desired"])

    # Questions that need responses = verse-single pool minus already done
    all_questions = get_verse_single_questions(model_name)
    todo = [q for q in all_questions if q not in already_done]

    print(f"Model:        {model_name}")
    print(f"Total questions in verse-single: {len(all_questions)}")
    print(f"Already in verse-long:           {len(already_done)}")
    print(f"To generate:                     {len(todo)}")

    if not todo:
        print("Nothing to generate. verse-long is already up to date.")
        return

    model, tokenizer = load_model_and_tokenizer(args.model_id, args.hf_token)

    for i, question in enumerate(todo):
        print(f"\n[{i+1}/{len(todo)}] {question}")

        print("  Generating verse response...")
        verse_resp = generate_response(model, tokenizer, question, "verse", args.max_new_tokens)
        print(f"  Verse: {verse_resp[:100]}...")

        print("  Generating prose response...")
        prose_resp = generate_response(model, tokenizer, question, "prose", args.max_new_tokens)
        print(f"  Prose: {prose_resp[:100]}...")

        next_id = get_line_count(files["verse_desired"])
        append_entry(files["verse_desired"],   question, "verse", verse_resp, next_id)
        append_entry(files["prose_undesired"], question, "prose", verse_resp, get_line_count(files["prose_undesired"]))
        append_entry(files["prose_desired"],   question, "prose", prose_resp, get_line_count(files["prose_desired"]))
        append_entry(files["verse_undesired"], question, "verse", prose_resp, get_line_count(files["verse_undesired"]))

    print("\nFinal counts:")
    for k, v in files.items():
        print(f"  {k}: {get_line_count(v)} entries")
    print("Done.")


if __name__ == "__main__":
    main()
