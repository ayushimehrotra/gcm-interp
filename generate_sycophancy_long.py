"""
Generate sycophancy-long responses for a new model using HuggingFace.

For each passage (train IDs), two responses are generated and cross-pollinated:
  - response_T: model response to "terrible"-framing prompt
  - response_O: model response to "okay"-framing prompt

Cross-pollination into four files:
  sycophancy-long-desired:     terrible_user_msg + response_T  (sycophantic: agrees it's terrible)
  sycophancy-long-undesired:   terrible_user_msg + response_O  (contrasting)
  non-sycophantic-desired:     okay_user_msg    + response_O  (honest: evaluates objectively)
  non-sycophantic-undesired:   okay_user_msg    + response_T  (contrasting)

Pairing: syco_desired.response == nonsyco_undesired.response (both response_T)
         syco_undesired.response == nonsyco_desired.response (both response_O)

For test IDs:
  non-sycophantic-test:        okay_user_msg    + response_O

Source prompts are read from an existing model's sycophancy-long files (user messages
are identical across all models).

Run from the gcm-interp repo root:
  python generate_sycophancy_long.py --model_id meta-llama/Llama-2-13b-chat-hf
  python generate_sycophancy_long.py --model_id lmsys/vicuna-13b-v1.5
"""

import argparse
import json
import os
import sys

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

sys.path.insert(0, os.path.dirname(__file__))
from eval.setup import set_seed

DATA_DIR = "./data"
SOURCE_MODEL = "OLMo-2-1124-13B-DPO"

MODEL_CONFIGS = {
    "meta-llama/Llama-2-13b-chat-hf": {"name": "Llama-2-13b-chat-hf"},
    "lmsys/vicuna-13b-v1.5": {
        "name": "vicuna-13b-v1.5",
        "system_prompt": (
            "You are a literary critic. Respond as a literary critic, not as anything else."
        ),
    },
}


def load_model_and_tokenizer(model_id, hf_token):
    print(f"Loading tokenizer for {model_id}...")
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


def generate_response(model, tokenizer, user_message, max_new_tokens=256, system_prompt=None):
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_message})
    try:
        prompt = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
    except ValueError:
        # Fallback for tokenizers without a chat template (e.g. Vicuna v1.5).
        # Uses the Vicuna USER/ASSISTANT conversation format.
        if system_prompt:
            prompt = f"{system_prompt}\n\nUSER: {user_message}\nASSISTANT:"
        else:
            prompt = f"USER: {user_message}\nASSISTANT:"
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


def load_user_messages(jsonl_path):
    """Return {id: user_message} from a sycophancy-long JSONL file."""
    messages = {}
    with open(jsonl_path) as f:
        for line in f:
            obj = json.loads(line)
            messages[obj["id"]] = obj["prompt"][0]["content"]
    return messages


def load_done_ids(jsonl_path):
    """Return set of IDs already written to a file (for resumability)."""
    if not os.path.exists(jsonl_path):
        return set()
    done = set()
    with open(jsonl_path) as f:
        for line in f:
            obj = json.loads(line)
            done.add(obj["id"])
    return done


def append_entry(fpath, entry_id, user_message, response, system_prompt=None):
    prompt = []
    if system_prompt:
        prompt.append({"role": "system", "content": system_prompt})
    prompt.append({"role": "user",      "content": user_message})
    prompt.append({"role": "assistant", "content": response})
    entry = {"id": entry_id, "prompt": prompt}
    with open(fpath, "a") as f:
        f.write(json.dumps(entry) + "\n")


def get_line_count(fpath):
    if not os.path.exists(fpath):
        return 0
    with open(fpath) as f:
        return sum(1 for _ in f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", required=True, choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--hf_token", default=os.environ.get("HF_TOKEN", ""))
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument(
        "--source_model",
        default=SOURCE_MODEL,
        help="Existing model folder to read source prompts from",
    )
    args = parser.parse_args()

    set_seed(42)

    model_name = MODEL_CONFIGS[args.model_id]["name"]
    model_dir = os.path.join(DATA_DIR, model_name, "sycophancy-long")
    source_dir = os.path.join(DATA_DIR, args.source_model, "sycophancy-long")

    os.makedirs(model_dir, exist_ok=True)

    files = {
        "syco_desired":    os.path.join(model_dir, "sycophancy-long-desired-all.jsonl"),
        "syco_undesired":  os.path.join(model_dir, "sycophancy-long-undesired-all.jsonl"),
        "nonsyco_desired": os.path.join(model_dir, "non-sycophantic-desired-all.jsonl"),
        "nonsyco_undesired": os.path.join(model_dir, "non-sycophantic-undesired-all.jsonl"),
        "test":            os.path.join(model_dir, "non-sycophantic-test.jsonl"),
    }

    # Load source user messages (prompt text is identical across models)
    terrible_msgs = load_user_messages(
        os.path.join(source_dir, "sycophancy-long-desired-all.jsonl")
    )
    okay_msgs = load_user_messages(
        os.path.join(source_dir, "non-sycophantic-desired-all.jsonl")
    )
    test_msgs = load_user_messages(
        os.path.join(source_dir, "non-sycophantic-test.jsonl")
    )

    # Ordered IDs
    train_ids = list(terrible_msgs.keys())
    test_ids  = list(test_msgs.keys())

    # Resume support: skip IDs already written to syco_desired
    done_train = load_done_ids(files["syco_desired"])
    done_test  = load_done_ids(files["test"])

    todo_train = [i for i in train_ids if i not in done_train]
    todo_test  = [i for i in test_ids  if i not in done_test]

    print(f"Model:             {model_name}")
    print(f"Source prompts:    {args.source_model}")
    print(f"Train IDs total:   {len(train_ids)},  remaining: {len(todo_train)}")
    print(f"Test  IDs total:   {len(test_ids)},   remaining: {len(todo_test)}")

    if not todo_train and not todo_test:
        print("Nothing to generate — all files are up to date.")
        return

    model, tokenizer = load_model_and_tokenizer(args.model_id, args.hf_token)
    system_prompt = MODEL_CONFIGS[args.model_id].get("system_prompt")

    # --- Train set ---
    for i, entry_id in enumerate(todo_train):
        print(f"\n[Train {i+1}/{len(todo_train)}] ID={entry_id}")

        terrible_msg = terrible_msgs[entry_id]
        okay_msg     = okay_msgs[entry_id]

        print("  Generating response to 'terrible' framing...")
        response_T = generate_response(model, tokenizer, terrible_msg, args.max_new_tokens, system_prompt)
        print(f"  response_T: {response_T[:80]}...")

        print("  Generating response to 'okay' framing...")
        response_O = generate_response(model, tokenizer, okay_msg, args.max_new_tokens, system_prompt)
        print(f"  response_O: {response_O[:80]}...")

        # Cross-pollinate: response_T shared by syco_desired + nonsyco_undesired
        #                  response_O shared by syco_undesired + nonsyco_desired
        append_entry(files["syco_desired"],      entry_id, terrible_msg, response_T, system_prompt)
        append_entry(files["syco_undesired"],    entry_id, terrible_msg, response_O, system_prompt)
        append_entry(files["nonsyco_desired"],   entry_id, okay_msg,     response_O, system_prompt)
        append_entry(files["nonsyco_undesired"], entry_id, okay_msg,     response_T, system_prompt)

    # --- Test set ---
    for i, entry_id in enumerate(todo_test):
        print(f"\n[Test {i+1}/{len(todo_test)}] ID={entry_id}")

        okay_msg = test_msgs[entry_id]
        print("  Generating response to 'okay' framing (test)...")
        response_O = generate_response(model, tokenizer, okay_msg, args.max_new_tokens, system_prompt)
        print(f"  response_O: {response_O[:80]}...")

        append_entry(files["test"], entry_id, okay_msg, response_O)

    print("\nFinal counts:")
    for k, v in files.items():
        print(f"  {k}: {get_line_count(v)} entries")
    print("Done.")


if __name__ == "__main__":
    main()
