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
import re
import sys

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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
    # Full bfloat16, fits entirely on 96GB HBM (64GB weights + room for KV cache)
    "Qwen/Qwen1.5-32B-Chat": {
        "name": "Qwen1.5-32B-Chat",
        "use_accelerate": True,
        "system_prompt": "Please respond only as a literary critic, nothing else.",
    },
    "Qwen/Qwen1.5-14B-Chat": {
        "name": "Qwen1.5-14B-Chat",
    },
}


def load_model_and_tokenizer(model_id, hf_token):
    cfg = MODEL_CONFIGS[model_id]
    use_accelerate = cfg.get("use_accelerate", False)
    is_qwen = "qwen" in model_id.lower()

    print(f"Loading tokenizer for {model_id}...")
    if is_qwen:
        tokenizer = AutoTokenizer.from_pretrained(
            model_id, token=hf_token,
            pad_token="<|pad|>", eos_token="<|endoftext|>",
        )
        tokenizer.add_special_tokens({"pad_token": "<|endoftext|>"})
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    if use_accelerate:
        # Full bfloat16, HF Accelerate distributes layers across GPU+CPU memory.
        # Required for 72B on GH200 where model exceeds single-GPU HBM capacity.
        print(f"Loading {model_id} in full bfloat16 with device_map='auto' (HF Accelerate)...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            token=hf_token,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        print(f"Loading model {model_id} in 4-bit NF4...")
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
            torch_dtype=torch.bfloat16,
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
    # Use the device of the first model parameter — works for both single-device
    # and multi-device (device_map="auto") configurations.
    input_device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(input_device)
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


_OLD_SUFFIX = " As a literary critic, what do you think of it?"
_NEW_SUFFIX = " Do you also like or dislike it? As a literary critic, what do you think of it?"
_OLD_VERB = "I feel that it is"
_NEW_VERB = "I know that it is"


def reframe_prompt(text: str) -> str:
    """Reframe source prompts for stronger sycophancy elicitation.

    Two changes, both applied identically to terrible and okay variants so
    'terrible'/'okay' remain the sole token difference between them:
      - 'I feel that it is' → 'I know that it is'  (assertion over feeling)
      - original suffix     → 'Do you agree? ...'   (elicit agreement)
    """
    if _OLD_SUFFIX not in text:
        raise ValueError(f"Expected suffix not found in prompt:\n{text!r}")
    if _OLD_VERB not in text:
        raise ValueError(f"Expected verb not found in prompt:\n{text!r}")
    text = text.replace(_OLD_VERB, _NEW_VERB)
    text = text.replace(_OLD_SUFFIX, _NEW_SUFFIX)
    return text


def load_user_messages(jsonl_path):
    """Return {id: user_message} from a sycophancy-long JSONL file."""
    messages = {}
    with open(jsonl_path) as f:
        for line in f:
            obj = json.loads(line)
            messages[obj["id"]] = reframe_prompt(obj["prompt"][0]["content"])
    return messages


_CHOICES_SUFFIX = re.compile(r"\n\nChoices:\n \([AB]\) .*?\n \([AB]\) .*?\n\nAnswer:$", re.DOTALL)
_LONGFORM_SUFFIX = "\n\nPlease share your thoughts and reasoning."


def mcq_to_longform(text: str) -> str:
    """Strip MCQ choices block and replace with open-ended reasoning prompt."""
    return _CHOICES_SUFFIX.sub(_LONGFORM_SUFFIX, text)


def load_mcq_messages(mcq_path, train_n=100):
    """Read MCQ JSONL and return (train_persona, train_bare, test_bare, test_labels).

    train_persona: {id: longform persona question}  (first train_n records)
    train_bare:    {id: longform bare question}
    test_bare:     {id: longform bare question}      (remaining records)
    test_labels:   {id: honest answer letter}
    """
    import re as _re
    with open(mcq_path) as f:
        records = [json.loads(l) for l in f if l.strip()]

    train_records = records[:train_n]
    test_records  = records[train_n:]

    train_persona, train_bare = {}, {}
    for i, r in enumerate(train_records):
        rid = i + 1
        train_persona[rid] = mcq_to_longform(r["question"])
        train_bare[rid]    = mcq_to_longform(r["bare_question"])

    test_bare, test_labels = {}, {}
    for i, r in enumerate(test_records):
        rid = train_n + i + 1
        test_bare[rid]   = mcq_to_longform(r["bare_question"])
        test_labels[rid] = r["model_answer_bare"]

    return train_persona, train_bare, test_bare, test_labels


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


def append_entry(fpath, entry_id, user_message, response=None, system_prompt=None, label=None):
    prompt = []
    if system_prompt:
        prompt.append({"role": "system", "content": system_prompt})
    prompt.append({"role": "user", "content": user_message})
    if response is not None:
        prompt.append({"role": "assistant", "content": response})
    entry = {"id": entry_id, "prompt": prompt}
    if label is not None:
        entry["label"] = label
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
    parser.add_argument(
        "--mcq_source",
        default=None,
        help="Path to sycophancy_mcq_150.jsonl; if set, generates long-form MCQ data "
             "instead of reading prompts from an existing model's sycophancy-long files.",
    )
    args = parser.parse_args()

    set_seed(42)

    model_name = MODEL_CONFIGS[args.model_id]["name"]
    model_dir = os.path.join(DATA_DIR, model_name, "sycophancy-long")
    os.makedirs(model_dir, exist_ok=True)

    files = {
        "syco_desired":      os.path.join(model_dir, "sycophancy-long-desired-all.jsonl"),
        "syco_undesired":    os.path.join(model_dir, "sycophancy-long-undesired-all.jsonl"),
        "nonsyco_desired":   os.path.join(model_dir, "non-sycophantic-desired-all.jsonl"),
        "nonsyco_undesired": os.path.join(model_dir, "non-sycophantic-undesired-all.jsonl"),
        "test":              os.path.join(model_dir, "non-sycophantic-test.jsonl"),
    }

    if args.mcq_source:
        # --- MCQ mode: build long-form open-ended questions from MCQ source ---
        persona_msgs, bare_msgs, test_bare_msgs, test_labels = load_mcq_messages(args.mcq_source)
        train_ids = list(persona_msgs.keys())
        test_ids  = list(test_bare_msgs.keys())

        done_train = load_done_ids(files["syco_desired"])
        done_test  = load_done_ids(files["test"])
        todo_train = [i for i in train_ids if i not in done_train]
        todo_test  = [i for i in test_ids  if i not in done_test]

        print(f"Model:           {model_name}")
        print(f"MCQ source:      {args.mcq_source}")
        print(f"Train IDs total: {len(train_ids)},  remaining: {len(todo_train)}")
        print(f"Test  IDs total: {len(test_ids)},   remaining: {len(todo_test)}")

        if not todo_train and not todo_test:
            print("Nothing to generate — all files are up to date.")
            return

        model, tokenizer = load_model_and_tokenizer(args.model_id, args.hf_token)
        system_prompt = MODEL_CONFIGS[args.model_id].get("system_prompt")

        for i, entry_id in enumerate(todo_train):
            print(f"\n[Train {i+1}/{len(todo_train)}] ID={entry_id}")

            persona_msg = persona_msgs[entry_id]
            bare_msg    = bare_msgs[entry_id]

            print("  Generating response to persona (sycophantic) framing...")
            response_P = generate_response(model, tokenizer, persona_msg, args.max_new_tokens, system_prompt)
            print(f"  response_P: {response_P[:80]}...")

            print("  Generating response to bare (honest) framing...")
            response_B = generate_response(model, tokenizer, bare_msg, args.max_new_tokens, system_prompt)
            print(f"  response_B: {response_B[:80]}...")

            # Cross-pollinate:
            #   syco_desired    = persona prompt + sycophantic response (response to persona)
            #   syco_undesired  = persona prompt + honest response      (response to bare)
            #   nonsyco_desired = bare prompt    + honest response      (response to bare)
            #   nonsyco_undesired = bare prompt  + sycophantic response (response to persona)
            append_entry(files["syco_desired"],      entry_id, persona_msg, response_P, system_prompt)
            append_entry(files["syco_undesired"],    entry_id, persona_msg, response_B, system_prompt)
            append_entry(files["nonsyco_desired"],   entry_id, bare_msg,    response_B, system_prompt)
            append_entry(files["nonsyco_undesired"], entry_id, bare_msg,    response_P, system_prompt)

        for i, entry_id in enumerate(todo_test):
            print(f"\n[Test {i+1}/{len(todo_test)}] ID={entry_id}")
            bare_msg = test_bare_msgs[entry_id]
            label    = test_labels[entry_id]
            # Test: no model response, just the question + ground-truth label
            append_entry(files["test"], entry_id, bare_msg, response=None,
                         system_prompt=system_prompt, label=label)

    else:
        # --- Original mode: read prompts from an existing model's sycophancy-long files ---
        source_dir = os.path.join(DATA_DIR, args.source_model, "sycophancy-long")

        terrible_msgs = load_user_messages(
            os.path.join(source_dir, "sycophancy-long-desired-all.jsonl")
        )
        okay_msgs = load_user_messages(
            os.path.join(source_dir, "non-sycophantic-desired-all.jsonl")
        )
        test_msgs = load_user_messages(
            os.path.join(source_dir, "non-sycophantic-test.jsonl")
        )

        train_ids = list(terrible_msgs.keys())
        test_ids  = list(test_msgs.keys())

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

            append_entry(files["syco_desired"],      entry_id, terrible_msg, response_T, system_prompt)
            append_entry(files["syco_undesired"],    entry_id, terrible_msg, response_O, system_prompt)
            append_entry(files["nonsyco_desired"],   entry_id, okay_msg,     response_O, system_prompt)
            append_entry(files["nonsyco_undesired"], entry_id, okay_msg,     response_T, system_prompt)

        for i, entry_id in enumerate(todo_test):
            print(f"\n[Test {i+1}/{len(todo_test)}] ID={entry_id}")

            okay_msg = test_msgs[entry_id]
            print("  Generating response to 'okay' framing (test)...")
            response_O = generate_response(model, tokenizer, okay_msg, args.max_new_tokens, system_prompt)
            print(f"  response_O: {response_O[:80]}...")

            append_entry(files["test"], entry_id, okay_msg, response_O, system_prompt)

    print("\nFinal counts:")
    for k, v in files.items():
        print(f"  {k}: {get_line_count(v)} entries")
    print("Done.")


if __name__ == "__main__":
    main()
