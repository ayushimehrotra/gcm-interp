"""
Build the verse-single and verse-long datasets for a model, end to end, in one script.

This replaces three formerly-separate scripts (clean_verse_dataset.py, generate_verse_mcqa.py,
generate_verse_long.py) which had grown an important inefficiency: verse-long used to
generate its own prose/verse responses from scratch via a slow, unbatched, 4-bit
transformers path -- even though the exact same responses (same prompt: "Respond in
{framing} and concisely. {question}") were already generated via a fast, batched vLLM
call while building verse-single, and sit cached in
data/<model>/verse-single/mcqa_response_cache.json. Regenerating wasted GPU time AND
risked producing responses that disagree with the ones already baked into the
verse-single MCQA distractors (different precision/max_tokens between a from-scratch
transformers run and the original vLLM run can change greedy output). This script always
reuses that cache, only falling back to (lazily-loaded) 4-bit transformers generation for
a question+framing pair that is genuinely not cached.

Three stages, each independently resumable/skippable if its output already exists:

  1. verse-single (build_verse_single): question generation, response generation, and
     iterative slot-filling/verification against the model's OWN MCQA accuracy (all via
     vLLM). Writes data/<model>/verse-single/{prose,verse-single}-{desired,undesired}-all.jsonl
     + prose-test.jsonl + mcqa_response_cache.json.
  2. verse-long (build_verse_long): reuses cached responses for the verse-single TRAIN
     questions (see above) to write data/<model>/verse-long/{verse-long,prose}-{desired,undesired}-all.jsonl.
  3. verse-long test file (build_verse_long_test): the 50 held-out verse-single TEST
     questions, stripped down to a bare "Respond in prose and concisely. {question}"
     user-only prompt (no MCQA wrapping, no response) -- these are the same 50 test
     prompts already generated for verse-single, just reformatted, per explicit
     instruction. Writes data/<model>/verse-long/prose-test.jsonl.

Usage:
    python generate_verse_data.py --model gemma-3-12b-it
    python generate_verse_data.py --model gemma-3-12b-it --only long,test   # skip stage 1
    python generate_verse_data.py --model Qwen1.5-14B-Chat --force single  # force rebuild
"""
import argparse
import json
import os
import random
import re
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_DIR = "data"

# model short-name -> (HF repo, gpu_memory_utilization for the vLLM stage). The 32B
# model needs a much larger memory fraction than the ~12-14B models to hold its weights
# + KV cache; gemma-3-12b-it's weights alone take ~23GiB (it's a multimodal-capable
# architecture even used text-only), so it needs more headroom than the ~0.45 that's
# enough for the other ~13-14B text-only models.
MODELS = {
    "Qwen1.5-14B-Chat":          ("Qwen/Qwen1.5-14B-Chat",             0.45),
    "OLMo-2-1124-13B-DPO":       ("allenai/OLMo-2-1124-13B-DPO",       0.45),
    "Qwen1.5-32B-Chat":          ("Qwen/Qwen1.5-32B-Chat",             0.80),
    "gemma-3-12b-it":            ("google/gemma-3-12b-it",             0.85),
    "SOLAR-10.7B-Instruct-v1.0": ("upstage/SOLAR-10.7B-Instruct-v1.0", 0.45),
    "phi-4":                     ("microsoft/phi-4",                   0.45),
    "Falcon3-10B-Instruct":      ("tiiuae/Falcon3-10B-Instruct",       0.45),
    "Llama-2-13b-chat-hf":       ("meta-llama/Llama-2-13b-chat-hf",    0.45),
}

N_TRAIN, N_TEST = 100, 50
TRAIN_IDS = list(range(0, N_TRAIN))            # ids 0..99
TEST_IDS = list(range(1000, 1000 + N_TEST))    # ids 1000..1049 (disjoint from train)

THEMES = [
    "time and change", "meaning and purpose in life", "love and human relationships", "death and mortality",
    "morality and ethics", "knowledge and truth", "society and community", "art and beauty",
    "emotion and feeling", "personal identity and the self", "freedom and choice", "happiness and contentment",
    "suffering and adversity", "nature and the universe", "technology and progress", "language and communication",
    "memory and the past", "hope and the future", "fear and courage", "success and ambition",
    "solitude and human connection", "wisdom and learning", "faith and doubt", "justice and fairness",
    "creativity and imagination", "power and responsibility", "trust and betrayal", "growth and aging",
    "dreams and aspiration", "work and leisure", "silence and sound", "tradition and innovation",
]

# ---------------------------------------------------------------------------
# MCQA prompt building (was generate_verse_mcqa.py) -- kept byte-identical so that
# judge-evals/compute_single_accuracies.py's independent reimplementation of this
# shuffle stays in sync (see MCQA_SHUFFLE_SEED there).
# ---------------------------------------------------------------------------
SHUFFLE_SEED = 42
QUESTION_PREFIX = "Question: "
MCQA_STEM = "Which of the following responses is written in {medium} and correctly answers the question? Please respond with only \"A\", \"B\", \"C\", or \"D\"."
ANSWER_CUE = "Answer: ("
LETTERS = ["A", "B", "C", "D"]

LONG_PREFIXES = ("Respond in verse and concisely. ", "Respond in prose and concisely. ")

LETTER_RE = re.compile(r"^\s*\(?\s*([ABCD])\b")   # bare leading letter, e.g. "B) ..." or "(B"
OPTION_RE = re.compile(r"\(\s*([ABCD])\s*\)")      # option marker "(B)" anywhere (tolerates a preface)


def cache_key(question, framing):
    return f"{framing}\t{question}"


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
        ("prose_on",  responses["prose_on"]),
        ("verse_on",  responses["verse_on"]),
        ("prose_off", responses["prose_off"]),
        ("verse_off", responses["verse_off"]),
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


def parse_letter(r):
    """Extract the chosen option letter (prefer a leading letter, fall back to '(X)')."""
    r = (r or "").upper()
    m = LETTER_RE.match(r)
    if m:
        return m.group(1)
    m = OPTION_RE.search(r)
    return m.group(1) if m else None


def assistant(role, medium, pL, vL):
    correct = pL if medium == "prose" else vL
    other = vL if medium == "prose" else pL
    return other if role == "undesired" else correct


# ---------------------------------------------------------------------------
# Stage 1: verse-single (was clean_verse_dataset.py)
# ---------------------------------------------------------------------------
VERSE_SINGLE_FILES = [
    "prose-desired-all.jsonl", "prose-undesired-all.jsonl",
    "verse-single-desired-all.jsonl", "verse-single-undesired-all.jsonl",
    "prose-test.jsonl",
]


def verse_single_dir(model_dir):
    return os.path.join(DATA_DIR, model_dir, "verse-single")


def verse_single_done(model_dir):
    d = verse_single_dir(model_dir)
    return all(os.path.exists(os.path.join(d, f)) for f in VERSE_SINGLE_FILES)


def load_existing_verse_single_questions(model_dir):
    qs = set()
    d = verse_single_dir(model_dir)
    if not os.path.isdir(d):
        return qs
    for f in os.listdir(d):
        if f.endswith(".jsonl"):
            for line in open(os.path.join(d, f)):
                c = json.loads(line)["prompt"][0]["content"]
                m = re.search(r"Question:\s*(.*?)\n", c)
                if m:
                    qs.add(m.group(1).strip().lower())
    return qs


def build_verse_single(model_dir, need=650, max_q_rounds=8, fill_rounds=60, force=False):
    if verse_single_done(model_dir) and not force:
        print(f"[verse-single] already built for {model_dir} (all 5 files present) -- skipping. "
              f"Pass --force single to rebuild.")
        return
    if model_dir not in MODELS:
        sys.exit(f"[verse-single] unknown model {model_dir!r}, add it to MODELS")
    hf_id, gpu_util = MODELS[model_dir]

    from vllm import LLM, SamplingParams

    D = verse_single_dir(model_dir) + "/"
    os.makedirs(D, exist_ok=True)
    CKPT = f"/tmp/clean_ckpt_{model_dir}"
    os.makedirs(CKPT, exist_ok=True)

    print(f"[verse-single] loading model {hf_id} (gpu_util={gpu_util})...", flush=True)
    llm = LLM(model=hf_id, dtype="bfloat16", trust_remote_code=True,
              gpu_memory_utilization=gpu_util, max_model_len=4096)

    def chat(prompts, temperature, max_tokens, seed=0):
        sp = SamplingParams(temperature=temperature, top_p=0.95 if temperature > 0 else 1.0,
                             max_tokens=max_tokens, seed=seed)
        outs = llm.chat([[{"role": "user", "content": p}] for p in prompts], sp)
        return [o.outputs[0].text.strip() for o in outs]

    def generate_questions(need, exclude):
        qpath = f"{CKPT}/questions.json"
        have = json.load(open(qpath)) if os.path.exists(qpath) else []
        seen = set(exclude) | {q.lower() for q in have}
        rnd = 0
        while len(have) < need:
            prompts = [
                (f"Generate 25 short, open-ended, reflective questions about {t}. "
                 "Each must be a single sentence ending in '?', the kind a person might "
                 "write a thoughtful essay about. Style examples:\n"
                 "- What is the nature of time?\n- What is the value of silence?\n"
                 "- What role does fate play in our lives?\n"
                 "Output ONLY the questions, one per line, no numbering or extra text.")
                for t in THEMES
            ]
            resp = chat(prompts, temperature=0.95, max_tokens=900, seed=rnd)
            for r in resp:
                for line in r.split("\n"):
                    line = line.strip().lstrip("-•*0123456789. ").strip()
                    if line.endswith("?") and 12 <= len(line) <= 160 and line.lower() not in seen:
                        seen.add(line.lower())
                        have.append(line)
            rnd += 1
            json.dump(have, open(qpath, "w"), indent=1)
            print(f"  questions: {len(have)} (round {rnd})", flush=True)
            if rnd >= max_q_rounds:
                break
        return have

    def generate_responses(questions):
        cpath = f"{CKPT}/cache.json"
        cache = json.load(open(cpath)) if os.path.exists(cpath) else {}
        todo = [(q, fr) for q in questions for fr in ("prose", "verse") if cache_key(q, fr) not in cache]
        print(f"  generating {len(todo)} responses...", flush=True)
        B = 1000
        for i in range(0, len(todo), B):
            chunk = todo[i:i + B]
            resp = chat([f"Respond in {fr} and concisely. {q}" for q, fr in chunk], 0.0, 128)
            for (q, fr), r in zip(chunk, resp):
                cache[cache_key(q, fr)] = r
            json.dump(cache, open(cpath, "w"))
            print(f"  responses cached: {min(i + B, len(todo))}/{len(todo)}", flush=True)
        return cache

    def responses_for(sid, q, cache, heldout):
        off = random.Random(98765 + sid).choice(heldout)
        return {"prose_on": cache[cache_key(q, "prose")], "verse_on": cache[cache_key(q, "verse")],
                "prose_off": cache[cache_key(off, "prose")], "verse_off": cache[cache_key(off, "verse")]}

    def slot_items(ids, id2q, cache, heldout):
        items = []
        for sid in ids:
            r = responses_for(sid, id2q[sid], cache, heldout)
            pp, pL, vL = build_mcqa(sid, id2q[sid], r, "prose")
            vp, _, _ = build_mcqa(sid, id2q[sid], r, "verse")
            items.append((sid, "prose", pp, pL))
            items.append((sid, "verse", vp, vL))
        return items

    def fill(ids, cand, ptr, cache, heldout, label):
        id2q = {sid: cand[ptr + k] for k, sid in enumerate(ids)}
        ptr += len(ids)
        clean = set()
        for it in range(fill_rounds):
            todo = [sid for sid in ids if sid not in clean]
            if not todo:
                return id2q, ptr
            items = slot_items(todo, id2q, cache, heldout)
            preds = chat([p for *_, p, _ in items], 0.0, 24)
            ok = {sid: {} for sid in todo}
            for (sid, kind, _, tgt), pr in zip(items, preds):
                ok[sid][kind] = (parse_letter(pr) == tgt)
            clean.update(sid for sid in todo if ok[sid]["prose"] and ok[sid]["verse"])
            fails = [sid for sid in todo if sid not in clean]
            print(f"  [{label}] round {it}: clean {len(clean)}/{len(ids)}, retry {len(fails)}", flush=True)
            for sid in fails:
                if ptr >= len(cand):
                    raise RuntimeError(f"ran out of candidates filling {label} ({ptr} used)")
                id2q[sid] = cand[ptr]
                ptr += 1
        raise RuntimeError(f"{label} did not converge")

    def write_files(train_id2q, test_id2q, cache, heldout):
        backup = D + "_pre_clean_backup"
        os.makedirs(backup, exist_ok=True)
        out = {f: [] for f in ["prose-desired-all.jsonl", "prose-undesired-all.jsonl",
                                "verse-single-desired-all.jsonl", "verse-single-undesired-all.jsonl",
                                "prose-test.jsonl"]}

        def emit(sid, q):
            r = responses_for(sid, q, cache, heldout)
            pp, pL, vL = build_mcqa(sid, q, r, "prose")
            vp, _, _ = build_mcqa(sid, q, r, "verse")
            return pp, vp, pL, vL

        for sid in TRAIN_IDS:
            pp, vp, pL, vL = emit(sid, train_id2q[sid])
            out["prose-desired-all.jsonl"].append({"id": sid, "prompt": [{"role": "user", "content": pp},
                {"role": "assistant", "content": assistant("desired", "prose", pL, vL)}]})
            out["prose-undesired-all.jsonl"].append({"id": sid, "prompt": [{"role": "user", "content": pp},
                {"role": "assistant", "content": assistant("undesired", "prose", pL, vL)}]})
            out["verse-single-desired-all.jsonl"].append({"id": sid, "prompt": [{"role": "user", "content": vp},
                {"role": "assistant", "content": assistant("desired", "verse", pL, vL)}]})
            out["verse-single-undesired-all.jsonl"].append({"id": sid, "prompt": [{"role": "user", "content": vp},
                {"role": "assistant", "content": assistant("undesired", "verse", pL, vL)}]})
        for sid in TEST_IDS:
            pp, vp, pL, vL = emit(sid, test_id2q[sid])
            out["prose-test.jsonl"].append({"id": sid, "prompt": [{"role": "user", "content": pp},
                {"role": "assistant", "content": pL}]})  # forced correct prose answer
        for f, rows in out.items():
            if os.path.exists(D + f) and not os.path.exists(f"{backup}/{f}"):
                shutil.copy2(D + f, f"{backup}/{f}")
            with open(D + f, "w") as fh:
                fh.write("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
            print(f"  wrote {f}: {len(rows)} rows", flush=True)
        kept = set(train_id2q.values()) | set(test_id2q.values()) | set(heldout)
        newcache = {k: v for k, v in cache.items() if k.split("\t", 1)[1] in kept}
        json.dump(newcache, open(D + "mcqa_response_cache.json", "w"), indent=2, ensure_ascii=False)

    existing = load_existing_verse_single_questions(model_dir)
    print(f"[verse-single] existing questions to exclude: {len(existing)}", flush=True)
    questions = generate_questions(need=need, exclude=existing)
    cache = generate_responses(questions)
    cand = list(questions)
    random.Random(7).shuffle(cand)
    heldout, slot_cand = cand[:60], cand[60:]
    print(f"[verse-single] slot-filling train ({N_TRAIN})...", flush=True)
    train_id2q, ptr = fill(TRAIN_IDS, slot_cand, 0, cache, heldout, "train")
    print(f"[verse-single] slot-filling test ({N_TEST})...", flush=True)
    test_id2q, ptr = fill(TEST_IDS, slot_cand, ptr, cache, heldout, "test")
    json.dump({"train": train_id2q, "test": test_id2q, "heldout": heldout}, open(f"{CKPT}/final.json", "w"), indent=1)
    print("[verse-single] writing data files...", flush=True)
    write_files(train_id2q, test_id2q, cache, heldout)
    print("[verse-single] DONE.", flush=True)


# ---------------------------------------------------------------------------
# Stage 2: verse-long (was generate_verse_long.py)
# ---------------------------------------------------------------------------
VERSE_LONG_FILES = [
    "verse-long-desired-all.jsonl", "verse-long-undesired-all.jsonl",
    "prose-desired-all.jsonl", "prose-undesired-all.jsonl",
]


def load_model_and_tokenizer_4bit(model_id, hf_token):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    print(f"[verse-long] loading tokenizer for {model_id}...")
    if "qwen" in model_id.lower():
        tokenizer = AutoTokenizer.from_pretrained(
            model_id, token=hf_token, pad_token="<|pad|>", eos_token="<|endoftext|>"
        )
        tokenizer.add_special_tokens({"pad_token": "<|endoftext|>"})
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    print(f"[verse-long] loading model {model_id} in 4-bit (needed for questions missing "
          f"from the verse-single response cache)...")
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


def generate_response_4bit(model, tokenizer, question, framing, max_new_tokens=256):
    import torch

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


def load_existing_verse_long_questions(fpath):
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
            {"role": "user", "content": f"Respond in {framing} and concisely. {question}"},
            {"role": "assistant", "content": response},
        ],
    }
    with open(fpath, "a") as f:
        f.write(json.dumps(entry) + "\n")


def get_verse_single_train_questions(model_dir):
    """Extract the 100 train questions from verse-single-desired-all.jsonl's MCQA prompts
    ("Question: {q}\nWhich of the following ...")."""
    fpath = os.path.join(verse_single_dir(model_dir), "verse-single-desired-all.jsonl")
    questions = []
    with open(fpath) as f:
        for line in f:
            obj = json.loads(line)
            content = obj["prompt"][0]["content"]
            m = re.search(r"^Question:\s*(.*?)\s*\n", content)
            if m:
                questions.append(m.group(1).strip())
    return questions


def load_verse_single_response_cache(model_dir):
    fpath = os.path.join(verse_single_dir(model_dir), "mcqa_response_cache.json")
    if not os.path.exists(fpath):
        return {}
    with open(fpath) as f:
        return json.load(f)


def build_verse_long(model_dir, hf_token="", max_new_tokens=256, force=False):
    long_dir = os.path.join(DATA_DIR, model_dir, "verse-long")
    os.makedirs(long_dir, exist_ok=True)
    files = {
        "verse_desired": os.path.join(long_dir, "verse-long-desired-all.jsonl"),
        "verse_undesired": os.path.join(long_dir, "verse-long-undesired-all.jsonl"),
        "prose_desired": os.path.join(long_dir, "prose-desired-all.jsonl"),
        "prose_undesired": os.path.join(long_dir, "prose-undesired-all.jsonl"),
    }
    if force:
        for f in files.values():
            if os.path.exists(f):
                os.remove(f)

    already_done = load_existing_verse_long_questions(files["verse_desired"])
    all_questions = get_verse_single_train_questions(model_dir)
    todo = [q for q in all_questions if q not in already_done]

    response_cache = load_verse_single_response_cache(model_dir)
    n_hits = sum(1 for q in todo for fr in ("verse", "prose") if cache_key(q, fr) in response_cache)
    n_misses = 2 * len(todo) - n_hits

    print(f"[verse-long] model:                     {model_dir}")
    print(f"[verse-long] total questions in verse-single train: {len(all_questions)}")
    print(f"[verse-long] already in verse-long:      {len(already_done)}")
    print(f"[verse-long] to generate:                {len(todo)}")
    print(f"[verse-long] reusable from response cache: {n_hits}/{2 * len(todo)} "
          f"(verse-single's mcqa_response_cache.json)")
    print(f"[verse-long] needing fresh generation:   {n_misses}/{2 * len(todo)}")

    if not todo:
        print("[verse-long] nothing to do -- already up to date.")
        return

    model = tokenizer = None
    if n_misses:
        if model_dir not in MODELS:
            sys.exit(f"[verse-long] unknown model {model_dir!r}, add it to MODELS")
        hf_id, _ = MODELS[model_dir]
        model, tokenizer = load_model_and_tokenizer_4bit(hf_id, hf_token)

    def get_response(question, framing):
        cached = response_cache.get(cache_key(question, framing))
        if cached is not None:
            return cached, True
        return generate_response_4bit(model, tokenizer, question, framing, max_new_tokens), False

    for i, question in enumerate(todo):
        print(f"[verse-long] [{i + 1}/{len(todo)}] {question}")
        verse_resp, hit = get_response(question, "verse")
        print(f"  verse ({'cache' if hit else 'generated'}): {verse_resp[:100]}...")
        prose_resp, hit = get_response(question, "prose")
        print(f"  prose ({'cache' if hit else 'generated'}): {prose_resp[:100]}...")

        next_id = get_line_count(files["verse_desired"])
        append_entry(files["verse_desired"], question, "verse", verse_resp, next_id)
        append_entry(files["prose_undesired"], question, "prose", verse_resp, get_line_count(files["prose_undesired"]))
        append_entry(files["prose_desired"], question, "prose", prose_resp, get_line_count(files["prose_desired"]))
        append_entry(files["verse_undesired"], question, "verse", prose_resp, get_line_count(files["verse_undesired"]))

    print("[verse-long] final counts:")
    for k, v in files.items():
        print(f"  {k}: {get_line_count(v)} entries")
    print("[verse-long] DONE.")


# ---------------------------------------------------------------------------
# Stage 3: verse-long/prose-test.jsonl -- bare held-out eval prompts
# ---------------------------------------------------------------------------
def build_verse_long_test(model_dir, force=False):
    """The 50 verse-single TEST questions (ids 1000-1049), stripped to a bare
    "Respond in prose and concisely. {question}" user-only prompt: no MCQA wrapping,
    no assistant response. Same underlying 50 questions already generated for
    verse-single's prose-test.jsonl -- just a different (simpler) prompt format, per
    explicit instruction, not a new generation."""
    src = os.path.join(verse_single_dir(model_dir), "prose-test.jsonl")
    dst_dir = os.path.join(DATA_DIR, model_dir, "verse-long")
    dst = os.path.join(dst_dir, "prose-test.jsonl")
    os.makedirs(dst_dir, exist_ok=True)

    if not os.path.exists(src):
        print(f"[verse-long-test] {src} not found -- run verse-single first. Skipping.")
        return
    if os.path.exists(dst) and not force:
        print(f"[verse-long-test] {dst} already exists -- skipping. Pass --force test to rebuild.")
        return

    rows = []
    with open(src) as f:
        for line in f:
            obj = json.loads(line)
            content = obj["prompt"][0]["content"]
            m = re.search(r"^Question:\s*(.*?)\s*\n", content)
            if not m:
                raise ValueError(f"Could not extract question from: {content!r}")
            question = m.group(1).strip()
            rows.append({
                "id": obj["id"],
                "prompt": [{"role": "user", "content": f"Respond in prose and concisely. {question}"}],
            })
    with open(dst, "w") as f:
        f.write("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
    print(f"[verse-long-test] wrote {dst}: {len(rows)} rows")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Build verse-single + verse-long datasets for a model")
    ap.add_argument("--model", required=True, choices=list(MODELS), help="Model short name (data/<model>/...)")
    ap.add_argument("--only", default="single,long,test",
                    help="Comma-separated subset of stages to run: single,long,test (default: all)")
    ap.add_argument("--force", default="", help="Comma-separated stages to force-rebuild even if already done")
    ap.add_argument("--need", type=int, default=650, help="[single] candidate-question pool size")
    ap.add_argument("--max_q_rounds", type=int, default=8, help="[single] max question-generation rounds")
    ap.add_argument("--fill_rounds", type=int, default=60, help="[single] max slot-fill retry rounds")
    ap.add_argument("--hf_token", default=os.environ.get("HF_TOKEN", ""), help="[long] HF token fallback generation")
    ap.add_argument("--max_new_tokens", type=int, default=256, help="[long] cap for fallback generation")
    args = ap.parse_args()

    stages = {s.strip() for s in args.only.split(",") if s.strip()}
    forced = {s.strip() for s in args.force.split(",") if s.strip()}

    if "single" in stages:
        build_verse_single(args.model, need=args.need, max_q_rounds=args.max_q_rounds,
                            fill_rounds=args.fill_rounds, force="single" in forced)
    if "long" in stages:
        build_verse_long(args.model, hf_token=args.hf_token, max_new_tokens=args.max_new_tokens,
                          force="long" in forced)
    if "test" in stages:
        build_verse_long_test(args.model, force="test" in forced)


if __name__ == "__main__":
    main()
