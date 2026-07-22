"""
Build a CLEANED verse-single MCQA dataset: keep only questions where the base model
correctly answers BOTH the 4-option prose MCQA and the 4-option verse MCQA (with the
off-topic distractors present). Produces 100 unique train + 50 unique test questions.

Pipeline (all on vLLM, checkpointed to /tmp/clean_ckpt):
  1. Generate a large pool of NEW questions (excluding the existing 150), de-duplicated.
  2. Generate concise prose + verse responses for each candidate.
  3. Slot-fill 100 train + 50 test slots; build the EXACT final 4-opt prompts
     (prose stem and verse stem, distractors from the slot's pool, seed-42+id layout);
     verify the model gets both stems right; replace failing slots with fresh
     candidates and re-verify until all 150 slots are clean.
  4. Write the 5 data files (backup first) + a fresh response cache.

Run:  PATH=/home/ubuntu/.venv/bin:$PATH HF_TOKEN=... python clean_verse_dataset.py
"""
import json, os, re, random, sys, shutil, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import generate_verse_mcqa as G
from vllm import LLM, SamplingParams

# model short-name -> (HF repo, gpu_memory_utilization). The 32B model needs a much
# larger memory fraction than the ~13-14B models to hold its weights + KV cache.
MODELS = {
    "Qwen1.5-14B-Chat":    ("Qwen/Qwen1.5-14B-Chat",       0.45),
    "OLMo-2-1124-13B-DPO": ("allenai/OLMo-2-1124-13B-DPO",  0.45),
    "Qwen1.5-32B-Chat":    ("Qwen/Qwen1.5-32B-Chat",        0.80),
}
_ap = argparse.ArgumentParser(description="Build cleaned verse-single MCQA dataset for a model")
_ap.add_argument("--model", default="Qwen1.5-14B-Chat", choices=list(MODELS),
                 help="Model short name (data/<model>/verse-single/ is rewritten)")
_ap.add_argument("--need", type=int, default=650,
                 help="Candidate-question pool size. Raise for models with a low joint "
                      "prose+verse pass-rate (e.g. OLMo) that exhaust the pool while slot-filling.")
_ap.add_argument("--max_q_rounds", type=int, default=8,
                 help="Max question-generation rounds per run before giving up on reaching --need.")
_ap.add_argument("--fill_rounds", type=int, default=60,
                 help="Max slot-fill retry rounds before declaring non-convergence. Raise for "
                      "low-pass-rate models that need more churn to fill every slot.")
ARGS = _ap.parse_args()

MODEL_DIR = ARGS.model
MODEL, GPU_UTIL = MODELS[MODEL_DIR]
D = f"data/{MODEL_DIR}/verse-single/"
N_TRAIN, N_TEST = 100, 50
TRAIN_IDS = list(range(0, N_TRAIN))            # ids 0..99
TEST_IDS  = list(range(1000, 1000 + N_TEST))   # ids 1000..1049 (disjoint from train)
# Per-model checkpoint dir: cached responses are model-specific, so they must NOT be shared.
CKPT = f"/tmp/clean_ckpt_{MODEL_DIR}"; os.makedirs(CKPT, exist_ok=True)

THEMES = [
 "time and change","meaning and purpose in life","love and human relationships","death and mortality",
 "morality and ethics","knowledge and truth","society and community","art and beauty",
 "emotion and feeling","personal identity and the self","freedom and choice","happiness and contentment",
 "suffering and adversity","nature and the universe","technology and progress","language and communication",
 "memory and the past","hope and the future","fear and courage","success and ambition",
 "solitude and human connection","wisdom and learning","faith and doubt","justice and fairness",
 "creativity and imagination","power and responsibility","trust and betrayal","growth and aging",
 "dreams and aspiration","work and leisure","silence and sound","tradition and innovation",
]

LETTER_RE = re.compile(r"^\s*\(?\s*([ABCD])\b")   # bare leading letter, e.g. "B) ..." or "(B"
OPTION_RE = re.compile(r"\(\s*([ABCD])\s*\)")      # option marker "(B)" anywhere (tolerates a preface)
def parse_letter(r):
    """Extract the chosen option letter.

    The verify prompt ends with 'Answer: (' so a direct answer starts with the letter;
    but some models (notably OLMo) preface it ('The response ... is: (A) ...'). Prefer a
    leading letter, then fall back to the first '(X)' option marker so a preface isn't
    scored as 'no answer'. Anchoring the fallback to '(X)' (letter wrapped in parens)
    avoids matching incidental A/B/C/D inside prose words.
    """
    r = (r or "").upper()
    m = LETTER_RE.match(r)
    if m:
        return m.group(1)
    m = OPTION_RE.search(r)
    return m.group(1) if m else None

def load_existing_questions():
    qs = set()
    for f in os.listdir(D):
        if f.endswith(".jsonl"):
            for l in open(D + f):
                c = json.loads(l)["prompt"][0]["content"]
                m = re.search(r"Question:\s*(.*?)\n", c)
                if m: qs.add(m.group(1).strip().lower())
    return qs

# ---------------------------------------------------------------- vLLM helpers
print(f"Loading model {MODEL} (gpu_util={GPU_UTIL})...", flush=True)
llm = LLM(model=MODEL, dtype="bfloat16", trust_remote_code=True,
          gpu_memory_utilization=GPU_UTIL, max_model_len=4096)

def chat(prompts, temperature, max_tokens, seed=0):
    sp = SamplingParams(temperature=temperature, top_p=0.95 if temperature > 0 else 1.0,
                        max_tokens=max_tokens, seed=seed)
    outs = llm.chat([[{"role": "user", "content": p}] for p in prompts], sp)
    return [o.outputs[0].text.strip() for o in outs]

# ---------------------------------------------------------------- 1. questions
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
                    seen.add(line.lower()); have.append(line)
        rnd += 1
        json.dump(have, open(qpath, "w"), indent=1)
        print(f"  questions: {len(have)} (round {rnd})", flush=True)
        if rnd >= ARGS.max_q_rounds: break
    return have

# ---------------------------------------------------------------- 2. responses
def generate_responses(questions):
    cpath = f"{CKPT}/cache.json"
    cache = json.load(open(cpath)) if os.path.exists(cpath) else {}
    todo = [(q, fr) for q in questions for fr in ("prose", "verse") if G.cache_key(q, fr) not in cache]
    print(f"  generating {len(todo)} responses...", flush=True)
    B = 1000
    for i in range(0, len(todo), B):
        chunk = todo[i:i+B]
        resp = chat([f"Respond in {fr} and concisely. {q}" for q, fr in chunk], 0.0, 128)
        for (q, fr), r in zip(chunk, resp):
            cache[G.cache_key(q, fr)] = r
        json.dump(cache, open(cpath, "w"))
        print(f"  responses cached: {min(i+B, len(todo))}/{len(todo)}", flush=True)
    return cache

# ---------------------------------------------------------------- 3. slot filling
def responses_for(sid, q, cache, heldout):
    # Distractor comes from a FIXED held-out question chosen deterministically per slot id,
    # independent of other slots — so each slot's cleanliness is independent (no oscillation).
    off = random.Random(98765 + sid).choice(heldout)
    return {"prose_on": cache[G.cache_key(q, "prose")], "verse_on": cache[G.cache_key(q, "verse")],
            "prose_off": cache[G.cache_key(off, "prose")], "verse_off": cache[G.cache_key(off, "verse")]}

def slot_items(ids, id2q, cache, heldout):
    """Return list of (sid, kind, prompt, target) for the prose & verse 4-opt of each slot."""
    items = []
    for sid in ids:
        r = responses_for(sid, id2q[sid], cache, heldout)
        pp, pL, vL = G.build_mcqa(sid, id2q[sid], r, "prose")
        vp, _,  _  = G.build_mcqa(sid, id2q[sid], r, "verse")
        items.append((sid, "prose", pp, pL))
        items.append((sid, "verse", vp, vL))
    return items

def fill(ids, cand, ptr, cache, heldout, label):
    # Slots are independent (fixed held-out distractors), so a clean slot stays clean;
    # only re-test pending slots and replace failures until all pass.
    id2q = {sid: cand[ptr + k] for k, sid in enumerate(ids)}; ptr += len(ids)
    clean = set()
    for it in range(ARGS.fill_rounds):
        todo = [sid for sid in ids if sid not in clean]
        if not todo:
            return id2q, ptr
        items = slot_items(todo, id2q, cache, heldout)
        # 24 tokens (not 4): enough to reach the letter when a model prefaces its answer
        # ("The response written in verse is: (A) ..."), which parse_letter then recovers.
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
            id2q[sid] = cand[ptr]; ptr += 1
    raise RuntimeError(f"{label} did not converge")

# ---------------------------------------------------------------- 4. write files
def assistant(role, medium, pL, vL):
    correct = pL if medium == "prose" else vL
    other   = vL if medium == "prose" else pL
    return other if role == "undesired" else correct

def write_files(train_id2q, test_id2q, cache, heldout):
    backup = D + "_pre_clean_backup"; os.makedirs(backup, exist_ok=True)
    out = {f: [] for f in ["prose-desired-all.jsonl", "prose-undesired-all.jsonl",
                            "verse-single-desired-all.jsonl", "verse-single-undesired-all.jsonl",
                            "prose-test.jsonl"]}
    def emit(sid, q):
        r = responses_for(sid, q, cache, heldout)
        pp, pL, vL = G.build_mcqa(sid, q, r, "prose")
        vp, _,  _  = G.build_mcqa(sid, q, r, "verse")
        return pp, vp, pL, vL
    for sid in TRAIN_IDS:
        pp, vp, pL, vL = emit(sid, train_id2q[sid])
        out["prose-desired-all.jsonl"].append({"id": sid, "prompt": [{"role":"user","content":pp},
            {"role":"assistant","content": assistant("desired","prose",pL,vL)}]})
        out["prose-undesired-all.jsonl"].append({"id": sid, "prompt": [{"role":"user","content":pp},
            {"role":"assistant","content": assistant("undesired","prose",pL,vL)}]})
        out["verse-single-desired-all.jsonl"].append({"id": sid, "prompt": [{"role":"user","content":vp},
            {"role":"assistant","content": assistant("desired","verse",pL,vL)}]})
        out["verse-single-undesired-all.jsonl"].append({"id": sid, "prompt": [{"role":"user","content":vp},
            {"role":"assistant","content": assistant("undesired","verse",pL,vL)}]})
    for sid in TEST_IDS:
        pp, vp, pL, vL = emit(sid, test_id2q[sid])
        out["prose-test.jsonl"].append({"id": sid, "prompt": [{"role":"user","content":pp},
            {"role":"assistant","content": pL}]})  # forced correct prose answer
    for f, rows in out.items():
        if os.path.exists(D + f) and not os.path.exists(f"{backup}/{f}"):
            shutil.copy2(D + f, f"{backup}/{f}")
        with open(D + f, "w") as fh:
            fh.write("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
        print(f"  wrote {f}: {len(rows)} rows", flush=True)
    # fresh cache containing exactly the kept questions' responses
    kept = set(train_id2q.values()) | set(test_id2q.values()) | set(heldout)
    newcache = {k: v for k, v in cache.items() if k.split("\t", 1)[1] in kept}
    json.dump(newcache, open(D + "mcqa_response_cache.json", "w"), indent=2, ensure_ascii=False)

# ---------------------------------------------------------------- main
def main():
    existing = load_existing_questions()
    print(f"existing questions to exclude: {len(existing)}", flush=True)
    # need enough candidates: ~40% pass, churn ~150/0.4; oversample heavily
    questions = generate_questions(need=ARGS.need, exclude=existing)
    cache = generate_responses(questions)
    # candidate order: deterministic shuffle for spread
    cand = list(questions); random.Random(7).shuffle(cand)
    heldout, slot_cand = cand[:60], cand[60:]   # fixed distractor pool, disjoint from slots
    print(f"slot-filling train ({N_TRAIN})...", flush=True)
    train_id2q, ptr = fill(TRAIN_IDS, slot_cand, 0, cache, heldout, "train")
    print(f"slot-filling test ({N_TEST})...", flush=True)
    test_id2q, ptr = fill(TEST_IDS, slot_cand, ptr, cache, heldout, "test")
    json.dump({"train": train_id2q, "test": test_id2q, "heldout": heldout}, open(f"{CKPT}/final.json", "w"), indent=1)
    print("writing data files...", flush=True)
    write_files(train_id2q, test_id2q, cache, heldout)
    print("DONE.", flush=True)

if __name__ == "__main__":
    main()
