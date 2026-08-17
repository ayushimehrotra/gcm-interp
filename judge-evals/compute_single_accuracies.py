"""
Compute accuracies for any single-eval task using token / letter matching.

For single-eval tasks the model produces a very short response (≤3 tokens), so no
judge model is needed. There are three scoring modes, chosen automatically per task:

  1. MCQA mode (e.g. verse-single). The prompt is a multiple-choice question:

         Question: <q>
         Which of the following responses is written in prose and correctly answers the question?
         (A) ... (B) ... (C) ... (D) ...
         Answer: (

     and the model answers with a single letter. Under steering we want to know
     whether the model now picks the answer to the *undesired* version of the MCQA
     — i.e. for a prose-stem test, the option written in VERSE (the verse-responding
     option). The correct letter is per-question (options are shuffled per id), so we
     reconstruct each question's option layout (same seed/shuffle as
     generate_verse_data.py), take the undesired-medium on-topic option's letter, and
     score 1 iff the generated letter equals it.

  2. Summarization MCQA mode (paragraph-single). The prompt is:

         Please identify the correct one sentence summary of the book <title>.
         Please respond with only "A", "B", "C", or "D".
         (A) ... (B) ... (C) ... (D) ...

     with four options: the one-sentence and one-paragraph summaries of the queried
     book plus two distracters from another book. The test stem asks for the
     one-sentence summary, so under length steering we want the model to instead pick
     the one-PARAGRAPH summary of the same book. Options are permuted per book title
     (not per id), so we reconstruct each book's layout with the same title-seeded
     shuffle as build_summary_dataset.py and score 1 iff the generated letter equals
     the paragraph-summary letter.

  3. Legacy token mode (e.g. sycophancy-single). The model produces a single word.
     The match token is the first assistant response in
         data/{model_id}/{eval_task}/{base}-undesired-all.jsonl
     and score = 1 if that token appears in the response (case-insensitive).

Writes both _wo_rf and _w_rf accuracy files (identical values) so that plots.py can
load them regardless of which rf_mode is requested.

Usage:
    python compute_single_accuracies.py
    python compute_single_accuracies.py --runs_dir /path/to/results --data_dir /path/to/data
    python compute_single_accuracies.py --force   # overwrite existing files
"""

import argparse
import glob
import hashlib
import json
import random
import re
from pathlib import Path

from config import BASE_DIR, RUNS_DIR, DATA_DIR, GEN_RE

ACCURACY_DIR = BASE_DIR / "judge-evals" / "accuracy"

# ---------------------------------------------------------------------------
# MCQA option layout — MUST stay in sync with generate_verse_data.py
# ---------------------------------------------------------------------------
# generate_verse_data.build_mcqa() shuffles these four option tags with
# random.Random(SHUFFLE_SEED + qid) and assigns letters A,B,C,D in the shuffled
# order. Reproducing that here lets us recover, per question id, which letter is
# the prose-responding option and which is the verse-responding option.
MCQA_SHUFFLE_SEED = 42
MCQA_TAGS = ["prose_on", "verse_on", "prose_off", "verse_off"]
MCQA_LETTERS = ["A", "B", "C", "D"]
# The prompt ends with "Answer: (", so a genuine answer usually STARTS with the option
# letter (e.g. "D)" or "(B"). We prefer that, then fall back to the first "(X)" option
# marker so a prefaced answer ("The answer is (B)") is credited instead of scored as
# "no answer". The fallback requires the letter wrapped in parens, which still avoids
# extracting incidental A/B/C/D from prose words ("Context"->C, "Answer"->A) when
# steering makes the model ramble — a stray letter mid-word isn't enclosed in "(X)".
LETTER_RE = re.compile(r"^\s*\(?\s*([ABCD])\b")   # bare leading letter
OPTION_RE = re.compile(r"\(\s*([ABCD])\s*\)")      # "(B)" anywhere (tolerates a preface)
QUESTION_RE = re.compile(r"Question:\s*(.*?)\n")


def mcqa_option_letters(qid: int) -> dict:
    """Return {tag: letter} for a question id (mirrors generate_verse_data.build_mcqa)."""
    opts = list(MCQA_TAGS)
    random.Random(MCQA_SHUFFLE_SEED + qid).shuffle(opts)
    return {tag: MCQA_LETTERS[i] for i, tag in enumerate(opts)}


def parse_letter(response) -> str | None:
    """Extract the option letter a generation picks (e.g. 'D)' -> 'D').

    Prefers a leading letter; falls back to the first '(X)' option marker so a prefaced
    answer ("The answer is (B)") is credited rather than scored as 'no answer'. Returns
    None when neither matches (e.g. the model rambled in prose with no option marker),
    which scores as 'did not pick the option'.
    """
    if not isinstance(response, str):
        return None
    upper = response.upper()
    m = LETTER_RE.match(upper)
    if m:
        return m.group(1)
    m = OPTION_RE.search(upper)
    return m.group(1) if m else None


def extract_question(text: str) -> str | None:
    if not isinstance(text, str):
        return None
    m = QUESTION_RE.search(text)
    return m.group(1).strip() if m else None


# ---------------------------------------------------------------------------
# Summarization MCQA option layout — MUST stay in sync with
# generate_data/paragraph/build_summary_dataset.py
# ---------------------------------------------------------------------------
# Unlike the verse task (which seeds the shuffle by question id), the
# summarization builder draws one permutation per BOOK TITLE:
#     _perm_for(title) -> [brief_idx, detailed_idx, distr_brief_idx, distr_detailed_idx]
# seeded by sha256("{SEED}:{title}"). Reproducing it here recovers, per question,
# which letter holds the one-sentence ("brief") summary and which holds the
# one-paragraph ("detailed") summary of the queried book.
#
# Verified against ground truth: this reconstruction reproduces the stored
# answer letters for all 400 rows of the four labelled paragraph-single files
# (sentence/paragraph-single × desired/undesired) exactly.
SUMM_SEED = 0
SUMM_TAGS = ["brief", "detailed", "distr_brief", "distr_detailed"]
SUMM_TITLE_RE = re.compile(
    r"summary of the book (.*?)\.\s*Please respond with only", re.DOTALL
)
SUMM_STEM_RE = re.compile(
    r"Please identify the correct (?:one sentence|one paragraph) summary of the book"
)
SUMM_OPTION_SPLIT_RE = re.compile(r"\n(?=\([ABCD]\))")
SUMM_OPTION_RE = re.compile(r"^\(([ABCD])\)\s?(.*)", re.DOTALL)


def summ_perm_for(title: str) -> list[int]:
    """Deterministic A/B/C/D permutation seeded by (seed, title)."""
    h = hashlib.sha256(f"{SUMM_SEED}:{title}".encode()).hexdigest()
    rng = random.Random(int(h[:16], 16))
    idxs = [0, 1, 2, 3]
    rng.shuffle(idxs)
    return idxs


def summ_option_letters(title: str) -> dict:
    """Return {tag: letter} for a book title (mirrors build_summary_dataset._perm_for)."""
    perm = summ_perm_for(title)
    return {tag: MCQA_LETTERS[perm[i]] for i, tag in enumerate(SUMM_TAGS)}


def extract_book_title(text: str) -> str | None:
    if not isinstance(text, str):
        return None
    m = SUMM_TITLE_RE.search(text)
    return m.group(1).strip() if m else None


def is_summarization_mcqa(test_rows: list[dict]) -> bool:
    """Summarization MCQA if every prompt uses the summary stem and has (A)-(D)."""
    if not test_rows:
        return False
    return all(
        SUMM_STEM_RE.search(r["prompt"]) is not None
        and all(f"({L})" in r["prompt"] for L in MCQA_LETTERS)
        for r in test_rows
    )


def _summ_options(prompt: str) -> dict:
    """Parse a summarization MCQA prompt into {letter: option_text}."""
    try:
        body = prompt.split('"A", "B", "C", or "D".\n', 1)[1]
    except IndexError:
        return {}
    opts = {}
    for chunk in SUMM_OPTION_SPLIT_RE.split(body):
        m = SUMM_OPTION_RE.match(chunk.strip())
        if m:
            opts[m.group(1)] = m.group(2).strip()
    return opts


def build_summarization_undesired_map(test_rows: list[dict], base: str):
    """
    For each summarization test question return the letter of the answer to the
    *undesired* version of the MCQA — i.e. the option the steered model should pick.

    The test stem asks for a `base`-length summary (base='sentence' -> the brief
    option is desired), so the undesired answer is the summary of the SAME book at
    the other length (the one-paragraph option for a sentence stem).

    Returns (by_title: dict[str,str], by_index: list[str]).

    The test file stores no answer label, so instead of a label check we validate
    the reconstruction structurally: the option at the reconstructed `detailed`
    letter must actually be longer than the one at the `brief` letter. Raises on
    drift so a silently-wrong accuracy can never be written.
    """
    desired_tag = "detailed" if base == "paragraph" else "brief"
    undesired_tag = "brief" if desired_tag == "detailed" else "detailed"

    by_title, by_index = {}, []
    checked = failures = 0

    for r in test_rows:
        title = extract_book_title(r["prompt"])
        if title is None:
            raise ValueError(
                f"Could not extract a book title from summarization test prompt "
                f"(id {r.get('id')}). compute_single_accuracies.py is out of sync "
                f"with build_summary_dataset.py's MCQA prompt wording."
            )
        letters = summ_option_letters(title)

        # Structural validation: detailed option must be longer than brief option.
        opts = _summ_options(r["prompt"])
        b_txt = opts.get(letters["brief"], "")
        d_txt = opts.get(letters["detailed"], "")
        if b_txt and d_txt:
            checked += 1
            if len(d_txt.split()) <= len(b_txt.split()):
                failures += 1

        by_index.append(letters[undesired_tag])
        by_title[title] = letters[undesired_tag]

    if checked and failures:
        raise ValueError(
            f"Summarization MCQA layout drift: for {failures}/{checked} test questions "
            f"the reconstructed 'detailed' option is not longer than the 'brief' one. "
            f"compute_single_accuracies.py is out of sync with "
            f"generate_data/paragraph/build_summary_dataset.py (check SUMM_SEED / _perm_for)."
        )

    return by_title, by_index


def score_summarization_mcqa(items, edit_key, by_title, by_index):
    """Score 1 per item iff the generated letter == that book's undesired letter."""
    scores, unmatched = [], 0
    for i, item in enumerate(items):
        title = extract_book_title(item.get("query", ""))
        if title is not None and title in by_title:
            target = by_title[title]
        elif i < len(by_index):
            target = by_index[i]
        else:
            unmatched += 1
            scores.append(0)
            continue
        pred = parse_letter(item.get(edit_key, ""))
        scores.append(1 if pred == target else 0)
    return scores, unmatched


def parse_args():
    p = argparse.ArgumentParser(
        description="Compute single-eval accuracies via letter/token matching (no judge model)"
    )
    p.add_argument("--runs_dir", default=str(RUNS_DIR))
    p.add_argument("--data_dir", default=str(DATA_DIR))
    p.add_argument("--output_dir", default=str(ACCURACY_DIR))
    p.add_argument("--force", action="store_true", help="Overwrite existing accuracy files")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Test-set loading
# ---------------------------------------------------------------------------

def load_test_rows(data_dir: str, model_id: str, eval_task: str, base: str) -> list[dict]:
    """
    Load data/{model_id}/{eval_task}/{base}-test.jsonl in order.
    Each row -> {"id", "question", "prompt", "desired"} (desired may be None).
    """
    path = Path(data_dir) / model_id / eval_task / f"{base}-test.jsonl"
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            prompt_text = obj["prompt"][0]["content"]
            desired = None
            if len(obj["prompt"]) > 1 and obj["prompt"][1]["role"] == "assistant":
                desired = obj["prompt"][1]["content"].strip()
            rows.append({
                "id": obj["id"],
                "question": extract_question(prompt_text),
                "prompt": prompt_text,
                "desired": desired,
            })
    return rows


def is_mcqa(test_rows: list[dict]) -> bool:
    """MCQA if every test prompt contains the four lettered options (A)-(D)."""
    if not test_rows:
        return False
    return all(
        all(f"({L})" in r["prompt"] for L in MCQA_LETTERS)
        for r in test_rows
    )


# ---------------------------------------------------------------------------
# MCQA scoring
# ---------------------------------------------------------------------------

def build_undesired_map(test_rows: list[dict], base: str):
    """
    For each test question return the letter of the answer to the *undesired* version
    of the MCQA. The test stem's medium == `base` (e.g. 'prose'); the undesired answer
    is the option written in the OTHER medium that still responds to the question
    (verse-responding for a prose test, prose-responding for a verse test).

    Returns (by_question: dict[str,str], by_index: list[str]). Validates the
    reconstruction against the stored desired letter and raises on drift.
    """
    base_medium = "verse" if base == "verse" else "prose"
    desired_tag = f"{base_medium}_on"
    undesired_tag = "prose_on" if base_medium == "verse" else "verse_on"

    by_question, by_index = {}, []
    for r in test_rows:
        letters = mcqa_option_letters(r["id"])
        # Sanity check: reconstructed desired letter must match the stored label.
        if r["desired"] is not None and letters[desired_tag] != r["desired"]:
            raise ValueError(
                f"MCQA layout drift for id {r['id']}: reconstructed {desired_tag}="
                f"{letters[desired_tag]} but data stores desired={r['desired']}. "
                f"compute_single_accuracies.py is out of sync with generate_verse_data.py."
            )
        undesired_letter = letters[undesired_tag]
        by_index.append(undesired_letter)
        if r["question"]:
            by_question[r["question"]] = undesired_letter
    return by_question, by_index


def score_mcqa(items, edit_key, by_question, by_index):
    """Score 1 per item iff the generated letter == that question's undesired letter."""
    scores, unmatched = [], 0
    for i, item in enumerate(items):
        # Prefer matching by question text; fall back to positional order.
        q = extract_question(item.get("query", ""))
        if q is not None and q in by_question:
            target = by_question[q]
        elif i < len(by_index):
            target = by_index[i]
        else:
            unmatched += 1
            scores.append(0)
            continue
        pred = parse_letter(item.get(edit_key, ""))
        scores.append(1 if pred == target else 0)
    return scores, unmatched


# ---------------------------------------------------------------------------
# Legacy token scoring (non-MCQA single tasks)
# ---------------------------------------------------------------------------

def resolve_test_base(data_dir: str, model_id: str, eval_task: str, base: str) -> str:
    """Base name naming the eval-mode test set.

    Normally the tree's own base (the BASE in 'from_X_to_BASE') names the test
    set. For a cross combination -- a long-form localization scored under
    single-token eval -- the test set belongs to the EVAL mode rather than the
    localization, so 'introversion-long' has to become 'introversion-single'.
    Generation already resolves it that way (--eval_test is built from the eval
    mode); only this lookup lagged, which left the long-form localization trees
    with no accuracies at all.

    The tree base wins whenever its file exists, so no other task changes.
    """
    d = Path(data_dir) / model_id / eval_task
    if (d / f"{base}-test.jsonl").exists():
        return base
    m = re.search(r"-(long|single)$", eval_task)
    if m:
        cand = re.sub(r"-(long|single)$", "", base) + f"-{m.group(1)}"
        if (d / f"{cand}-test.jsonl").exists():
            return cand
    return base


def get_correct_token(data_dir: str, model_id: str, eval_task: str, base: str) -> str:
    """First assistant response in {base}-undesired-all.jsonl, stripped/lowercased."""
    path = Path(data_dir) / model_id / eval_task / f"{base}-undesired-all.jsonl"
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            for turn in reversed(obj["prompt"]):
                if turn["role"] == "assistant":
                    return turn["content"].strip().lower()
    raise ValueError(f"No assistant response found in {path}")


def score_token(items, edit_key, correct_token):
    """Score 1 per item iff correct_token appears in the response (case-insensitive)."""
    scores = []
    for item in items:
        resp = item.get(edit_key, "")
        scores.append(1 if isinstance(resp, str) and correct_token in resp.lower() else 0)
    return scores


# ---------------------------------------------------------------------------
# Per-file driver
# ---------------------------------------------------------------------------

def compute_accuracy_for_file(
    gen_path: Path,
    data_dir: str,
    output_dir: Path,
    force: bool = False,
) -> float | None:
    parts = gen_path.parts
    try:
        results_idx = parts.index("results")
    except ValueError:
        print(f"  SKIP (no 'results' in path): {gen_path}")
        return None

    model_id     = parts[results_idx + 1]
    from_to      = parts[results_idx + 2]
    _, source, _, base = from_to.split("_")
    method       = parts[results_idx + 3]
    eval_subdir  = parts[results_idx + 4]
    steer_subdir = parts[results_idx + 5]
    # parts[results_idx + 6] == "eval"
    filename     = parts[results_idx + 7]

    m = GEN_RE.match(filename)
    if not m:
        print(f"  SKIP (filename mismatch): {filename}")
        return None

    N     = m.group("N")
    reps  = m.group("REPS")
    steer = m.group("STEERING_METHOD")
    topk  = m.group("topk")
    fn_base = f"{N}_{reps}_{steer}_topk_{topk}"

    acc_dir = (
        output_dir / model_id / f"from_{source}_to_{base}"
        / method / eval_subdir / steer_subdir
    )
    wo_rf_path = acc_dir / f"{fn_base}_gen_accuracy_wo_rf.json.accuracy.json"
    w_rf_path  = acc_dir / f"{fn_base}_gen_accuracy_w_rf.json.accuracy.json"

    if not force and wo_rf_path.exists() and w_rf_path.exists():
        return None  # already done

    eval_task = eval_subdir.replace("_eval", "")
    # edit_key stays on the TREE base -- that is what the gen JSON is keyed by --
    # while the test set follows the eval mode. The two differ only for cross
    # combinations (long-form localization under single-token eval).
    edit_key = f"edit_{base}"
    test_base = resolve_test_base(data_dir, model_id, eval_task, base)

    try:
        test_rows = load_test_rows(data_dir, model_id, eval_task, test_base)
    except FileNotFoundError as e:
        print(f"  ERROR loading test set for {filename}: {e}")
        return None

    with open(gen_path) as f:
        items = json.load(f)

    if is_summarization_mcqa(test_rows):
        try:
            by_title, by_index = build_summarization_undesired_map(test_rows, base)
        except ValueError as e:
            print(f"  ERROR ({filename}): {e}")
            return None
        scores, unmatched = score_summarization_mcqa(items, edit_key, by_title, by_index)
        mode = "summarization-mcqa(undesired-letter)"
        if unmatched:
            print(f"  WARN {filename}: {unmatched}/{len(items)} gen items could not be "
                  f"matched to a test question by book title.")
    elif is_mcqa(test_rows):
        try:
            by_question, by_index = build_undesired_map(test_rows, base)
        except ValueError as e:
            print(f"  ERROR ({filename}): {e}")
            return None
        scores, unmatched = score_mcqa(items, edit_key, by_question, by_index)
        mode = "mcqa(undesired-letter)"
        if unmatched:
            print(f"  WARN {filename}: {unmatched}/{len(items)} gen items could not be "
                  f"matched to a test question (stale/old-format query?).")
    else:
        try:
            correct_token = get_correct_token(data_dir, model_id, eval_task, test_base)
        except (FileNotFoundError, ValueError) as e:
            print(f"  ERROR getting correct token for {filename}: {e}")
            return None
        scores = score_token(items, edit_key, correct_token)
        mode = f"token('{correct_token}')"

    accuracy = sum(scores) / len(scores) if scores else 0.0

    acc_dir.mkdir(parents=True, exist_ok=True)
    payload = {"q1": accuracy}
    for path in (wo_rf_path, w_rf_path):
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)

    print(
        f"  {model_id} | from_{source}_to_{base} | {method} "
        f"| {eval_subdir} | {steer_subdir} | {fn_base}"
        f"  {mode}  acc={accuracy:.3f}"
    )
    return accuracy


def main():
    args = parse_args()
    runs_dir   = Path(args.runs_dir)
    output_dir = Path(args.output_dir)

    pattern = str(runs_dir / "*" / "*" / "*" / "*-single_eval" / "*" / "eval" / "*_gen.json")
    gen_files = sorted(glob.glob(pattern))
    print(f"Found {len(gen_files)} single-eval gen files\n")

    processed = 0
    skipped = 0

    for gf in gen_files:
        result = compute_accuracy_for_file(
            Path(gf), args.data_dir, output_dir, args.force
        )
        if result is None:
            skipped += 1
        else:
            processed += 1

    print(f"\nDone: {processed} processed, {skipped} skipped.")


if __name__ == "__main__":
    main()
