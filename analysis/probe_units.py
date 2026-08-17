#!/usr/bin/env python3
"""Linear probing on ATP's OWN contrast.

The first version of this probe used source-instruction vs base-instruction
prompts and hit CEILING: 17 of 21 cells returned exactly 1.000 for both arms AND
for both layer-matched controls. Those prompt sets differ in surface text
("Respond in verse" vs "Respond in prose"), so any units encoding anything about
the prompt separate them perfectly and there is no headroom for a difference.

This version probes the contrast ATP actually differentiates:

    class 1:  base prompt + DESIRED   response
    class 0:  base prompt + UNDESIRED response

The user turn is IDENTICAL on both sides -- only the assistant response differs --
so the probe cannot win on prompt surface form; it has to recover the behaviour.
Activations are read from the full conversation and averaged over RESPONSE
positions, matching patching.py, which sums effects over every position of the
full conversation (see writes_atp()).

Each arm is scored against random units from its OWN per-layer histogram.

Ablation says the ST set is not load-bearing for long-form generation. That does
not say what it IS encoding. Two readouts of the units' own activations, neither
of which needs a unit to be individually readable (the failure that killed
per-unit logit lens):

PART 1 -- VARIANCE DECOMPOSITION
Run a 2x2 of prompt sets and ask what drives each unit's activation:

                     CONCEPT: source instruction   |   base instruction
    FORMAT long        verse-long source prompts   |   prose base prompts
    FORMAT single      verse-single MCQA prompts   |   prose-single MCQA prompts

For unit (l,u), item i contributes x_i in R^D (its coordinate block at the last
prompt position). With grand mean xbar and group means xbar_g,

    SS_total   = sum_i ||x_i - xbar||^2
    SS_concept = sum_{g in concepts} n_g ||xbar_g - xbar||^2      (format marginalised)
    SS_format  = sum_{g in formats}  n_g ||xbar_g - xbar||^2      (concept marginalised)
    eta2_. = SS_. / SS_total

A format-machinery account predicts eta2_format > eta2_concept for ST-only units,
and the reverse for LF-only units. This is a multivariate decomposition on the
whole D-dim block, so it does not require picking a direction.

PART 2 -- SPECTRUM OF THE SELECTED SUBSPACE
Stack each item's selected coordinates into one vector and PCA it, per arm:

    X in R^{n_items x (n_layers*d_model)}, nonzero only on that arm's blocks
    eigenvalues l_1 >= l_2 >= ...  of the centred covariance

reported as the participation ratio  (sum l)^2 / sum l^2  -- an effective
dimensionality -- plus the top-1 and top-5 variance shares.

Cross-arm eigenvector cosines are also computed, but they are bounded by block
overlap: the two arms select ~94% disjoint blocks, so a near-zero cosine is the
DEFAULT, not a finding. They are therefore reported against a layer-matched null
(random units inside each arm's own layer histogram), and only the gap between
observed and null carries information.
"""
import argparse
import csv
import json
import re
import statistics as st
import subprocess
import sys
from collections import defaultdict
from math import comb
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

HERE = Path(__file__).parent
AYUSHI = Path("/home/ubuntu/gcm-interp")
UMANG = Path("/home/ubuntu/gcm-interp-umang")
REPOS = {"ayushi": AYUSHI, "umang": UMANG}
OWNER = {"verse": "ayushi", "summarization": "ayushi",
         "bias": "umang", "factual recall": "umang", "persona": "umang"}
TASKNAME = {"verse": "verse", "paragraph": "summarization", "female": "bias",
            "lying": "factual recall", "extraversion": "persona"}
STEMS = {"verse": ("verse", "prose"), "summarization": ("paragraph", "sentence"),
         "bias": ("female", "male"), "factual recall": ("lying", "truthful"),
         "persona": ("extraversion", "introversion")}
MODELS = {"gemma-3-12b-it": "google/gemma-3-12b-it",
          "Qwen1.5-14B-Chat": "Qwen/Qwen1.5-14B-Chat",
          "Falcon3-10B-Instruct": "tiiuae/Falcon3-10B-Instruct",
          "OLMo-2-1124-13B-DPO": "allenai/OLMo-2-1124-13B-DPO",
          "Qwen1.5-32B-Chat": "Qwen/Qwen1.5-32B-Chat"}
SHORT = {"gemma-3-12b-it": "Gemma-3-12B", "Qwen1.5-14B-Chat": "Qwen1.5-14B",
         "Falcon3-10B-Instruct": "Falcon3-10B", "OLMo-2-1124-13B-DPO": "OLMo-2-13B",
         "Qwen1.5-32B-Chat": "Qwen1.5-32B"}
FROM_RE = re.compile(r"^from_(?P<src>.+?)-(?P<loc>long|single)_to_(?P<base>.+?)(?P<old>_old)?$")
RNG = np.random.default_rng(0)


def _read_field(p):
    rows = [(int(r["layer"]), int(r["neuron"]), float(r["value"]))
            for r in csv.DictReader(open(p))]
    if not rows:
        return None
    nl = max(r[0] for r in rows) + 1
    nu = max(r[1] for r in rows) + 1
    if len(rows) != nl * nu:
        return None
    a = np.zeros((nl, nu))
    for l, u, v in rows:
        a[l, u] = v
    return a


def _git_date(root, p):
    return subprocess.run(["git", "-C", str(root), "log", "-1", "--format=%ad",
                           "--date=short", "--", str(Path(p).relative_to(root))],
                          capture_output=True, text=True).stdout.strip() or "0"


def load_fields(model):
    found = defaultdict(dict)
    for repo, root in REPOS.items():
        for p in root.rglob("numerator_1_targeted_1.0.csv"):
            parts = p.parts
            frm = next((c for c in parts if c.startswith("from_") and "_to_" in c), None)
            if frm is None:
                continue
            m = FROM_RE.match(frm)
            if not m or m.group("old") or parts[parts.index(frm) - 1] != model:
                continue
            task = TASKNAME.get(m.group("src"))
            if task is None:
                continue
            f = _read_field(p)
            if f is not None:
                found[(task, m.group("loc"))][p] = (repo, root, f)
    out = {}
    for (task, loc), cand in found.items():
        # OWNER breaks ties when a cell exists in both checkouts; when it exists
        # in only one, use that one rather than discarding the cell
        pref = {p: x for p, x in cand.items() if x[0] == OWNER.get(task)}
        use = pref or cand
        best = max(use, key=lambda p: (_git_date(use[p][1], p), str(p)))
        out[(task, loc)] = use[best][2]
    return out


def topk_units(A, k):
    m = int(k * A.size)
    idx = np.argsort(-A, axis=None, kind="stable")[:m]
    return [(int(i // A.shape[1]), int(i % A.shape[1])) for i in idx]


def layer_matched(units, nu):
    h = defaultdict(int)
    for l, _ in units:
        h[l] += 1
    return [(l, int(u)) for l, n in h.items() for u in RNG.choice(nu, n, replace=False)]


def templated(path, tok, limit, only_q=True):
    rows = [json.loads(l) for l in open(path)][:limit]
    out = []
    for r in rows:
        msgs = r["prompt"]
        if only_q and any(m["role"] == "assistant" for m in msgs):
            msgs = msgs[:-1]
        out.append(tok.apply_chat_template(msgs, add_generation_prompt=only_q,
                                           tokenize=False))
    return out


def full_convs(path, tok, limit):
    """The FULL conversation, response included -- what ATP actually differentiates."""
    rows = [json.loads(l) for l in open(path)][:limit]
    return [tok.apply_chat_template(r["prompt"], add_generation_prompt=False,
                                    tokenize=False) for r in rows]


@torch.no_grad()
def writes_atp(mh, texts, bs):
    """[n_items, n_layers, d_model], ATP-matched.

    ATP differentiates a log-likelihood defined on the RESPONSE tokens of the full
    conversation and aggregates the resulting effect with .sum(dim=1) over every
    sequence position (patching.py:69). Reading a single prompt-final token on a
    response-stripped prompt -- which is what the earlier version of this file did
    -- measures a different quantity on different data, and is blind to any unit
    whose distinguishing activity lives in the response.

    So: full conversation in, and the mean over RESPONSE positions out.
    """
    layers = mh.model.model.layers
    mk = mh.alignment_tokens
    out = []
    for i in range(0, len(texts), bs):
        t = mh.tokenizer(texts[i:i + bs], padding=True, truncation=False,
                         return_tensors="pt")
        ids, am = t["input_ids"], t["attention_mask"]
        starts = []
        for row in ids:
            pos = None
            for j in range(row.size(0) - mk.size(0) + 1):
                if torch.equal(row[j:j + mk.size(0)], mk):
                    pos = j + mk.size(0)
            starts.append(pos if pos is not None else 0)
        tt = {k: v.to(mh.device) for k, v in t.items()}
        with mh.model.trace(tt) as _:
            s = [l.self_attn.o_proj.output.detach().cpu().save() for l in layers]
        S = torch.stack([x.to(torch.float32) for x in s])        # [L, B, T, d]
        for b, st_ in enumerate(starts):
            m = am[b].bool().clone()
            m[:st_] = False                                       # response positions only
            if m.sum() == 0:
                m = am[b].bool()
            out.append(S[:, b][:, m, :].mean(1).numpy())          # [L, d]
    return np.stack(out)



def probe_acc(X, y, seed=0):
    """5-fold CV accuracy of a logistic regression on standardised inputs."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.1))
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    return float(cross_val_score(clf, X, y, cv=cv, scoring="accuracy").mean())


def base_pair(task, model):
    """(desired, undesired) files for the BASE prompt -- ATP's base_toks."""
    src, base = STEMS[task]
    d = REPOS[OWNER[task]] / "data" / model / f"{src}-long"
    for des, und in ((d / f"{base}-desired-all.jsonl", d / f"{base}-undesired-all.jsonl"),
                     (d / f"{base}-long-desired-all.jsonl", d / f"{base}-long-undesired-all.jsonl")):
        if des.exists() and und.exists():
            return des, und
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--tasks", default="verse,summarization,bias,factual recall,persona")
    ap.add_argument("--k", type=float, default=0.05)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--n_items", type=int, default=60)
    ap.add_argument("--n_control", type=int, default=3)
    a = ap.parse_args()
    sys.path.insert(0, str(AYUSHI))
    from model_handler import ModelHandler
    rows = []
    for model in [m for m in a.models.split(",") if m in MODELS]:
        F = load_fields(model)
        tasks = [t for t in a.tasks.split(",") if (t, "long") in F and (t, "single") in F]
        if not tasks:
            continue
        cfg = SimpleNamespace(args=SimpleNamespace(
            model_id=MODELS[model], device=a.device, pyreft=False,
            full_precision=False, source=STEMS[tasks[0]][0]))
        mh = ModelHandler(cfg); D = mh.dim
        print(f"\n{'='*100}\n{SHORT[model]}  k={a.k}\n{'='*100}", flush=True)
        for task in tasks:
            fdes, fund = base_pair(task, model)
            if not fdes:
                print(f"  {task}: base desired/undesired pair missing -- skipped"); continue
            Wd = writes_atp(mh, full_convs(fdes, mh.tokenizer, a.n_items), a.batch_size)
            Wu = writes_atp(mh, full_convs(fund, mh.tokenizer, a.n_items), a.batch_size)
            W = np.concatenate([Wd, Wu], 0)
            y = np.array([1]*len(Wd) + [0]*len(Wu))
            nu = W.shape[2] // D
            LF = topk_units(F[(task, "long")], a.k); ST = topk_units(F[(task, "single")], a.k)
            grp = {"LF-only": [x for x in LF if x not in set(ST)],
                   "ST-only": [x for x in ST if x not in set(LF)]}
            def gather(units):
                return np.concatenate([W[:, l, D*u:D*(u+1)] for l, u in units], 1)
            print(f"\n  {task}: {len(y)} conversations (same prompt, two responses)  "
                  f"LF-only {len(grp['LF-only'])}, ST-only {len(grp['ST-only'])} units")
            print(f"    {'group':<10}{'probe acc':>11}{'layer-matched control':>24}{'excess':>9}")
            for g, units in grp.items():
                acc = probe_acc(gather(units), y)
                ctl = [probe_acc(gather(layer_matched(units, nu)), y, seed=s)
                       for s in range(a.n_control)]
                rows.append(dict(model=model, task=task, group=g, n_units=len(units),
                                 probe_acc=acc, control=float(np.mean(ctl)),
                                 excess=acc-float(np.mean(ctl))))
                print(f"    {g:<10}{acc:>11.3f}{np.mean(ctl):>18.3f} +-{np.std(ctl):.3f}"
                      f"{acc-np.mean(ctl):>9.3f}", flush=True)
        del mh; torch.cuda.empty_cache()
    if rows:
        with open(HERE/"probe_units.csv","w",newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        print("\nwrote probe_units.csv")


if __name__ == "__main__":
    main()
