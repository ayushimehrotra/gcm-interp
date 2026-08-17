#!/usr/bin/env python3
"""Is the long-form advantage just MORE POSITIONS OF SUPERVISION?

The one algorithmic difference between the two localizations is how many response
tokens ATP's objective integrates over. Measured on the data:

    long-form arm    base responses ~51 words, source ~26 words
    single-token arm base responses   1 word,  source   1 word

ATP (patching.py:48-69) differentiates the summed response log-likelihood

    L = LL(undesired response) - LL(desired response)
    effect[l] = grad(a_l) * (a_l^source - a_l^base),  summed over sequence

so LF's gradient signal is a sum of ~50 token-level terms and ST's is one term.
A sum of 50 noisy terms is a lower-variance estimate of the same quantity than a
sum of 1. That alone could produce every downstream difference we measured -- a
sharper field, an earlier and tighter layer profile, a smaller budget to reach
the behaviour -- without the arms looking at different things at all.

THE EXPERIMENT
--------------
Recompute long-form ATP with the objective truncated to the first T response
tokens, holding data, model and everything else fixed:

    L_T = sum over the first T response tokens of [ logP_undesired - logP_desired ]

for T in {1, 4, 16, all}. Only the number of supervised positions varies, so T=1
is long-form attribution given single-token supervision.

WHY THIS DRIVES THE REPO'S OWN CODE
-----------------------------------
An earlier version reimplemented ATP independently and failed validation: at
T=all it reproduced the on-disk long-form field with Jaccard 0.027 and a mean
layer of 39.5 against the real 23.8, stable across sample sizes, so the gap was
systematic rather than noise. Rather than keep hunting for the missing detail,
this version constructs the repo's real Config / ModelHandler / DataHandler /
BatchHandler / Patching and calls Patching.apply_patching() unchanged. The ONLY
modification is PatchingUtils.get_response_logits, wrapped to sum over the first
T response tokens instead of all of them; with T=None it delegates to the
original, so the untruncated path is the pipeline's own code byte for byte.
Alignment, response-start detection, padding, head reduction and averaging are
all the pipeline's, so the replication is exact by construction.

The reduction after patching mirrors eval/logits_handler.py:56-67 and
get_top_k_layer_and_head exactly:

    per example  net_effects (n_layers, batch, hidden) -> squeeze -> (l, hidden, 1)
    concatenate over examples along the last axis      -> (l, hidden, n)
    einops.reduce 'l (n m) b -> l n b' with sum        -> (l, n_heads, n)
    mean over the example axis                         -> (l, n_heads)

VALIDATION GATE
---------------
The T=all field is compared against the on-disk long-form CSV before anything is
reported. If Jaccard there is below --min_jaccard the replication is wrong and
the truncation results would be meaningless, so the cell is withheld rather than
reported.

MEASURES, per (model, task) and per T
-------------------------------------
  capture@k    share of total positive attribution in the top k, on the signed
               order the pipeline selects with (eval/logits_handler.py:92)
  mean layer   depth of the selected blocks
  jac_vs_LF    Jaccard against the real long-form field (validation at T=all)
  jac_vs_ST    Jaccard against the real single-token field -- the quantity that
               decides the question
  rho_vs_LF    Spearman against the real long-form field over all blocks
"""
import argparse
import csv
import re
import statistics as st
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import os

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
FROM_RE = re.compile(
    r"^from_(?P<src>.+?)-(?P<loc>long|single)_to_(?P<base>.+?)(?P<old>_old)?$")

TRUNC = [None]          # current truncation, read by the patched objective


# ------------------------------------------------------- on-disk reference fields
def _read_field(path):
    rows = [(int(r["layer"]), int(r["neuron"]), float(r["value"]))
            for r in csv.DictReader(open(path))]
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


def _git_date(root, path):
    out = subprocess.run(["git", "-C", str(root), "log", "-1", "--format=%ad",
                          "--date=short", "--", str(Path(path).relative_to(root))],
                         capture_output=True, text=True).stdout.strip()
    return out or "0000-00-00"


def load_fields(model):
    found = defaultdict(dict)
    for repo, root in REPOS.items():
        for p in root.rglob("numerator_1_targeted_1.0.csv"):
            parts = p.parts
            frm = next((c for c in parts
                        if c.startswith("from_") and "_to_" in c), None)
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


def top_units(A, frac):
    """Signed descending -- eval/logits_handler.py:92 uses flat.topk."""
    n = max(1, int(round(frac * A.size)))
    idx = np.argsort(-A, axis=None, kind="stable")[:n]
    return {(int(i // A.shape[1]), int(i % A.shape[1])) for i in idx}


def capture(A, frac):
    a = np.maximum(A, 0.0).ravel()
    tot = a.sum()
    if tot <= 0:
        return float("nan")
    n = max(1, int(round(frac * a.size)))
    return float(np.sort(a)[::-1][:n].sum() / tot)


def rank(x):
    x = np.asarray(x, float)
    o = x.argsort()
    r = np.empty(len(x), float)
    r[o] = np.arange(len(x), dtype=float)
    for v in np.unique(x):
        m = x == v
        if m.sum() > 1:
            r[m] = r[m].mean()
    return r


def spearman(a, b):
    x, y = rank(np.asarray(a).ravel()), rank(np.asarray(b).ravel())
    return float(np.corrcoef(x, y)[0, 1]) if x.std() and y.std() else float("nan")


def jac(a, b):
    return len(a & b) / max(1, len(a | b))


# ------------------------------------------------------------ the one modification
def install_truncated_objective():
    """Wrap PatchingUtils.get_response_logits to sum over the first T tokens."""
    import torch.nn.functional as F
    from patching_utils import PatchingUtils
    original = PatchingUtils.get_response_logits

    def truncated(self, toks, resp_start_positions, logits, retain_grad=False):
        T = TRUNC[0]
        if T is None:
            return original(self, toks, resp_start_positions, logits, retain_grad)
        log_probs = F.log_softmax(logits, dim=-1)
        ids = toks["input_ids"]
        out = []
        for i, rsp in enumerate(resp_start_positions):
            rsp = int(rsp)
            last = log_probs.shape[1] - 1
            end = min(rsp + T, last)
            if end <= rsp:
                end = min(rsp + 1, last)
            tgt = ids[i, rsp + 1:end + 1].to(log_probs.device)
            sel = log_probs[i, rsp:end, :].gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
            out.append(sel.sum())
        return torch.stack(out)

    PatchingUtils.get_response_logits = truncated


# ---------------------------------------------------------------- run the pipeline
def base_arg(task):
    """args.base for this task's repo naming convention."""
    _, base = STEMS[task]
    return base if OWNER[task] == "ayushi" else f"{base}-long"


def build(source, base, model_id, device, n_items):
    """Construct the repo's objects exactly as run.py does."""
    from config import Config
    from model_handler import ModelHandler
    from data_handler import DataHandler
    from batch_handler import BatchHandler
    from patching import Patching

    argv = ["run.py", "--model_id", model_id, "--batch_size", "1",
            "--patch_algo", "atp", "--source", source, "--base", base,
            "--device", device, "--patch_model"]
    old = sys.argv
    sys.argv = argv
    try:
        config = Config()
    finally:
        sys.argv = old

    mh = ModelHandler(config)
    config.args.batch_size = 5
    dh = DataHandler(config, mh)
    config.args.batch_size = 1
    dh.truncate_to_len(min(dh.LEN, n_items))
    bh = BatchHandler(config, dh, 0, min(1, dh.LEN))
    return mh, dh, bh, Patching(mh, bh, config)


def atp_field(mh, dh, bh, patching, T):
    """The repo's own ATP with the objective truncated to T response tokens."""
    import einops
    TRUNC[0] = T
    chunks = []
    try:
        for idx in range(0, dh.LEN, 1):
            bh.update(idx, min(idx + 1, dh.LEN))
            ne = patching.apply_patching()            # (n_layers, batch, hidden)
            chunks.append(ne.squeeze().unsqueeze(-1))  # logits_handler.py:56
    finally:
        TRUNC[0] = None
    all_logits = torch.cat(chunks, dim=-1)             # (l, hidden, n_examples)
    heads = einops.reduce(all_logits, "l (n m) b -> l n b", "sum",
                          n=mh.num_heads)              # logits_handler.py:67
    return heads.to(torch.float32).mean(dim=-1).numpy()   # mean over examples


def boot(d, n=10000, seed=0):
    if len(d) < 2:
        return float("nan"), float("nan")
    r = np.random.default_rng(seed)
    m = np.sort(r.choice(np.asarray(d, float), (n, len(d))).mean(1))
    return float(m[int(.025 * n)]), float(m[int(.975 * n)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--tasks", default="verse,summarization,bias,factual recall,persona")
    ap.add_argument("--Ts", default="1,4,16,0", help="0 = all response tokens")
    ap.add_argument("--k", type=float, default=0.05)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n_items", type=int, default=32)
    ap.add_argument("--min_jaccard", type=float, default=0.5)
    ap.add_argument("--repo", default="ayushi", choices=["ayushi", "umang"],
                    help="which checkout to drive; data_path is cwd-relative")
    ap.add_argument("--out", default="position_count.csv")
    a = ap.parse_args()
    root = REPOS[a.repo]
    os.chdir(root)                      # config.py:18 builds ./data/{model}/
    sys.path.insert(0, str(root))
    install_truncated_objective()

    Ts = [None if int(x) == 0 else int(x) for x in a.Ts.split(",")]
    if None not in Ts:
        Ts.append(None)
    rows = []
    for model in [m for m in a.models.split(",") if m in MODELS]:
        F = load_fields(model)
        for task in [t for t in a.tasks.split(",")
                     if (t, "long") in F and (t, "single") in F
                     and OWNER[t] == a.repo]:
            src, base = STEMS[task]
            L_ref, S_ref = F[(task, "long")], F[(task, "single")]
            uL, uS = top_units(L_ref, a.k), top_units(S_ref, a.k)
            print(f"\n{'='*104}\n{SHORT[model]}  {task}   "
                  f"source={src}-long  base={base_arg(task)}  repo={a.repo}"
                  f"\n{'='*104}", flush=True)
            try:
                mh, dh, bh, patching = build(f"{src}-long", base_arg(task),
                                             MODELS[model], a.device, a.n_items)
            except Exception as e:
                print(f"  setup FAILED {type(e).__name__}: {e}", flush=True)
                continue
            fields = {}
            for T in Ts:
                try:
                    fields[T] = atp_field(mh, dh, bh, patching, T)
                except Exception as e:
                    print(f"  T={T}: FAILED {type(e).__name__}: {e}", flush=True)
            del patching, bh, dh, mh
            torch.cuda.empty_cache()

            if None not in fields:
                print("  no T=all field -- cannot validate, cell skipped")
                continue
            uAll = top_units(fields[None], a.k)
            vj, vr = jac(uAll, uL), spearman(fields[None], L_ref)
            print(f"  VALIDATION  T=all vs disk long-form:  "
                  f"Jaccard {vj:.3f}   Spearman {vr:+.3f}", flush=True)
            if vj < a.min_jaccard:
                print(f"  ** does not reproduce the pipeline (need >= "
                      f"{a.min_jaccard}) -- results withheld **", flush=True)
                continue
            print(f"\n  {'T':>6}{'capture@k':>12}{'mean layer':>12}"
                  f"{'jac vs LF':>12}{'jac vs ST':>12}{'rho vs LF':>12}")
            for T in Ts:
                if T not in fields:
                    continue
                A = fields[T]
                uT = top_units(A, a.k)
                ml = float(np.mean([l for l, _ in uT]))
                rows.append(dict(model=model, task=task,
                                 T=("all" if T is None else T),
                                 capture=capture(A, a.k), mean_layer=ml,
                                 jac_vs_LF=jac(uT, uL), jac_vs_ST=jac(uT, uS),
                                 rho_vs_LF=spearman(A, L_ref), val_jaccard=vj))
                print(f"  {str('all' if T is None else T):>6}"
                      f"{capture(A, a.k):>12.4f}{ml:>12.1f}{jac(uT, uL):>12.3f}"
                      f"{jac(uT, uS):>12.3f}{spearman(A, L_ref):>+12.3f}",
                      flush=True)
            print(f"  {'disk':>6}{capture(L_ref, a.k):>12.4f}"
                  f"{np.mean([l for l, _ in uL]):>12.1f}{1.0:>12.3f}"
                  f"{jac(uL, uS):>12.3f}{1.0:>+12.3f}  (real LF)")
            print(f"  {'disk':>6}{capture(S_ref, a.k):>12.4f}"
                  f"{np.mean([l for l, _ in uS]):>12.1f}{jac(uL, uS):>12.3f}"
                  f"{1.0:>12.3f}{spearman(S_ref, L_ref):>+12.3f}  (real ST)")

    if not rows:
        print("\nno validated cells -- nothing to report")
        return
    with open(HERE / a.out, "w", newline="") as f:  # HERE is absolute
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print("\n" + "=" * 104)
    print("DOES TRUNCATING THE OBJECTIVE MOVE LONG-FORM TOWARD SINGLE-TOKEN?")
    print("=" * 104)
    idx = defaultdict(dict)
    for r in rows:
        idx[(r["model"], r["task"])][r["T"]] = r
    print(f"  {'T':>6}{'n':>5}{'capture':>11}{'mean layer':>13}"
          f"{'jac vs LF':>12}{'jac vs ST':>12}{'rho vs LF':>12}")
    for T in [("all" if t is None else t) for t in Ts]:
        v = [d[T] for d in idx.values() if T in d]
        if not v:
            continue
        print(f"  {str(T):>6}{len(v):>5}{st.mean(x['capture'] for x in v):>11.4f}"
              f"{st.mean(x['mean_layer'] for x in v):>13.1f}"
              f"{st.mean(x['jac_vs_LF'] for x in v):>12.3f}"
              f"{st.mean(x['jac_vs_ST'] for x in v):>12.3f}"
              f"{st.mean(x['rho_vs_LF'] for x in v):>+12.3f}")
    both = [c for c in idx.values() if 1 in c and "all" in c]
    if both:
        d = [c[1]["jac_vs_ST"] - c["all"]["jac_vs_ST"] for c in both]
        lo, hi = boot(d)
        pos = sum(1 for x in d if x > 0)
        print(f"\n  T=1 moves toward real ST by {st.mean(d):+.4f} Jaccard "
              f"[{lo:+.4f},{hi:+.4f}]   in {pos}/{len(d)} cells")
        dc = [c["all"]["capture"] - c[1]["capture"] for c in both]
        lo2, hi2 = boot(dc)
        print(f"  truncating costs {st.mean(dc):+.4f} capture [{lo2:+.4f},{hi2:+.4f}]"
              f"   (positive = T=all more concentrated)")
        dl = [c[1]["mean_layer"] - c["all"]["mean_layer"] for c in both]
        print(f"  and moves the mean layer {st.mean(dl):+.2f} "
              f"(real ST sits later than real LF by +3.5)")
    print("\n  if capture falls and jac-vs-ST rises as T -> 1, position count is")
    print("  the mechanism. if T=1 still looks like T=all, it is not.")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
