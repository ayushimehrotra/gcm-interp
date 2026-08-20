#!/usr/bin/env python3
"""SVD-based comparison of the two localizations, across three tensor shapings.

WHY THIS EXISTS
---------------
`analysis/activation_geometry.py` reports that the free-form and single-token
localizations select activations that are near-linearly-equivalent, even though
the two arms share only ~6% of their units. That was measured ONE way: every
(prompt, position) pair as a row, SVCCA at rank r=24.

Three things about that were unexamined, and this script exists to examine them.

1. THE SHAPING. Activations come out of the model as [B, S, H] -- prompts,
   response positions, features. Flattening to a matrix is a CHOICE, and it
   decides what counts as one observation:

       (1) bs_h   (B*S, H)    a sample is one (prompt, token)   n = B*S
       (2) s_h    (S, H)      a sample is a position, averaged  n = S
                              over prompts, within polarity
       (3) b_sh   (B, S*H)    a sample is a whole prompt        n = B

   These are not cosmetic. n is what sets the null floor of every alignment
   statistic, and the three differ by an order of magnitude. Shaping (1) has the
   most rows but they are NOT independent -- 12 tokens from one response are
   near-duplicate activations, so nominal n badly overstates the evidence.
   Shaping (3) has honestly independent rows (prompts really are independent)
   but far fewer of them. Shaping (2) asks a different question entirely: about
   the average positional trajectory, not the concept contrast.

   If the claim is "the input activations differ in a way that causes the
   outputs to differ", the output is per-PROMPT, so shaping (3) is the one that
   supports it. `--extract last_prompt` goes further and takes a single
   activation per prompt at the last prompt token -- the pre-generation state,
   and the same extraction point `eval/activations.py:steering_reps_cache` uses
   to build the actual steering vectors. That is the causally relevant moment:
   activations sampled DURING the response are partly a consequence of the
   output, not purely an input to it.

2. THE TRUNCATION. Plain CCA is degenerate here and it is worth being blunt
   about why. X_L is [768 x 9120]: 9120 columns living in a 768-dimensional
   sample space, so its column space is ALL of R^767. So is X_S's. Two
   subspaces that are both "everything" are identical, and every canonical
   correlation is exactly 1 -- measured on the real gemma verse data:

       r=24    rho 0.9858   floor 0.2850   excess +0.7008
       r=384   rho 1.0000   floor 0.9999   excess +0.0000
       r=767   rho 1.0000   floor 1.0000   excess +0.0000

   So truncation is not denoising, it is what makes the question exist. But the
   choice of r is unprincipled, so this script also provides measures that do
   not need one:

       cka     linear CKA. NO rank parameter. Weights every direction by its
               variance instead of hard-cutting at r.
       pwcca   canonical correlations weighted by how much variance each
               canonical direction actually explains. Still takes r, but is far
               less sensitive to it than a top-5 mean.
       svcca   the original: mean of the top-5 canonical correlations at rank r.

   Empirically on gemma verse, CKA excess is +0.733 against SVCCA's +0.70 at
   r=24, so the result does not hinge on the truncation. PWCCA moves 0.75 ->
   0.51 across r=8..128 where plain SVCCA moves 0.83 -> 0.00.

3. THE FLOOR. Every one of these statistics has a null that is NOT zero and
   that MOVES with n and r. It is measured here the only reliable way: shuffle
   the rows of one side to destroy the pairing while preserving each side's own
   covariance, and recompute. Only `excess = observed - floor` is evidence.
   The floor is large: 0.313 of SVCCA's 0.969 at the published setting.

DEGENERACY GUARD
----------------
Two r-dimensional subspaces of an (n-1)-dimensional space are FORCED to
intersect once 2r >= n-1, which pins the leading cosines at 1 regardless of the
data. In the original 1320-point sweep every single non-positive excess sat
inside that region and none outside it. Points violating the rule are skipped
and logged rather than reported as measurements.

SELF-CONTAINED
--------------
Only `model_handler.py` (this repo) and the usual runtime deps are needed. Paths
resolve relative to this file. All 22 of the paper's cells run from this repo.

USAGE
-----
    python svcca_sweep.py --dry_run                     # list cells, no GPU
    python svcca_sweep.py                               # all shapings + measures
    python svcca_sweep.py --shapings b_sh               # prompt-level only
    python svcca_sweep.py --extract last_prompt         # pre-generation state
    python svcca_sweep.py --measures cka                # truncation-free only
    python svcca_sweep.py --models gemma-3-12b-it --tasks verse
    python svcca_sweep.py --dump_spectra                # raw singular values too
"""
import argparse
import csv
import json
import re
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO = Path(__file__).resolve().parent

# ---------------------------------------------------------------- the grid
TASKNAME = {"verse": "verse", "paragraph": "summarization",
            "extraversion": "persona", "female": "bias", "lying": "factual recall"}
STEMS = {"verse": "verse", "summarization": "paragraph", "persona": "extraversion",
         "bias": "female", "factual recall": "lying"}
MODELS = {"gemma-3-12b-it": "google/gemma-3-12b-it",
          "Qwen1.5-14B-Chat": "Qwen/Qwen1.5-14B-Chat",
          "Falcon3-10B-Instruct": "tiiuae/Falcon3-10B-Instruct",
          "OLMo-2-1124-13B-DPO": "allenai/OLMo-2-1124-13B-DPO",
          "Qwen1.5-32B-Chat": "Qwen/Qwen1.5-32B-Chat"}
FROM_RE = re.compile(
    r"^from_(?P<src>.+?)-(?P<loc>long|single)_to_(?P<base>.+?)(?P<old>_old)?$")

SHAPINGS = ("bs_h", "s_h", "b_sh")
MEASURES = ("cka", "svcca", "pwcca")


# ------------------------------------------------------- attribution fields
def read_field(path):
    """(layer, neuron, value) csv -> dense [n_layers, n_units] signed field."""
    rows = [(int(r["layer"]), int(r["neuron"]), float(r["value"]))
            for r in csv.DictReader(open(path))]
    if not rows:
        return None
    nl = max(r[0] for r in rows) + 1
    nu = max(r[1] for r in rows) + 1
    if len(rows) != nl * nu:          # partial file: refuse rather than guess
        return None
    a = np.zeros((nl, nu))
    for l, u, v in rows:
        a[l, u] = v
    return a


RESULT_DIRS = ("results", "results_with_answers")


def load_fields(model, roots):
    """(task, arm) -> field, searching `roots` in order (first match wins)."""
    out = {}
    for root in roots:
        for sub in RESULT_DIRS:
            res = root / sub
            if not res.is_dir():
                continue
            for p in res.rglob("numerator_1_targeted_1.0.csv"):
                parts = p.parts
                frm = next((c for c in parts
                            if c.startswith("from_") and "_to_" in c), None)
                if frm is None:
                    continue
                m = FROM_RE.match(frm)
                if not m or m.group("old"):
                    continue
                if parts[parts.index(frm) - 1] != model:
                    continue
                task = TASKNAME.get(m.group("src"))
                if task is None or (task, m.group("loc")) in out:
                    continue
                f = read_field(p)
                if f is not None:
                    out[(task, m.group("loc"))] = f
    return out


def find_data(model, task, roots):
    """desired/undesired jsonl for a cell, or (None, None)."""
    src = STEMS[task]
    for root in roots:
        d = root / "data" / model / f"{src}-long"
        fd = d / f"{src}-long-desired-all.jsonl"
        fu = d / f"{src}-long-undesired-all.jsonl"
        if fd.exists() and fu.exists():
            return fd, fu
    return None, None


def top_units(A, frac):
    """Signed descending, matching eval/logits_handler.py:92 (flat.topk).

    Ranking by |value| instead gives a set overlapping the pipeline's by only
    Jaccard 0.43 -- a different experiment from the one the paper runs.
    """
    n = max(1, int(round(frac * A.size)))
    idx = np.argsort(-A, axis=None, kind="stable")[:n]
    return [(int(i // A.shape[1]), int(i % A.shape[1])) for i in idx]


# ------------------------------------------------------------------ geometry
def _centre(M):
    return M - M.mean(0, keepdims=True)


def _basis(M, r):
    """Orthonormal basis of the top-r principal subspace, in SAMPLE space.

    Any orthonormal basis of the column space gives identical canonical
    correlations (verified: QR vs SVD agree to 2.6e-15). SVD is used only
    because it ALSO orders directions by variance, which is what makes the
    top-r truncation meaningful.
    """
    M = _centre(M)
    U, s, _ = np.linalg.svd(M, full_matrices=False)
    keep = min(r, int((s > 1e-8 * max(s[0], 1e-30)).sum()))
    return U[:, :keep]


def canonical_correlations(X, Y, r):
    """Cosines of the principal angles between the two rank-r subspaces.

    These are exactly the canonical correlations of CCA, and exactly the
    singular values of the cosine-similarity matrix between the two arms'
    left singular vectors (verified against scipy.linalg.subspace_angles
    to 4.3e-15). Returns None if either side collapses.
    """
    Qx, Qy = _basis(X, r), _basis(Y, r)
    if Qx.shape[1] == 0 or Qy.shape[1] == 0:
        return None
    rho = np.linalg.svd(Qx.T @ Qy, compute_uv=False)
    return np.clip(rho, 0, 1)


def m_svcca(X, Y, r, ntop=5):
    """Mean of the top-`ntop` canonical correlations. The original measure.

    NOTE min(ntop, len(rho)): at r=2 this averages 2 values and at r=4 four, so
    the statistic is not identical across the low-rank end of an r sweep.
    """
    rho = canonical_correlations(X, Y, r)
    if rho is None or len(rho) == 0:
        return None
    return float(np.mean(rho[:min(ntop, len(rho))]))


def m_pwcca(X, Y, r):
    """Projection-weighted CCA (Morcos et al. 2018).

    Weights each canonical correlation by how much of X it actually accounts
    for, so low-variance directions stop dominating the average. Much less
    sensitive to r than a flat top-5 mean.
    """
    Qx, Qy = _basis(X, r), _basis(Y, r)
    if Qx.shape[1] == 0 or Qy.shape[1] == 0:
        return None
    W, rho, _ = np.linalg.svd(Qx.T @ Qy)
    rho = np.clip(rho, 0, 1)
    H = Qx @ W                                   # canonical variates, sample space
    alpha = np.abs(H.T @ _centre(X)).sum(1)
    tot = alpha.sum()
    if tot <= 0:
        return None
    return float((alpha / tot) @ rho)


def m_cka(X, Y, r=None):
    """Linear centred kernel alignment. NO rank parameter -- nothing truncated.

    Every direction contributes, weighted by its variance, rather than being
    hard-cut at r. This is the measure to reach for when you do not want to
    justify a choice of rank.
    """
    A, B = _centre(X), _centre(Y)
    da = np.linalg.norm(A.T @ A, "fro")
    db = np.linalg.norm(B.T @ B, "fro")
    if da <= 0 or db <= 0:
        return None
    return float(np.linalg.norm(B.T @ A, "fro") ** 2 / (da * db))


MEASURE_FN = {"cka": m_cka, "svcca": m_svcca, "pwcca": m_pwcca}
NEEDS_RANK = {"cka": False, "svcca": True, "pwcca": True}


def with_floor(fn, X, Y, r, n_perm, seed=0):
    """(observed, floor, excess). Floor = same statistic with rows shuffled.

    Shuffling one side destroys the row pairing while preserving each side's own
    covariance and spectrum, so whatever survives is what the estimator
    manufactures from dimension alone. Only the excess is evidence.
    """
    obs = fn(X, Y, r) if NEEDS_RANK[fn_name(fn)] else fn(X, Y)
    if obs is None:
        return None
    rng = np.random.default_rng(seed)
    perm = []
    for _ in range(n_perm):
        Yp = Y[rng.permutation(len(Y))]
        v = fn(X, Yp, r) if NEEDS_RANK[fn_name(fn)] else fn(X, Yp)
        if v is not None:
            perm.append(v)
    if not perm:
        return None
    floor = float(np.mean(perm))
    return float(obs), floor, float(obs) - floor


def fn_name(fn):
    for k, v in MEASURE_FN.items():
        if v is fn:
            return k
    raise KeyError(fn)


def degenerate(r, n):
    """Two r-dim subspaces of an (n-1)-dim space MUST intersect when 2r >= n-1.

    Inside that region the leading cosines are pinned at 1 on any data, real or
    shuffled, so excess is identically 0. Verified on the 1320-point sweep:
    all 132 non-positive points were inside, zero of 1122 outside.
    """
    return 2 * r >= n - 1


def structure(X, y=None):
    """Describe one arm's subspace on its own terms -- no cross-arm comparison.

    participation ratio is the effective dimensionality: gemma verse measures
    PR ~ 28 out of 9120 coordinates, which is independent justification for a
    rank in the twenties and explains why excess peaks around r=8.
    """
    M = _centre(X)
    s = np.linalg.svd(M, compute_uv=False)
    v = s ** 2
    tot = v.sum()
    if tot <= 0:
        return {}
    p = v / tot
    pr = float(v.sum() ** 2 / (v ** 2).sum())
    cum = np.cumsum(p)
    out = dict(pr=round(pr, 3), pr_frac=round(pr / M.shape[1], 6),
               dims50=int(np.searchsorted(cum, 0.50) + 1),
               dims90=int(np.searchsorted(cum, 0.90) + 1),
               top1_var=round(float(p[0]), 6))
    if y is not None and len(np.unique(y)) > 1:
        U = np.linalg.svd(M, full_matrices=False)[0]
        k = min(5, U.shape[1])
        out["label_corr"] = round(max(
            abs(float(np.corrcoef(U[:, i], y)[0, 1])) for i in range(k)), 6)
        w = M[y > 0].mean(0) - M[y < 0].mean(0)
        nw = np.linalg.norm(w)
        if nw > 0:
            out["class_var_frac"] = round(
                float((M @ (w / nw)).var() / M.var(0).sum()), 6)
    return out


# ------------------------------------------------------------------ sampling
def prep(mh, path, limit, torch):
    """Tokenize prompts and locate each one's response-start index."""
    rows = [json.loads(l) for l in open(path)][:limit]
    text = [mh.tokenizer.apply_chat_template(r["prompt"],
                                             add_generation_prompt=False,
                                             tokenize=False) for r in rows]
    toks = mh.tokenizer(text, padding=True, truncation=False, return_tensors="pt")
    marker = mh.alignment_tokens
    starts = []
    for ids in toks["input_ids"]:
        pos = None
        for j in range(ids.size(0) - marker.size(0) + 1):
            if torch.equal(ids[j:j + marker.size(0)], marker):
                pos = j + marker.size(0)
        starts.append(pos if pos is not None else 0)
    return toks, torch.tensor(starts)


def collect_structured(mh, toks, starts, blocks, D, n_pos, bs, torch,
                       extract="response"):
    """Activations with the [B, S, H] structure KEPT, so it can be reshaped.

    extract="response"    : n_pos positions spread across the response, the
                            positions ATP itself differentiates.
    extract="last_prompt" : the single token immediately before the response
                            begins -- the pre-generation state, and the same
                            moment eval/activations.py builds steering vectors
                            from. S == 1, so all three shapings coincide.

    Prompts that cannot supply the full n_pos positions are DROPPED rather than
    padded, so the returned array is exactly [B_kept, S, H] and every prompt
    carries equal weight. The count dropped is returned for logging: a ragged
    collection silently reweights prompts, and min(len(Xd), len(Xu)) later
    truncates by array position, which would drop whole prompts off one side.
    """
    want = 1 if extract == "last_prompt" else n_pos
    rows, dropped = [], 0
    with torch.no_grad():
        for i in range(0, toks["input_ids"].shape[0], bs):
            sl = slice(i, i + bs)
            batch = {k: v[sl].to(mh.device) for k, v in toks.items()}
            with mh.model.trace(batch):
                saved = [l.self_attn.o_proj.output.detach().cpu().save()
                         for l in mh.model.model.layers]
            A = torch.stack([s.to(torch.float32) for s in saved])
            am = toks["attention_mask"][sl]
            for j, s_ in enumerate(starts[sl]):
                if extract == "last_prompt":
                    t = int(s_) - 1
                    take = [t] if t >= 0 and am[j, t] == 1 else []
                else:
                    valid = [t for t in range(int(s_), am.shape[1])
                             if am[j, t] == 1]
                    if len(valid) < want:
                        take = []
                    else:
                        take = sorted({valid[int(round(x))] for x in
                                       np.linspace(0, len(valid) - 1, want)})
                        if len(take) < want:      # collisions from rounding
                            take = valid[:want]
                if len(take) != want:
                    dropped += 1
                    continue
                per_pos = []
                for t in take:
                    v = A[:, j, t, :]
                    per_pos.append(torch.cat(
                        [v[l, D * u:D * (u + 1)] for l, u in blocks]).numpy())
                rows.append(np.stack(per_pos))       # [S, H]
            del A, saved
    if not rows:
        return None, dropped
    return np.stack(rows), dropped                    # [B, S, H]


def shape_matrix(T, shaping, n_items):
    """[B, S, H] -> a 2-D matrix, one of the three shapings. T is [des; und]."""
    B, S, H = T.shape
    if shaping == "bs_h":
        return T.reshape(B * S, H)
    if shaping == "b_sh":
        return T.reshape(B, S * H)
    if shaping == "s_h":
        # average over prompts WITHIN each polarity, so the contrast survives
        return np.concatenate([T[:n_items].mean(0), T[n_items:].mean(0)])
    raise ValueError(shaping)


def shape_labels(shaping, B, S, n_items):
    """+1 for desired rows, -1 for undesired, matching shape_matrix's order."""
    if shaping == "bs_h":
        return np.r_[np.ones(n_items * S), -np.ones((B - n_items) * S)]
    if shaping == "b_sh":
        return np.r_[np.ones(n_items), -np.ones(B - n_items)]
    if shaping == "s_h":
        return np.r_[np.ones(S), -np.ones(S)]
    raise ValueError(shaping)


def balanced_subsample(X, y, n_total, rng):
    """Take n_total rows, half from each polarity, preserving the pairing index."""
    d = np.flatnonzero(y > 0)
    u = np.flatnonzero(y < 0)
    per = n_total // 2
    if per > min(len(d), len(u)):
        return None
    return np.concatenate([rng.choice(d, per, replace=False),
                           rng.choice(u, per, replace=False)])


# ---------------------------------------------------------------------- main
def runnable_cells(models, tasks, roots):
    """Cells with both arms' fields AND both data files, without loading a model."""
    out = []
    for model in models:
        F = load_fields(model, roots)
        for task in tasks:
            if (task, "long") not in F or (task, "single") not in F:
                continue
            fd, fu = find_data(model, task, roots)
            if fd is None:
                continue
            out.append((model, task, F, fd, fu))
    return out


def main():
    ap = argparse.ArgumentParser(
        description="SVD-based comparison of the two localizations, three shapings")
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--tasks",
                    default="verse,summarization,persona,bias,factual recall")
    ap.add_argument("--k", type=float, default=0.05,
                    help="localization budget; 0.05 matches activation_geometry.py")
    ap.add_argument("--shapings", default=",".join(SHAPINGS),
                    help="bs_h=(B*S,H)  s_h=(S,H) prompt-averaged  b_sh=(B,S*H)")
    ap.add_argument("--measures", default=",".join(MEASURES),
                    help="cka (no rank needed), svcca, pwcca")
    ap.add_argument("--extract", default="response",
                    choices=("response", "last_prompt"),
                    help="last_prompt = pre-generation state, S=1, one row/prompt")
    ap.add_argument("--ranks", default="2,4,8,16,24,32,48,64,96,128")
    ap.add_argument("--n_grid", default="",
                    help="optional row-subsample sweep; blank = use all rows")
    ap.add_argument("--n_items", type=int, default=50,
                    help="prompts per polarity (files hold 100)")
    ap.add_argument("--n_pos", type=int, default=12,
                    help="response positions per prompt; capped by the shortest "
                         "response in the grid (16 tokens on gemma)")
    ap.add_argument("--n_perm", type=int, default=10)
    ap.add_argument("--n_boot", type=int, default=3,
                    help="subsample repeats per n (averaged); only used with --n_grid")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--extra_repo", default=None)
    ap.add_argument("--out", default="svcca_sweep.csv")
    ap.add_argument("--struct_out", default="svcca_structure.csv",
                    help="per-arm subspace description (PR, dims50, label corr)")
    ap.add_argument("--dump_spectra", action="store_true",
                    help="also write every canonical-correlation spectrum")
    ap.add_argument("--spectra_out", default="svcca_spectra.csv")
    ap.add_argument("--dry_run", action="store_true")
    a = ap.parse_args()

    roots = [REPO] + ([Path(a.extra_repo).resolve()] if a.extra_repo else [])
    models = [m for m in a.models.split(",") if m in MODELS]
    tasks = [t.strip() for t in a.tasks.split(",")]
    ranks = [int(x) for x in a.ranks.split(",")]
    shapings = [s.strip() for s in a.shapings.split(",") if s.strip() in SHAPINGS]
    measures = [m.strip() for m in a.measures.split(",") if m.strip() in MEASURES]
    ngrid = sorted(int(x) for x in a.n_grid.split(",")) if a.n_grid.strip() else []

    if a.extract == "last_prompt":
        print("extract=last_prompt -> S=1; the three shapings coincide at (B, H)")
        shapings = ["b_sh"]

    print(f"repo roots : {[str(r) for r in roots]}")
    print(f"shapings   : {shapings}")
    print(f"measures   : {measures}   (cka takes no rank)")
    print(f"ranks      : {ranks}")
    print(f"extract    : {a.extract}   n_items={a.n_items}  n_pos={a.n_pos}")
    cells = runnable_cells(models, tasks, roots)
    print(f"\n{len(cells)} runnable cells:")
    for model, task, _, _, _ in cells:
        print(f"    {model:<24}{task}")
    missing = [(m, t) for m in models for t in tasks
               if not any(c[0] == m and c[1] == t for c in cells)]
    if missing:
        print(f"\n  not runnable here ({len(missing)}): "
              + ", ".join(f"{m}/{t}" for m, t in missing))
    if a.dry_run:
        return
    if not cells:
        print("\nnothing to run")
        return

    import torch
    sys.path.insert(0, str(REPO))
    from model_handler import ModelHandler

    rows, srows, spec = [], [], []
    by_model = defaultdict(list)
    for c in cells:
        by_model[c[0]].append(c)

    def flush():
        for path, data in ((a.out, rows), (a.struct_out, srows),
                           (a.spectra_out, spec if a.dump_spectra else [])):
            if data:
                with open(REPO / path, "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=list(data[0].keys()))
                    w.writeheader()
                    w.writerows(data)

    for model, cs in by_model.items():
        cfg = SimpleNamespace(args=SimpleNamespace(
            model_id=MODELS[model], device=a.device, pyreft=False,
            full_precision=False, source=STEMS[cs[0][1]]))
        mh = ModelHandler(cfg)
        D = mh.dim
        print(f"\n{'='*100}\n{model}   block width D={D}   k={a.k}\n{'='*100}",
              flush=True)

        for _, task, F, fd, fu in cs:
            L, S_ = F[(task, "long")], F[(task, "single")]
            uL, uS = top_units(L, a.k), top_units(S_, a.k)
            union = sorted(set(uL) | set(uS))
            pos = {b: i for i, b in enumerate(union)}

            td, sd = prep(mh, fd, a.n_items, torch)
            tu, su = prep(mh, fu, a.n_items, torch)
            Td, dd = collect_structured(mh, td, sd, union, D, a.n_pos,
                                        a.batch_size, torch, a.extract)
            Tu, du = collect_structured(mh, tu, su, union, D, a.n_pos,
                                        a.batch_size, torch, a.extract)
            if Td is None or Tu is None:
                print(f"  {task}: no usable positions, skipped")
                continue
            nb = min(len(Td), len(Tu))
            T = np.concatenate([Td[:nb], Tu[:nb]])          # [2*nb, S, H_union]
            B, Spos, _ = T.shape
            print(f"  {task}: [B,S,H] = {T.shape}  "
                  f"({nb} prompts/polarity, {dd+du} dropped short)", flush=True)

            def cols(units):
                return np.concatenate([np.arange(pos[b] * D, (pos[b] + 1) * D)
                                       for b in units])
            cL, cS = cols(uL), cols(uS)

            for shaping in shapings:
                MT = shape_matrix(T, shaping, nb)
                y = shape_labels(shaping, B, Spos, nb)
                XL, XS = MT[:, cL] if shaping != "b_sh" else None, None
                if shaping == "b_sh":
                    # columns are position-major blocks of H; slice each position
                    H = T.shape[2]
                    XL = np.concatenate([MT[:, p * H + cL] for p in range(Spos)], 1)
                    XS = np.concatenate([MT[:, p * H + cS] for p in range(Spos)], 1)
                else:
                    XL, XS = MT[:, cL], MT[:, cS]
                n = len(XL)

                for arm, Xa in (("long", XL), ("single", XS)):
                    d = structure(Xa, y)
                    srows.append(dict(model=model, task=task, shaping=shaping,
                                      arm=arm, n=n, d=Xa.shape[1], **d))
                print(f"    {shaping:<6} n={n:<6} d={XL.shape[1]:<7}"
                      f" PR_long={srows[-2].get('pr','?')}", flush=True)

                subsets = ngrid or [n]
                for nn in subsets:
                    if nn > n:
                        continue
                    reps = 1 if nn == n else a.n_boot
                    for meas in measures:
                        fn = MEASURE_FN[meas]
                        rlist = ranks if NEEDS_RANK[meas] else [0]
                        for r in rlist:
                            if NEEDS_RANK[meas] and degenerate(r, nn):
                                continue
                            acc = []
                            for b in range(reps):
                                rng = np.random.default_rng(1000 * b + 7)
                                if nn == n:
                                    A_, B_ = XL, XS
                                else:
                                    idx = balanced_subsample(XL, y, nn, rng)
                                    if idx is None:
                                        continue
                                    A_, B_ = XL[idx], XS[idx]
                                res = with_floor(fn, A_, B_, r, a.n_perm, seed=b)
                                if res is not None:
                                    acc.append(res)
                            if not acc:
                                continue
                            obs, fl, ex = (float(np.mean([x[i] for x in acc]))
                                           for i in range(3))
                            rows.append(dict(
                                model=model, task=task, shaping=shaping,
                                measure=meas, k=a.k, rank=(r if NEEDS_RANK[meas]
                                                           else ""),
                                n=nn, n_full=n, d_L=XL.shape[1], d_S=XS.shape[1],
                                B=B, S=Spos, extract=a.extract, reps=len(acc),
                                observed=round(obs, 6), floor=round(fl, 6),
                                excess=round(ex, 6)))
                            tag = f"r={r}" if NEEDS_RANK[meas] else "  -"
                            print(f"      {meas:<6}{tag:<7} n={nn:<5}"
                                  f" obs={obs:.4f} floor={fl:.4f} exc={ex:+.4f}",
                                  flush=True)
                            if a.dump_spectra and meas == "svcca":
                                rho = canonical_correlations(XL, XS, r)
                                if rho is not None:
                                    spec.append(dict(
                                        model=model, task=task, shaping=shaping,
                                        rank=r, n=nn,
                                        rho=";".join(f"{v:.6f}" for v in rho)))
            flush()
        del mh
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not rows:
        print("\nno results")
        return
    flush()
    print(f"\nwrote {REPO / a.out}  ({len(rows)} rows)")
    print(f"wrote {REPO / a.struct_out}  ({len(srows)} rows)")
    if a.dump_spectra:
        print(f"wrote {REPO / a.spectra_out}  ({len(spec)} spectra)")

    # ------------------------------------------------------------- summary
    print(f"\n{'='*100}\nEXCESS over the shuffle floor, averaged across cells")
    print("  cka needs no rank, so it is the truncation-free reference")
    print(f"{'='*100}")
    for shaping in shapings:
        print(f"\n  shaping {shaping}")
        for meas in measures:
            sel = [x for x in rows if x["shaping"] == shaping
                   and x["measure"] == meas]
            if not sel:
                continue
            if not NEEDS_RANK[meas]:
                print(f"    {meas:<6}  excess {st.mean(x['excess'] for x in sel):+.3f}"
                      f"   over {len(sel)} points")
                continue
            line = f"    {meas:<6}  "
            for r in ranks:
                v = [x["excess"] for x in sel if x["rank"] == r]
                line += f"r={r}:{st.mean(v):+.3f}  " if v else ""
            print(line)
    neg = [x for x in rows if x["excess"] <= 0]
    print(f"\n  points with excess <= 0: {len(neg)} of {len(rows)}")
    for x in neg[:10]:
        print(f"    {x['model']}/{x['task']} {x['shaping']}/{x['measure']}"
              f" r={x['rank']} n={x['n']}  excess {x['excess']:+.4f}")


if __name__ == "__main__":
    main()
