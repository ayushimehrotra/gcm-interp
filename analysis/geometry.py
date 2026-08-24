#!/usr/bin/env python3
"""Representational geometry of the two localizations -- one entry point.

Consolidates what used to be four scripts. They were always layered rather than
parallel: svcca_cosmat, cosmat_sweep_n and cosmat_compare all consumed
svcca_sweep's loading/shaping/measure core, and svcca_cosmat imported it by name.
Splitting the core from its three consumers meant the shared half was edited in
one file and re-imported in another; here it is defined once.

SUBCOMMANDS
-----------
  sweep     SVD comparison of the two localizations across tensor shapings
            (CKA / SVCCA / PWCCA, with permutation floors). The core analysis.
  cosmat    cosine-similarity matrices between the arms' left singular vectors,
            per shaping and n, plus the singular-value spectra.
  sweep-n   alignment as a function of n, swept across every legal rank.
  compare   side-by-side table of the two extraction points, per shaping and n.

  python geometry.py sweep   --models ... --tasks ...
  python geometry.py cosmat  --models ... --tasks ...
  python geometry.py sweep-n --indir ... --outdir ...
  python geometry.py compare --a ... --b ...

WHAT THE MERGE CHANGED
----------------------
Only three things, all forced:

1. REPO now points at the repo ROOT. All four files set
   `REPO = Path(__file__).resolve().parent`, which was correct while they sat at
   the top level. They were later moved into analysis/, which silently repointed
   REPO at analysis/ -- and since every data lookup is `roots = [REPO]` searching
   `REPO/"results"`, and analysis/results does not exist, the moved copies found
   ZERO runnable cells. Anchoring to the root restores them.

2. SHAPINGS and TASKS are per-subcommand, because they genuinely differed
   (sweep uses b_sh, the cosmat family uses b_last_h, compare spans both), so
   they carry a subcommand prefix instead of one silently shadowing another.

3. INK is the union of the two near-identical palettes that existed: the cosmat
   copy had a "mid" key, the sweep-n copy a "band" key, everything else equal.
   Each consumer still reads only its own key.

Every analysis function is otherwise byte-identical to what it was.
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
import zlib

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm, Normalize
from matplotlib.colors import LinearSegmentedColormap

import matplotlib
matplotlib.use("Agg")


# --------------------------------------------------------------- shared config
# Anchored to the repo ROOT, not to this file's directory -- see the docstring.
REPO = Path(__file__).resolve().parent.parent

# The site/span helpers are the single source of truth for which tensor a run
# reads and what its results directory is called; duplicating either here is how
# the analysis and the pipeline drift apart. Needs REPO on the path, which the
# subcommand mains also do later for ModelHandler.
sys.path.insert(0, str(REPO))
from eval.patch_site import (SITE_IN, SITE_OUT, SITES, attn_proxy,
                             dir_suffix as site_suffix)
from eval.response_span import (SPANS, SPAN_LEGACY, dir_suffix as span_suffix)


def algo_dir_for(site, span, algo="atp"):
    """Results-directory name for a (site, span), matching Config.set_output_prefix."""
    return f"{algo}{site_suffix(site)}{span_suffix(span)}"

SEQ_BLUE = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
            "#3f89e0", "#2a78d6", "#1f6ac2", "#175aa8", "#104281", "#0b2f5c"]
DIV_RED = ["#7a1f1f", "#a02c2c", "#c03636", "#d03b3b", "#e34948", "#e87070",
           "#f0a0a0", "#f7cfcf"]
# union of the two palettes that used to live in svcca_cosmat and cosmat_sweep_n
# ("mid" was the former's key, "band" the latter's; both are kept)
INK = {"light": dict(surface="#fcfcfb", plane="#f9f9f7", primary="#0b0b0b",
                     secondary="#52514e", muted="#898781", grid="#e1e0d9",
                     axis="#c3c2b7", mid="#f0efec", band="#f0efec"),
       "dark": dict(surface="#1a1a19", plane="#0d0d0d", primary="#ffffff",
                    secondary="#c2c0b6", muted="#8a8880", grid="#2f2f2c",
                    axis="#46443f", mid="#232321", band="#232321")}



# ==========================================================================
# svcca_sweep.py
# ==========================================================================

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

SWEEP_SHAPINGS = ("bs_h", "s_h", "b_sh")

MEASURES = ("cka", "svcca", "pwcca")

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

# Two copies of one localization's field correlating at least this much are taken
# to be the same attribution separated by run-to-run numerical noise. Below it,
# they are different results and the run stops rather than picking by rglob order.
FIELD_AGREE_RHO = 0.99

def load_fields(model, roots, algo_dir="atp", steer_dir=None):
    """(task, arm) -> field, searching `roots` in order (first match wins).

    `algo_dir` is the patch_algo DIRECTORY name and it is matched EXACTLY --
    "atp", "atp-o_proj_in", "atp-o_proj_in-respfix". This is not optional
    hygiene. The results tree holds one numerator_1_targeted_1.0.csv per
    (localization, site, span), all with the same filename, so an unfiltered
    rglob returns whichever the filesystem yields first and silently mixes
    sites: a field indexed by o_proj.input units (head_dim wide) paired with a
    block width computed for o_proj.output (hidden//n_heads). On gemma those are
    256 and 240 -- different objects on different axes, and nothing downstream
    would raise.

    Ambiguity within one algo_dir is an error, not a coin flip. The same field is
    written under several {eval}_eval/{steer}_steer subdirectories and should be
    byte-identical everywhere, since all copies are save_top_k(k=1.0) of one
    numerator_1_heads.pt. Measured over the 50 localizations in this tree: 42
    agree exactly, 6 have a single copy, and 2 disagree --

        Qwen1.5-14B-Chat / paragraph-single   pearson 1.0000, top-5% Jaccard 0.975
        Qwen1.5-32B-Chat / verse-single       pearson 0.2127, top-5% Jaccard 0.123

    The first is float formatting and is accepted. The second is two genuinely
    different fields, split by *steer* subdirectory (the same signature
    eval/logits_handler.atp_reference_csv reports for Qwen1.5-14B), which means a
    stale numerator_1_heads.pt was regenerated partway through that cell's eval
    passes. Attribution cannot depend on the steering vector, so one copy is
    simply wrong and there is no way to tell which from the CSVs alone -- the .pt
    files are gitignored and absent. Rather than let rglob order pick, this
    raises and asks for `steer_dir` to name the copy explicitly.
    """
    out, seen = {}, {}
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
                i = parts.index(frm)
                if parts[i - 1] != model:
                    continue
                # the component right after from_*_to_* is the patch_algo dir
                if i + 1 >= len(parts) or parts[i + 1] != algo_dir:
                    continue
                if steer_dir is not None and steer_dir not in parts:
                    continue
                task = TASKNAME.get(m.group("src"))
                if task is None:
                    continue
                key = (task, m.group("loc"))
                f = read_field(p)
                if f is None:
                    continue
                if key in out:
                    if np.array_equal(out[key], f):
                        continue
                    # Gate on CORRELATION, not elementwise closeness. The field is
                    # consumed as a ranking (top_units) and as a set of columns, so
                    # that is the scale agreement has to be judged on. Elementwise
                    # tolerance answers the wrong question: the Qwen1.5-14B copies
                    # have max relative difference 5.2 on near-zero entries yet
                    # rank identically (top-1% Jaccard 1.000, top-5% 0.975) -- two
                    # attribution runs separated by numerical nondeterminism, not
                    # two different results. The Qwen1.5-32B copies correlate 0.21
                    # and share 12% of their top 5%: a different field.
                    rho = float(np.corrcoef(out[key].ravel(), f.ravel())[0, 1])
                    if rho >= FIELD_AGREE_RHO:
                        print(f"    note: {model}/{key[0]}/{key[1]} has two field "
                              f"copies correlating {rho:.4f}; using {seen[key]}")
                        continue
                    raise ValueError(
                        f"{model}/{key} has genuinely different fields under "
                        f"{algo_dir} (pearson {rho:.4f}):\n  {seen[key]}\n  {p}\n"
                        f"All copies are meant to be save_top_k(k=1.0) of one "
                        f"numerator_1_heads.pt, so one is stale. Name the copy to "
                        f"use with --steer_dir <NAME>_steer, or regenerate the "
                        f"attribution for this localization.")
                    continue
                out[key], seen[key] = f, p
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
                       extract="response", site=None):
    """Activations with the [B, S, H] structure KEPT, so it can be reshaped.

    extract="response"    : n_pos positions spread across the response, the
                            positions ATP itself differentiates.
    extract="prompt"      : n_pos positions spread across the INPUT prompt --
                            every token before the response starts. These are
                            purely inputs to the generation, never a consequence
                            of it, which is the property "response" lacks, and
                            unlike "last_prompt" it keeps S == n_pos so the
                            three shapings stay distinct.
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
    _site = site or getattr(mh, "patch_site", SITE_OUT)
    want = 1 if extract == "last_prompt" else n_pos
    rows, dropped = [], 0
    # Which token window each mode draws from. "prompt" is the input-side
    # counterpart of "response": n_pos positions spread across the tokens BEFORE
    # the response begins, so S stays > 1 and the three shapings remain
    # distinct -- which "last_prompt" cannot do, since it pins S = 1. Padding is
    # left-side (model_handler sets padding_side='left'), so the attention mask,
    # not index 0, is what marks where a prompt actually starts.
    with torch.no_grad():
        for i in range(0, toks["input_ids"].shape[0], bs):
            sl = slice(i, i + bs)
            batch = {k: v[sl].to(mh.device) for k, v in toks.items()}
            with mh.model.trace(batch):
                # The activations MUST come from the same tensor the field
                # indexes. o_proj.output is W_O @ concat(z) -- residual
                # coordinates, hidden_size wide; o_proj.input is concat(z),
                # num_heads*head_dim wide, where block u IS head u. Slicing an
                # o_proj.input field's blocks out of o_proj.output activations
                # reads the wrong coordinates and, on models where the widths
                # differ (gemma 4096 vs 3840), silently runs off the end.
                saved = [attn_proxy(l, _site).detach().cpu().save()
                         for l in mh.model.model.layers]
            A = torch.stack([s.to(torch.float32) for s in saved])
            am = toks["attention_mask"][sl]
            for j, s_ in enumerate(starts[sl]):
                if extract == "last_prompt":
                    t = int(s_) - 1
                    take = [t] if t >= 0 and am[j, t] == 1 else []
                else:
                    lo, hi = ((0, int(s_)) if extract == "prompt"
                              else (int(s_), am.shape[1]))
                    valid = [t for t in range(lo, hi) if am[j, t] == 1]
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
    return np.stack(rows), dropped

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

def runnable_cells(models, tasks, roots, algo_dir="atp", steer_dir=None):
    """Cells with both arms' fields AND both data files, without loading a model."""
    out = []
    for model in models:
        F = load_fields(model, roots, algo_dir, steer_dir)
        for task in tasks:
            if (task, "long") not in F or (task, "single") not in F:
                continue
            fd, fu = find_data(model, task, roots)
            if fd is None:
                continue
            out.append((model, task, F, fd, fu))
    return out

def _sweep_main():
    ap = argparse.ArgumentParser(
        description="SVD-based comparison of the two localizations, three shapings")
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--tasks",
                    default="verse,summarization,persona,bias,factual recall")
    ap.add_argument("--k", type=float, default=0.05,
                    help="localization budget; 0.05 matches activation_geometry.py")
    ap.add_argument("--shapings", default=",".join(SWEEP_SHAPINGS),
                    help="bs_h=(B*S,H)  s_h=(S,H) prompt-averaged  b_sh=(B,S*H)")
    ap.add_argument("--measures", default=",".join(MEASURES),
                    help="cka (no rank needed), svcca, pwcca")
    ap.add_argument("--steer_dir", default=None,
                    help="disambiguate localizations whose {steer}_steer copies of "
                         "the attribution field disagree, e.g. 'verse-long_steer'")
    ap.add_argument("--patch_site", default=SITE_OUT, choices=list(SITES),
                    help="which tensor the fields index AND the activations are "
                         "read from; both must agree (eval/patch_site.py)")
    ap.add_argument("--response_span", default=SPAN_LEGACY, choices=list(SPANS),
                    help="which results tree to read: 'full' selects the "
                         "-respfix localizations (eval/response_span.py)")
    ap.add_argument("--extract", default="response",
                    choices=("response", "prompt", "last_prompt"),
                    help="prompt = input tokens only, S=n_pos; "
                         "last_prompt = pre-generation state, S=1, one row/prompt")
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
    shapings = [s.strip() for s in a.shapings.split(",") if s.strip() in SWEEP_SHAPINGS]
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
    algo_dir = algo_dir_for(a.patch_site, a.response_span)
    print(f"site/span  : {a.patch_site} / {a.response_span}   -> results dir '{algo_dir}'")
    cells = runnable_cells(models, tasks, roots, algo_dir, a.steer_dir)
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
        # patch_site must be in the namespace: ModelHandler reads it through
        # get_site() to set .dim, and without it .dim silently defaults to the
        # o_proj.output width (240 on gemma) while the fields index 256-wide
        # o_proj.input blocks.
        cfg = SimpleNamespace(args=SimpleNamespace(
            model_id=MODELS[model], device=a.device, pyreft=False,
            full_precision=False, source=STEMS[cs[0][1]],
            patch_site=a.patch_site))
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
                                        a.batch_size, torch, a.extract,
                                        site=a.patch_site)
            Tu, du = collect_structured(mh, tu, su, union, D, a.n_pos,
                                        a.batch_size, torch, a.extract,
                                        site=a.patch_site)
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
                                      arm=arm, n=n, d=Xa.shape[1],
                                      site=a.patch_site, span=a.response_span, **d))
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
                                B=B, S=Spos, extract=a.extract,
                                site=a.patch_site, span=a.response_span,
                                reps=len(acc),
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


# ==========================================================================
# svcca_cosmat.py
# ==========================================================================

matplotlib.use("Agg")

COSMAT_SHAPINGS = ("bs_h", "s_h", "b_last_h")

def make_cmaps(mode):
    """(diverging, sequential) colormaps. Midpoint is gray, never a hue.

    Equal step count per arm, blue for negative and red for positive, so the
    two poles read as opposites and zero recedes into the surface.
    """
    mid = INK[mode]["mid"]
    blue_arm = [SEQ_BLUE[-1], SEQ_BLUE[-4], SEQ_BLUE[-7], SEQ_BLUE[2]]
    red_arm = [DIV_RED[-1], DIV_RED[-4], DIV_RED[3], DIV_RED[0]]
    div = LinearSegmentedColormap.from_list(
        "blue_gray_red", blue_arm + [mid] + red_arm)
    seq = LinearSegmentedColormap.from_list("blue_seq", SEQ_BLUE)
    return div, seq

def collect_prompt_tokens(mh, toks, starts, blocks, D, bs, torch, limit=None,
                          site=None):
    """EVERY prompt token of every prompt, ragged: a list of [T_i, H] arrays.

    `svcca_sweep.collect_structured` returns a rectangular [B, S, H] and to do
    that it must drop prompts that cannot supply S positions. Prompts are
    naturally 27-66 tokens here, so a fixed S either throws prompts away or
    throws tokens away. Ragged keeps both; the shaping functions decide how to
    rectangularise, and only s_h ever needs to.

    Tokens are the ones BEFORE the response starts. Padding is left-side
    (model_handler sets padding_side='left'), so the attention mask -- not index
    0 -- marks where each prompt actually begins.

    `limit` caps tokens per prompt, taken from the END (nearest the response) if
    set; None means all of them.
    """
    # Flat column index into the (layer, hidden) plane, in `blocks` order. This
    # reproduces torch.cat([v[l, D*u:D*(u+1)] for l, u in blocks]) exactly --
    # flattening [n_layers, hidden] puts A[l, ..., c] at l*hidden + c -- but as
    # one fancy-index instead of a per-token Python loop over ~155 blocks.
    seqs = []
    flat = None
    with torch.no_grad():
        for i in range(0, toks["input_ids"].shape[0], bs):
            sl = slice(i, i + bs)
            batch = {k: v[sl].to(mh.device) for k, v in toks.items()}
            with mh.model.trace(batch):
                saved = [attn_proxy(l, site or getattr(mh, "patch_site", SITE_OUT))
                         .detach().cpu().save()
                         for l in mh.model.model.layers]
            A = torch.stack([x.to(torch.float32) for x in saved])  # [L,B,T,hid]
            L, Bb, Tt, hid = A.shape
            if flat is None:
                flat = torch.cat([torch.arange(l * hid + D * u,
                                               l * hid + D * (u + 1))
                                  for l, u in blocks])
            A2 = A.permute(1, 2, 0, 3).reshape(Bb, Tt, L * hid)
            am = toks["attention_mask"][sl]
            for j, s_ in enumerate(starts[sl]):
                take = [t for t in range(0, int(s_)) if am[j, t] == 1]
                if not take:
                    seqs.append(None)
                    continue
                if limit is not None and len(take) > limit:
                    take = take[-limit:]
                seqs.append(A2[j][take][:, flat].numpy())     # [T_i, H_union]
            del A, A2, saved
    return seqs

def build_arms_all(seqs, n_items, shaping, cL, cS):
    """(XL, XS, note) for one shaping, from the ragged per-prompt activations.

    `seqs` is [desired..., undesired...] with n_items desired prompts first.
    `note` records what the shaping had to discard, so a panel never silently
    hides a truncation.
    """
    B = len(seqs)
    if shaping == "bs_h":
        M = np.concatenate(seqs, 0)                       # (sum T_i, H)
        note = f"{B} prompts, {M.shape[0]} tokens"
    elif shaping == "b_last_h":
        M = np.stack([q[-1] for q in seqs])               # (B, H)
        note = f"{B} prompts, last prompt token"
    elif shaping == "s_h":
        # Right-aligned: position -1 is the last prompt token in EVERY prompt,
        # so the average is over comparable positions rather than over whatever
        # happened to land at index k. Truncated to the shortest prompt.
        S = min(len(q) for q in seqs)
        A = np.stack([q[-S:] for q in seqs])              # (B, S, H)
        M = np.concatenate([A[:n_items].mean(0), A[n_items:].mean(0)])
        note = (f"S_min={S} of "
                f"{min(len(q) for q in seqs)}-{max(len(q) for q in seqs)}")
    else:
        raise ValueError(shaping)
    return M[:, cL], M[:, cS], note

def bases(X, max_n):
    """Left singular vectors once, sliced for every n -- U does not depend on n.

    _basis(X, n) is U[:, :n] of the SAME decomposition for every n, so computing
    it per n would repeat a (6000 x 10240) SVD three times over.
    """
    return _basis(X, max_n)

def cosmat_from_bases(QL, QS, n, perm=None):
    """n x n cosines from precomputed bases; `perm` gives the shuffle control.

    Permuting the rows of a matrix permutes the rows of its left-singular basis
    exactly -- centring is permutation-invariant and P*U is still orthonormal --
    so the shuffled control is QS[perm] and needs no second SVD. Verified
    against a fresh SVD of the permuted matrix: principal angles agree to 1e-15.
    """
    k = min(QL.shape[1], QS.shape[1], n)
    if k == 0:
        return None, 0
    Y = QS[perm] if perm is not None else QS
    return QL[:, :k].T @ Y[:, :k], k

def cosine_matrix(X, Y, n):
    """n x n cosines between the two arms' top-n left singular vectors.

    Columns of each basis are orthonormal, so the dot product IS the cosine --
    no renormalisation. Returns the matrix and the actual n kept (a rank-
    deficient arm can supply fewer than n directions).
    """
    Qx, Qy = _basis(X, n), _basis(Y, n)
    k = min(Qx.shape[1], Qy.shape[1])
    if k == 0:
        return None, 0
    return Qx[:, :k].T @ Qy[:, :k], k

def panel_stats(C):
    """Numbers that back up what the picture shows.

    diag_mean       how component-by-component the alignment is
    rowmax_mean     greedy best partner per long-arm component. CLAUDE.md 8.10
                    measures 0.477 here against a diagonal of 0.248 and
                    singular values of 0.986 -- the gap between rowmax_mean and
                    svcca5 is exactly how ROTATIONAL the correspondence is, and
                    it is the reason the entrywise picture looks like noise
                    while the scalar is decisive
    offdiag_max     the largest cross-component cosine; if this beats the
                    diagonal the alignment is rotational, not index-aligned
    rho             the singular values SVCCA would have averaged
    """
    A = np.abs(C)
    k = C.shape[0]
    off = A.copy()
    np.fill_diagonal(off, 0.0)
    rho = np.clip(np.linalg.svd(C, compute_uv=False), 0, 1)
    return dict(
        k=k,
        diag_mean=float(np.mean(np.abs(np.diag(C)))),
        diag_max=float(np.max(np.abs(np.diag(C)))),
        offdiag_mean=float(off.sum() / max(k * k - k, 1)),
        offdiag_max=float(off.max()) if k > 1 else 0.0,
        rowmax_mean=float(np.mean(A.max(1))),
        frob_frac=float((A ** 2).sum() / k),       # 1.0 == subspaces identical
        svcca5=float(np.mean(rho[:min(5, k)])),
        # mean over ALL n singular values of the cosine matrix. For orthonormal
        # bases these are the cosines of the principal angles between the two
        # subspaces, so this is the full-n analogue of svcca5 and is the
        # rotation-invariant summary of how aligned the arms are: it does not
        # care which component matches which, only how much of one subspace
        # lies in the other. diag_mean is the index-aligned counterpart, and
        # the gap between them is how rotational the correspondence is.
        sv_mean=float(np.mean(rho)),
        rho=rho)

def draw_grid(mats, title, subtitle, path, mode, absolute):
    """One figure: rows = shapings, cols = n. Shared scale, one legend.

    `mats` is {(shaping, n): (C, meta)}; meta carries n_samples, degeneracy and
    the stats dict. Panels that could not be computed are drawn as an explicit
    "not available" cell rather than silently dropped, so a missing panel is
    never mistaken for a blank one.
    """
    ink = INK[mode]
    div, seq = make_cmaps(mode)
    cmap = seq if absolute else div
    norm = Normalize(0, 1) if absolute else TwoSlopeNorm(vcenter=0, vmin=-1, vmax=1)

    shapings = sorted({k[0] for k in mats}, key=COSMAT_SHAPINGS.index)
    ns = sorted({k[1] for k in mats})
    nr, nc = len(shapings), len(ns)

    two_line = any(e[1]["degenerate"] for e in mats.values() if e[0] is not None)
    fig, axes = plt.subplots(
        nr, nc,
        figsize=(3.05 * nc + 1.45,
                 3.05 * nr + 1.4 + (0.14 if two_line else 0)),
        squeeze=False)
    fig.patch.set_facecolor(ink["plane"])

    im = None
    for i, sh in enumerate(shapings):
        for j, n in enumerate(ns):
            ax = axes[i][j]
            ax.set_facecolor(ink["surface"])
            entry = mats.get((sh, n))
            if entry is None or entry[0] is None:
                ax.text(0.5, 0.5, "not available", ha="center", va="center",
                        color=ink["muted"], fontsize=9, transform=ax.transAxes)
                ax.set_title(f"{sh}   n={n}", fontsize=9.5,
                             color=ink["primary"], pad=20, loc="left")
                ax.set_xticks([]); ax.set_yticks([])
                for s in ax.spines.values():
                    s.set_color(ink["axis"])
                continue
            C, meta = entry
            M = np.abs(C) if absolute else C
            im = ax.imshow(M, cmap=cmap, norm=norm, interpolation="nearest",
                           aspect="equal")

            k = C.shape[0]
            step = 1 if k <= 12 else 4
            ticks = list(range(0, k, step))
            ax.set_xticks(ticks); ax.set_yticks(ticks)
            ax.set_xticklabels([str(t + 1) for t in ticks], fontsize=7,
                               color=ink["muted"])
            ax.set_yticklabels([str(t + 1) for t in ticks], fontsize=7,
                               color=ink["muted"])
            ax.tick_params(length=0)
            # hairline cell separators: 2px of surface between fills
            ax.set_xticks(np.arange(-.5, k, 1), minor=True)
            ax.set_yticks(np.arange(-.5, k, 1), minor=True)
            ax.grid(which="minor", color=ink["surface"], linewidth=0.6)
            ax.tick_params(which="minor", length=0)

            deg = meta["degenerate"]
            for s in ax.spines.values():
                s.set_color("#d03b3b" if deg else ink["axis"])
                s.set_linewidth(1.8 if deg else 0.8)
            st = meta["stats"]
            head = f"{sh}   n={k}"
            if deg:
                head += "   ⚠ DEGENERATE"
            ax.set_title(head, fontsize=9.5, color=ink["primary"], pad=20,
                         loc="left")
            ax.text(0.0, 1.012, f"rows={meta['n_samples']}   "
                                f"|diag|={st['diag_mean']:.2f}   "
                                f"row-max={st['rowmax_mean']:.2f}   "
                                f"ρ̄₅={st['svcca5']:.2f}",
                    transform=ax.transAxes, fontsize=7.3, color=ink["muted"],
                    va="bottom")
            if i == nr - 1:
                ax.set_xlabel("single-arm component", fontsize=8.5,
                              color=ink["secondary"])
            if j == 0:
                ax.set_ylabel("long-arm component", fontsize=8.5,
                              color=ink["secondary"])

    fig.suptitle(title, fontsize=13, color=ink["primary"], x=0.012, ha="left",
                 y=0.985)
    fig.text(0.012, 0.947, subtitle, fontsize=9, color=ink["secondary"],
             ha="left", va="top")
    # positioned in INCHES, not axes fractions: the same fractions that fit a
    # 3-column grid clip the colorbar label on a 2-column one
    fig_w = fig.get_size_inches()[0]
    fig.tight_layout(rect=[0, 0.035, (fig_w - 1.45) / fig_w, 0.925])

    if im is not None:
        cax = fig.add_axes([(fig_w - 1.30) / fig_w, 0.10, 0.13 / fig_w, 0.62])
        cb = fig.colorbar(im, cax=cax)
        cb.set_label("|cos| between singular vectors" if absolute
                     else "cos between singular vectors",
                     fontsize=8.5, color=ink["secondary"])
        cb.ax.tick_params(labelsize=7.5, colors=ink["muted"], length=0)
        cb.outline.set_edgecolor(ink["axis"])
        cb.outline.set_linewidth(0.6)

    # the degeneracy sentence only earns its place when a panel is degenerate
    note = ("Singular vectors are sign-arbitrary, so a negative cell means the "
            "same angle as its positive twin; read |cos| for magnitude.")
    if two_line:
        note += ("\nRed border = 2n ≥ rows−1, where the leading cosines "
                 "are pinned at 1 on any data, real or shuffled.")
    fig.text(0.012, 0.012, note, fontsize=7.2, color=ink["muted"],
             ha="left", va="bottom", linespacing=1.5)
    fig.savefig(path, dpi=170, facecolor=ink["plane"])
    plt.close(fig)
    return path

def draw_spectrum(mats, title, path, mode):
    """The singular values of every C, i.e. the canonical-correlation spectra.

    This is the quantitative companion to the heatmaps: the heatmap shows WHERE
    the alignment lives, this shows HOW MUCH there is and where it falls off.
    One line per (shaping, n); shaping picks the hue, n picks the marker, so
    identity never rests on colour alone.
    """
    ink = INK[mode]
    hues = {"bs_h": "#2a78d6", "s_h": "#eb6834", "b_last_h": "#1baf7a"}
    all_ns = sorted({k[1] for k in mats})
    marks = {n: m for n, m in zip(all_ns, ["o", "s", "^", "D", "v", "P"])}
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    fig.patch.set_facecolor(ink["plane"])
    ax.set_facecolor(ink["surface"])
    seen = set()
    for (sh, n), (C, meta) in sorted(mats.items(),
                                     key=lambda kv: (COSMAT_SHAPINGS.index(kv[0][0]),
                                                     kv[0][1])):
        if C is None:
            continue
        rho = meta["stats"]["rho"]
        style = "--" if meta["degenerate"] else "-"
        ax.plot(np.arange(1, len(rho) + 1), rho, style,
                color=hues.get(sh, "#4a3aa7"),
                marker=marks.get(n, "o"), markersize=4.5, linewidth=1.8,
                markeredgecolor=ink["surface"], markeredgewidth=0.7,
                label=f"{sh}  n={n}" + (" (degenerate)" if meta["degenerate"]
                                        else ""))
        seen.add(sh)
    ax.set_xlabel("canonical direction", fontsize=9.5, color=ink["secondary"])
    ax.set_ylabel("canonical correlation ρ", fontsize=9.5,
                  color=ink["secondary"])
    ax.set_ylim(0, 1.02)
    ax.grid(axis="y", color=ink["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(ink["axis"]); ax.spines[s].set_linewidth(0.8)
    ax.tick_params(labelsize=8, colors=ink["muted"], length=0)
    ax.set_title(title, fontsize=12, color=ink["primary"], loc="left", pad=10)
    leg = ax.legend(fontsize=8, frameon=False, labelcolor=ink["secondary"],
                    loc="lower left", ncol=2)
    for t in leg.get_texts():
        t.set_color(ink["secondary"])
    fig.tight_layout()
    fig.savefig(path, dpi=170, facecolor=ink["plane"])
    plt.close(fig)
    return path

def replot(outdir, mode, ns_filter=None):
    """Redraw every figure from the saved .npz + cosmat_stats.csv -- no GPU.

    Collecting activations is the expensive half and the matrices are already on
    disk, so a change to the FIGURE should never mean re-running the model. The
    npz carries C itself; the CSV carries the row count and degeneracy flag that
    the panels annotate. Everything else is recomputed from C.
    """
    outdir = Path(outdir)
    csv_path = outdir / "cosmat_stats.csv"
    if not csv_path.exists():
        print(f"no {csv_path} -- nothing to replot")
        return
    meta_by = {}
    for r in csv.DictReader(open(csv_path)):
        meta_by[(r["model"], r["task"], r["arm"], r["shaping"], int(r["n"]))] = r

    for npz_path in sorted(outdir.glob("cosmat_*.npz")):
        stem = npz_path.stem[len("cosmat_"):]
        model, rest = stem.split("_", 1)
        task_tag, extract = rest.rsplit("_", 1)
        task = task_tag.replace("-", " ")
        z = np.load(npz_path)
        mats, ctrl = {}, {}
        for key in z.files:
            shuffled = key.startswith("shuf_")
            sh, ntag = (key[5:] if shuffled else key).rsplit("_n", 1)
            n = int(ntag)
            if ns_filter and n not in ns_filter:
                continue
            row = meta_by.get((model, task, "shuffled" if shuffled else
                               "observed", sh, n))
            if row is None:
                print(f"  {npz_path.name}: no stats row for {key}, skipped")
                continue
            C = z[key]
            entry = (C, dict(n_samples=int(row["rows"]),
                             degenerate=bool(int(row["degenerate"])),
                             stats=panel_stats(C)))
            (ctrl if shuffled else mats)[(sh, n)] = entry
        if not mats:
            continue
        # Rebuild the real caption from the stats rows rather than naming the
        # npz: bs_h's row count IS the token total and b_last_h's IS the prompt
        # count, so the run parameters are recoverable without re-running.
        def _rows(sh):
            e = next((v for (s_, _), v in mats.items() if s_ == sh), None)
            return e[1]["n_samples"] if e else None
        toks, prompts = _rows("bs_h"), _rows("b_last_h")
        bits = ["long vs single localisation", "k=0.05"]
        if prompts:
            bits.append(f"{prompts // 2} prompts/polarity")
        bits.append(f"every prompt token ({toks} total)" if toks
                    else "prompt tokens")
        bits.append("rows differ by shaping")
        sub = " \u00b7 ".join(bits)
        tag = f"{model}_{task_tag}_{extract}"
        for absolute, suffix in ((False, "signed"), (True, "abs")):
            print("  wrote", draw_grid(
                mats, f"Cosine similarity between left singular vectors "
                      f"— {model} / {task}", sub,
                outdir / f"cosmat_{tag}_{suffix}.png", mode, absolute))
        print("  wrote", draw_spectrum(
            mats, f"Canonical correlation spectra — {model} / {task}",
            outdir / f"spectrum_{tag}.png", mode))
        if ctrl:
            for absolute, suffix in ((False, "signed"), (True, "abs")):
                print("  wrote", draw_grid(
                    ctrl, f"SHUFFLE CONTROL — rows permuted — {model} / {task}",
                    "one side's rows permuted: whatever survives is what "
                    "dimension alone manufactures",
                    outdir / f"cosmat_{tag}_shuffled_{suffix}.png",
                    mode, absolute))

def write_stats(outdir, rows):
    """Flush the per-panel numbers. Called after every cell, not just at the end."""
    if not rows:
        return None
    out = outdir / "cosmat_stats.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return out

def _cosmat_main():
    ap = argparse.ArgumentParser(
        description="Cosine-similarity matrices between the arms' left "
                    "singular vectors, per shaping and per n")
    ap.add_argument("--models", default="Qwen1.5-14B-Chat")
    ap.add_argument("--tasks", default="verse")
    ap.add_argument("--ns", default="10,20,50",
                    help="how many leading components to compare")
    ap.add_argument("--shapings", default=",".join(COSMAT_SHAPINGS))
    ap.add_argument("--k", type=float, default=0.05)
    ap.add_argument("--n_items", type=int, default=100,
                    help="prompts per polarity; 100 is the whole file, and "
                         "b_last_h needs it for n=50 to clear degeneracy")
    ap.add_argument("--save_bases", type=int, default=0,
                    help="also store the first N left singular vectors per arm "
                         "in the .npz, so any n can be swept later without "
                         "re-running the model; 0 = off")
    ap.add_argument("--max_tokens", type=int, default=0,
                    help="cap prompt tokens per prompt, taken from the end; "
                         "0 = every prompt token (the default)")
    ap.add_argument("--layer_control", action="store_true",
                    help="add a layer-matched random arm: same block COUNT per "
                         "LAYER as each real arm, units chosen at random inside "
                         "those layers. Holds depth fixed and varies only which "
                         "heads are picked, so it separates 'the localizations "
                         "agree' from 'anything from these layers agrees'.")
    ap.add_argument("--shuffle_control", action="store_true",
                    help="also draw the grid with one side's rows permuted")
    ap.add_argument("--mode", default="light", choices=("light", "dark"))
    ap.add_argument("--patch_site", default=SITE_OUT, choices=list(SITES))
    ap.add_argument("--response_span", default=SPAN_LEGACY, choices=list(SPANS))
    ap.add_argument("--steer_dir", default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--full_precision", action="store_true",
                    help="bf16 instead of the default nf4 quantisation")
    ap.add_argument("--extra_repo", default=None)
    ap.add_argument("--outdir", default="cosmat_figs")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--replot", action="store_true",
                    help="redraw from the saved .npz + stats csv; no model load")
    a = ap.parse_args()

    roots = [REPO] + ([Path(a.extra_repo).resolve()] if a.extra_repo else [])
    models = [m for m in a.models.split(",") if m in MODELS]
    tasks = [t.strip() for t in a.tasks.split(",")]
    ns = sorted(int(x) for x in a.ns.split(","))
    shapings = [s.strip() for s in a.shapings.split(",") if s.strip() in COSMAT_SHAPINGS]

    outdir = REPO / a.outdir
    if a.replot:
        replot(outdir, a.mode, set(ns))
        return
    algo_dir = algo_dir_for(a.patch_site, a.response_span)
    print(f"site/span: {a.patch_site} / {a.response_span}  -> results dir '{algo_dir}'")
    cells = runnable_cells(models, tasks, roots, algo_dir, a.steer_dir)
    print(f"shapings : {shapings}")
    print(f"ns       : {ns}")
    print(f"outdir   : {outdir}")
    print(f"\n{len(cells)} runnable cells:")
    for model, task, _, _, _ in cells:
        print(f"    {model:<24}{task}")
    if a.dry_run:
        return
    if not cells:
        print("\nnothing to run")
        return
    outdir.mkdir(parents=True, exist_ok=True)

    import torch
    sys.path.insert(0, str(REPO))
    from model_handler import ModelHandler

    stat_rows = []
    by_model = defaultdict(list)
    for c in cells:
        by_model[c[0]].append(c)

    for model, cs in by_model.items():
        # One load per model, not one per task. `source` only reaches the marker
        # choice, and for every task in this grid it lands on the same branch --
        # svcca_sweep._cosmat_main takes the first cell's stem for the same reason.
        cfg = SimpleNamespace(args=SimpleNamespace(
            model_id=MODELS[model], device=a.device, pyreft=False,
            full_precision=a.full_precision, source=STEMS[cs[0][1]],
            patch_site=a.patch_site))
        mh = ModelHandler(cfg)
        D = mh.dim
        print(f"\n{'='*100}\n{model}   block width D={D}   k={a.k}   "
              f"{len(cs)} tasks\n{'='*100}", flush=True)

        for _, task, F, fd, fu in cs:
            print(f"\n{'-'*100}\n{model} / {task}\n{'-'*100}", flush=True)
            L, S_ = F[(task, "long")], F[(task, "single")]
            uL, uS = top_units(L, a.k), top_units(S_, a.k)
            # Layer-matched random twins. Same count per layer as the real arm,
            # units drawn uniformly inside that layer. Seeded off the cell so a
            # rerun reproduces the same draw, and drawn from ALL units in the
            # layer (not just unselected ones): the control answers "would a
            # typical block set from these layers align?", and excluding the
            # real picks would make it a different, easier question.
            rL = rS = None
            if a.layer_control:
                n_units = L.shape[1]
                # zlib.crc32, NOT hash(): Python randomises string hashing per
                # process (PYTHONHASHSEED), so hash() would draw a different
                # control on every run and the "same draw on rerun" promise
                # above would be false.
                seed = zlib.crc32(f"{model}/{task}".encode()) % (2 ** 31)
                lrng = np.random.default_rng(seed)

                def layer_matched(units):
                    by_layer = {}
                    for lay, _ in units:
                        by_layer[lay] = by_layer.get(lay, 0) + 1
                    out = []
                    for lay, cnt in sorted(by_layer.items()):
                        pick = lrng.choice(n_units, size=min(cnt, n_units),
                                           replace=False)
                        out += [(lay, int(u)) for u in pick]
                    return sorted(out)

                rL, rS = layer_matched(uL), layer_matched(uS)
            union = sorted(set(uL) | set(uS)
                           | (set(rL) | set(rS) if rL is not None else set()))
            pos = {b_: i for i, b_ in enumerate(union)}

            cap = a.max_tokens or None
            td, sd = prep(mh, fd, a.n_items, torch)
            tu, su = prep(mh, fu, a.n_items, torch)
            sq_d = collect_prompt_tokens(mh, td, sd, union, D, a.batch_size,
                                         torch, cap)
            sq_u = collect_prompt_tokens(mh, tu, su, union, D, a.batch_size,
                                         torch, cap)
            # A prompt with no unmasked prompt token cannot contribute to any
            # shaping; drop it from BOTH polarities' counts explicitly rather
            # than letting a None reach the stack.
            sq_d = [q for q in sq_d if q is not None]
            sq_u = [q for q in sq_u if q is not None]
            nb = min(len(sq_d), len(sq_u))
            if nb == 0:
                print(f"  {task}: no usable prompt tokens, skipped")
                continue
            seqs = sq_d[:nb] + sq_u[:nb]
            lens = [len(q) for q in seqs]
            print(f"  {nb} prompts/polarity, prompt tokens per prompt "
                  f"min={min(lens)} med={int(np.median(lens))} max={max(lens)} "
                  f"total={sum(lens)}", flush=True)

            def cols(units):
                return np.concatenate([np.arange(pos[b_] * D, (pos[b_] + 1) * D)
                                       for b_ in units])
            cL, cS = cols(uL), cols(uS)
            cRL, cRS = ((cols(rL), cols(rS)) if rL is not None else (None, None))

            mats, ctrl, store = {}, {}, {}
            rng = np.random.default_rng(0)
            for sh in shapings:
                XL, XS, note = build_arms_all(seqs, nb, sh, cL, cS)
                n_samples = len(XL)
                print(f"    {sh:<9} rows={n_samples:<6} d_L={XL.shape[1]:<7}"
                      f" d_S={XS.shape[1]:<7} ({note})", flush=True)
                # one SVD per arm, sliced for every n
                # Widen the decomposition when bases are being stored: U does
                # not depend on n, so one SVD at the widest rank serves every
                # later sweep.
                nmax = max(max(ns), a.save_bases)
                QL, QS = bases(XL, nmax), bases(XS, nmax)
                if a.save_bases:
                    store[f"{sh}_QL"] = QL[:, :a.save_bases].astype(np.float32)
                    store[f"{sh}_QS"] = QS[:, :a.save_bases].astype(np.float32)
                if cRL is not None:
                    # identical treatment to the real arms: same shaping, same
                    # centring, same SVD, same ranks -- only the blocks differ
                    RL, RS, _ = build_arms_all(seqs, nb, sh, cRL, cRS)
                    QRL, QRS = bases(RL, nmax), bases(RS, nmax)
                    for n in ns:
                        Cr, kr = cosmat_from_bases(QRL, QRS, n)
                        if Cr is None:
                            continue
                        strr = panel_stats(Cr)
                        stat_rows.append(dict(
                            model=model, task=task, shaping=sh, n=n, k=kr,
                            rows=n_samples, d_L=RL.shape[1], d_S=RS.shape[1],
                            arm="layermatched",
                            degenerate=int(degenerate(kr, n_samples)),
                            **{q: round(strr[q], 6) for q in
                               ("diag_mean", "diag_max", "offdiag_mean",
                                "offdiag_max", "rowmax_mean", "frob_frac",
                                "svcca5", "sv_mean")},
                            rho=";".join(f"{v:.6f}" for v in strr["rho"])))
                    shown = []
                    for n in ns:
                        Cr, _ = cosmat_from_bases(QRL, QRS, n)
                        if Cr is not None:
                            shown.append(f"n={n} sv={panel_stats(Cr)['sv_mean']:.3f}")
                    print("      layer-matched control: " + "  ".join(shown),
                          flush=True)
                    del RL, RS, QRL, QRS
                perm = rng.permutation(n_samples)
                for n in ns:
                    C, k = cosmat_from_bases(QL, QS, n)
                    meta = dict(n_samples=n_samples,
                                degenerate=degenerate(n, n_samples),
                                stats=panel_stats(C) if C is not None else None)
                    mats[(sh, n)] = (C, meta)
                    if C is None:
                        continue
                    st = meta["stats"]
                    flag = " DEGENERATE" if meta["degenerate"] else ""
                    print(f"      n={n:<3} k={k:<3} |diag|={st['diag_mean']:.3f}"
                          f" rowmax={st['rowmax_mean']:.3f}"
                          f" svcca5={st['svcca5']:.3f}{flag}", flush=True)
                    stat_rows.append(dict(
                        model=model, task=task, shaping=sh, n=n, k=k,
                        rows=n_samples, d_L=XL.shape[1], d_S=XS.shape[1],
                        arm="observed", degenerate=int(meta["degenerate"]),
                        **{q: round(st[q], 6) for q in
                           ("diag_mean", "diag_max", "offdiag_mean",
                            "offdiag_max", "rowmax_mean", "frob_frac",
                            "svcca5", "sv_mean")},
                        rho=";".join(f"{v:.6f}" for v in st["rho"])))
                    if a.shuffle_control:
                        Cc, kc = cosmat_from_bases(QL, QS, n, perm)
                        mc = dict(n_samples=n_samples,
                                  degenerate=meta["degenerate"],
                                  stats=panel_stats(Cc))
                        ctrl[(sh, n)] = (Cc, mc)
                        stc = mc["stats"]
                        stat_rows.append(dict(
                            model=model, task=task, shaping=sh, n=n, k=kc,
                            rows=n_samples, d_L=XL.shape[1], d_S=XS.shape[1],
                            arm="shuffled", degenerate=int(mc["degenerate"]),
                            **{q: round(stc[q], 6) for q in
                               ("diag_mean", "diag_max", "offdiag_mean",
                                "offdiag_max", "rowmax_mean", "frob_frac",
                                "svcca5", "sv_mean")},
                            rho=";".join(f"{v:.6f}" for v in stc["rho"])))
                del XL, XS, QL, QS

            tag = f"{model}_{task.replace(' ', '-')}_prompt"
            sub = (f"long vs single localisation · k={a.k} · {nb} "
                   f"prompts/polarity · every prompt token "
                   f"({sum(lens)} total) · rows differ by shaping")
            for absolute, suffix in ((False, "signed"), (True, "abs")):
                print("  wrote", draw_grid(
                    mats, f"Cosine similarity between left singular vectors "
                          f"— {model} / {task}", sub,
                    outdir / f"cosmat_{tag}_{suffix}.png", a.mode, absolute))
            print("  wrote", draw_spectrum(
                mats, f"Canonical correlation spectra — {model} / {task}",
                outdir / f"spectrum_{tag}.png", a.mode))
            if a.shuffle_control:
                for absolute, suffix in ((False, "signed"), (True, "abs")):
                    print("  wrote", draw_grid(
                        ctrl, f"SHUFFLE CONTROL — rows permuted — "
                              f"{model} / {task}",
                        "one side's rows permuted: whatever survives is what "
                        "dimension alone manufactures",
                        outdir / f"cosmat_{tag}_shuffled_{suffix}.png",
                        a.mode, absolute))
            np.savez_compressed(
                outdir / f"cosmat_{tag}.npz",
                **{f"{sh}_n{n}": C for (sh, n), (C, _) in mats.items()
                   if C is not None},
                **{f"shuf_{sh}_n{n}": C for (sh, n), (C, _) in ctrl.items()
                   if C is not None},
                **store)
            write_stats(outdir, stat_rows)
            del seqs, sq_d, sq_u, mats, ctrl

        del mh
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    out = write_stats(outdir, stat_rows)
    if out:
        print(f"\nwrote {out}  ({len(stat_rows)} rows)")


# ==========================================================================
# cosmat_sweep_n.py
# ==========================================================================

matplotlib.use("Agg")

SWEEPN_TASKS = [("persona", "Persona"), ("bias", "Bias"), ("verse", "Verse"),
         ("factual-recall", "Factual recall"),
         ("summarization", "Summarization")]

SWEEPN_SHAPINGS = ("bs_h", "s_h", "b_last_h")

HUES = {"light": ["#2a78d6", "#eb6834", "#1baf7a"],
        "dark": ["#3987e5", "#d95926", "#199e70"]}

def sweep(QL, QS, perm, ns):
    """(mean rho, mean rho^2, min rho, and the same for the shuffle) per n.

    One SVD per n of an n x n matrix -- cheap. The bases themselves are never
    recomputed: U does not depend on n, so QL[:, :n] is the rank-n basis.
    """
    out = {k: [] for k in ("mean", "sq", "min", "fmean", "fsq", "fmin")}
    QSp = QS[perm]
    for n in ns:
        r = np.linalg.svd(QL[:, :n].T @ QS[:, :n], compute_uv=False)
        f = np.linalg.svd(QL[:, :n].T @ QSp[:, :n], compute_uv=False)
        r = np.clip(r, 0, 1); f = np.clip(f, 0, 1)
        out["mean"].append(r.mean());   out["fmean"].append(f.mean())
        out["sq"].append((r ** 2).mean()); out["fsq"].append((f ** 2).mean())
        out["min"].append(r.min());     out["fmin"].append(f.min())
    return {k: np.array(v) for k, v in out.items()}

STATS = [("mean", "mean \u03c1"),
         ("sq", "mean \u03c1\u00b2"),
         ("min", "min \u03c1")]

def draw(data, shaping, path, mode):
    """Rows = statistic, columns = task, one figure per shaping.

    Stacking the three statistics puts them on a shared x-axis so a single
    column reads as one task's full story: how much overlap there is (mean),
    what fraction of the subspace that is (rho^2), and whether ANY direction is
    unshared (min) -- which the two means structurally cannot show.
    """
    ink = INK[mode]; hues = HUES[mode]
    nr, nc = len(STATS), len(data)
    fig, axes = plt.subplots(nr, nc, figsize=(2.55 * nc + 0.9, 2.35 * nr + 0.9),
                             squeeze=False, sharey=True)
    fig.patch.set_facecolor(ink["plane"])
    for i, (stat, slabel) in enumerate(STATS):
        for j, (name, d) in enumerate(data):
            ax = axes[i][j]
            ax.set_facecolor(ink["surface"])
            ns, m = d["ns"], d["m"]
            bound = (m - 1) / 2.0
            if bound < ns[-1]:
                ax.axvspan(bound, ns[-1], color=ink["band"], zorder=0, lw=0)
                if i == 0:
                    ax.text(bound, 1.03, " degenerate", fontsize=6.8,
                            color=ink["muted"], ha="left", va="bottom")
            ax.plot(ns, d[stat], "-", color=hues[0], lw=1.7, label="observed")
            ax.plot(ns, d["f" + stat], "--", color=hues[1], lw=1.5,
                    label="shuffled")
            if stat == "sq":
                ax.plot(ns, ns / (m - 1), ":", color=ink["muted"], lw=1.3,
                        label="chance  n/d")
            if i == 0:
                ax.set_title(f"{name}\n{m} rows", fontsize=9.5,
                             color=ink["primary"], loc="left", pad=7)
            if i == nr - 1:
                ax.set_xlabel("n  (components kept)", fontsize=8.5,
                              color=ink["secondary"])
            if j == 0:
                ax.set_ylabel(slabel, fontsize=10, color=ink["secondary"])
            ax.set_ylim(0, 1.06); ax.set_xlim(ns[0], ns[-1])
            ax.grid(axis="y", color=ink["grid"], lw=0.8)
            ax.set_axisbelow(True)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            for sp in ("left", "bottom"):
                ax.spines[sp].set_color(ink["axis"])
                ax.spines[sp].set_linewidth(0.8)
            ax.tick_params(labelsize=7.5, colors=ink["muted"], length=0)
            if i == 1 and j == nc - 1:
                leg = ax.legend(fontsize=7.5, frameon=False, loc="upper left")
                for t in leg.get_texts():
                    t.set_color(ink["secondary"])
    fig.suptitle(f"Alignment across every legal rank \u2014 {shaping}",
                 fontsize=12.5, color=ink["primary"], x=0.007, ha="left",
                 y=0.995)
    fig.text(0.007, 0.008,
             "\u03c1 are the singular values of M at that n. min \u03c1 is the only "
             "one that can say whether the subspaces COINCIDE \u2014 a mean trades a "
             "lost direction against a perfect one.\n"
             "Shaded: 2n \u2265 rows\u22121, where the leading cosines are pinned at 1 "
             "on any data. Full rank is solid white on noise, which is why n exists.",
             fontsize=7.2, color=ink["muted"], ha="left", va="bottom",
             linespacing=1.5)
    fig.tight_layout(rect=[0, 0.055, 1, 0.945])
    fig.savefig(path, dpi=170, facecolor=ink["plane"])
    plt.close(fig)
    return path

def draw_fulln(entries, task, path, mode):
    """The cosine matrix itself at full rank, beside the largest legal rank.

    Rows = shaping. Left column is the biggest n that clears 2n < rows-1; right
    column keeps EVERY component. The right column is what "all n" means, and
    it is shown with its degeneracy stated rather than withheld -- past the
    bound the leading cosines are pinned at 1 by dimension counting, which is
    visible as the bright diagonal filling in.
    """
    ink = INK[mode]
    seq = LinearSegmentedColormap.from_list("blue_seq", SEQ_BLUE)
    nr = len(entries)
    fig, axes = plt.subplots(nr, 2, figsize=(8.2, 3.5 * nr + 1.1),
                             squeeze=False)
    fig.patch.set_facecolor(ink["plane"])
    im = None
    for i, (shaping, QL, QS, m) in enumerate(entries):
        top = min(QL.shape[1], QS.shape[1], m - 1)
        safe = max(1, (m - 1) // 2 - 1)
        for j, n in enumerate((min(safe, top), top)):
            ax = axes[i][j]
            ax.set_facecolor(ink["surface"])
            M = np.abs(QL[:, :n].T @ QS[:, :n])
            im = ax.imshow(M, cmap=seq, vmin=0, vmax=1,
                           interpolation="nearest", aspect="equal")
            deg = 2 * n >= m - 1
            for sp in ax.spines.values():
                sp.set_color("#d03b3b" if deg else ink["axis"])
                sp.set_linewidth(1.8 if deg else 0.8)
            head = f"{shaping}   n={n}" + ("   \u26a0 DEGENERATE" if deg else "")
            ax.set_title(head, fontsize=10, color=ink["primary"], loc="left",
                         pad=18)
            ax.text(0.0, 1.012,
                    f"rows={m}   |diag|={np.abs(np.diag(M)).mean():.2f}   "
                    f"mean \u03c1={np.linalg.svd(M, compute_uv=False).mean():.2f}",
                    transform=ax.transAxes, fontsize=7.3, color=ink["muted"],
                    va="bottom")
            step = max(1, n // 8)
            ticks = list(range(0, n, step))
            ax.set_xticks(ticks); ax.set_yticks(ticks)
            ax.set_xticklabels([str(t + 1) for t in ticks], fontsize=7,
                               color=ink["muted"])
            ax.set_yticklabels([str(t + 1) for t in ticks], fontsize=7,
                               color=ink["muted"])
            ax.tick_params(length=0)
            if i == nr - 1:
                ax.set_xlabel("single-arm component", fontsize=8.5,
                              color=ink["secondary"])
            if j == 0:
                ax.set_ylabel("long-arm component", fontsize=8.5,
                              color=ink["secondary"])
    fig.suptitle(f"Cosine similarity matrix at full rank \u2014 {task}",
                 fontsize=12.5, color=ink["primary"], x=0.01, ha="left",
                 y=0.995)
    fig.text(0.01, 0.955,
             "left: the largest n that clears the degeneracy bound   \u00b7   "
             "right: every component",
             fontsize=8.5, color=ink["secondary"], ha="left", va="top")
    fig.tight_layout(rect=[0, 0.035, 0.90, 0.945])
    cax = fig.add_axes([0.915, 0.12, 0.016, 0.6])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("|cos| between singular vectors", fontsize=8.5,
                 color=ink["secondary"])
    cb.ax.tick_params(labelsize=7.5, colors=ink["muted"], length=0)
    cb.outline.set_edgecolor(ink["axis"]); cb.outline.set_linewidth(0.6)
    fig.text(0.01, 0.008,
             "Red border: 2n \u2265 rows\u22121, where dimension counting forces "
             "2n\u2212(rows\u22121) of the cosines to 1 on any data at all.",
             fontsize=7.2, color=ink["muted"], ha="left", va="bottom")
    fig.savefig(path, dpi=170, facecolor=ink["plane"])
    plt.close(fig)
    return path

def draw_ladder(rows_data, shaping, path, mode):
    """The cosine matrix across a ladder of n, up to full rank.

    Rows = task, columns = n. The last column keeps EVERY component, which is
    what "all n" means; because each task's activation matrix has its own rank
    that column's n differs per row and is labelled per panel.

    Past 2n >= rows-1 dimension counting forces 2n-(rows-1) of the cosines to 1
    on any data at all, so those panels get a red border. The bright diagonal
    filling in as you move right IS that effect, not a stronger result.
    """
    ink = INK[mode]
    seq = LinearSegmentedColormap.from_list("blue_seq", SEQ_BLUE)
    caps = [min(QL.shape[1], QS.shape[1], m - 1) for _, QL, QS, m in rows_data]
    ladder = [n for n in (10, 25, 50, 100, 200, 400) if n <= min(caps)]
    ncol = len(ladder) + 1
    nr = len(rows_data)
    fig, axes = plt.subplots(nr, ncol, figsize=(2.35 * ncol + 1.5,
                                                2.35 * nr + 1.5), squeeze=False)
    fig.patch.set_facecolor(ink["plane"])
    im = None
    for i, (name, QL, QS, m) in enumerate(rows_data):
        top = caps[i]
        for j, n in enumerate(ladder + [top]):
            ax = axes[i][j]
            ax.set_facecolor(ink["surface"])
            M = np.abs(QL[:, :n].T @ QS[:, :n])
            im = ax.imshow(M, cmap=seq, vmin=0, vmax=1,
                           interpolation="nearest", aspect="equal")
            deg = 2 * n >= m - 1
            for sp in ax.spines.values():
                sp.set_color("#d03b3b" if deg else ink["axis"])
                sp.set_linewidth(1.6 if deg else 0.8)
            tag = f"n={n}" + ("  (all)" if j == ncol - 1 else "")
            ax.set_title(tag, fontsize=9, color=ink["primary"], loc="left",
                         pad=13)
            ax.text(0.0, 1.01,
                    f"|diag|={np.abs(np.diag(M)).mean():.2f}",
                    transform=ax.transAxes, fontsize=6.8, color=ink["muted"],
                    va="bottom")
            ax.set_xticks([]); ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(f"{name}\n{m} rows", fontsize=9,
                              color=ink["secondary"])
    fig.suptitle(f"Cosine similarity matrix across every n \u2014 {shaping}",
                 fontsize=12.5, color=ink["primary"], x=0.008, ha="left",
                 y=0.995)
    fig.text(0.008, 0.955,
             "columns are the truncation rank n; the last column keeps every "
             "component. Rows are tasks.",
             fontsize=8.5, color=ink["secondary"], ha="left", va="top")
    fig.tight_layout(rect=[0, 0.05, 0.90, 0.945])
    cax = fig.add_axes([0.915, 0.14, 0.014, 0.6])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("|cos| between singular vectors", fontsize=8.5,
                 color=ink["secondary"])
    cb.ax.tick_params(labelsize=7.5, colors=ink["muted"], length=0)
    cb.outline.set_edgecolor(ink["axis"]); cb.outline.set_linewidth(0.6)
    fig.text(0.008, 0.008,
             "Red border: 2n \u2265 rows\u22121. There, dimension counting forces "
             "2n\u2212(rows\u22121) of the cosines to exactly 1 on ANY data \u2014 the "
             "diagonal brightening to the right is that, not a stronger result.",
             fontsize=7.2, color=ink["muted"], ha="left", va="bottom")
    fig.savefig(path, dpi=170, facecolor=ink["plane"])
    plt.close(fig)
    return path

def _sweepn_main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--indir", default="cosmat_figs_sweep")
    ap.add_argument("--outdir", default="cosmat_figs_sweep")
    ap.add_argument("--model", default="Qwen1.5-14B-Chat")
    ap.add_argument("--shapings", default=",".join(SWEEPN_SHAPINGS))
    ap.add_argument("--mode", default="light", choices=("light", "dark"))
    ap.add_argument("--ladder", action="store_true",
                    help="heatmap of M across a ladder of n up to full rank")
    ap.add_argument("--heatmaps", action="store_true",
                    help="draw the cosine matrix itself at full rank, per task")
    a = ap.parse_args()
    ind, outd = REPO / a.indir, REPO / a.outdir
    outd.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    shs = [s for s in a.shapings.split(",") if s in SWEEPN_SHAPINGS]
    if a.ladder:
        for shaping in shs:
            rd = []
            for slug, name in SWEEPN_TASKS:
                f = ind / f"cosmat_{a.model}_{slug}_prompt.npz"
                if not f.exists():
                    continue
                z = np.load(f)
                if f"{shaping}_QL" not in z.files:
                    continue
                QL, QS = z[f"{shaping}_QL"], z[f"{shaping}_QS"]
                rd.append((name, QL, QS, QL.shape[0]))
            if rd:
                print("  wrote", draw_ladder(
                    rd, shaping, outd / f"ladder_{shaping}.png", a.mode))
        return
    if a.heatmaps:
        for slug, name in SWEEPN_TASKS:
            f = ind / f"cosmat_{a.model}_{slug}_prompt.npz"
            if not f.exists():
                continue
            z = np.load(f)
            entries = [(sh, z[f"{sh}_QL"], z[f"{sh}_QS"], z[f"{sh}_QL"].shape[0])
                       for sh in shs if f"{sh}_QL" in z.files]
            if entries:
                print("  wrote", draw_fulln(
                    entries, name, outd / f"fulln_{slug}.png", a.mode))
        return

    for shaping in shs:
        data = []
        for slug, name in SWEEPN_TASKS:
            f = ind / f"cosmat_{a.model}_{slug}_prompt.npz"
            if not f.exists():
                continue
            z = np.load(f)
            if f"{shaping}_QL" not in z.files:
                print(f"  {f.name}: no stored bases -- rerun with --save_bases")
                return
            QL, QS = z[f"{shaping}_QL"], z[f"{shaping}_QS"]
            m = QL.shape[0]
            # A centred m-row matrix has rank <= m-1; _basis keeps anything
            # above 1e-8*s[0], which can admit one spurious null direction on
            # the small shapings. Cap so the sweep never plots it.
            top = min(QL.shape[1], QS.shape[1], m - 1)
            # every legal rank, thinned once the curve is smooth so the sweep
            # stays quick on the 10k-row shapings
            ns = list(range(1, min(top, 60) + 1)) + \
                 list(range(65, top + 1, 5))
            ns = [n for n in ns if n <= top]
            d = sweep(QL, QS, rng.permutation(m), ns)
            d["ns"] = np.array(ns); d["m"] = m
            data.append((name, d))
            print(f"  {shaping:<9}{name:<16} rows={m:<6} swept n=1..{ns[-1]}")
        if not data:
            continue
        print("  wrote", draw(data, shaping,
                              outd / f"sweep_{shaping}.png", a.mode))


# ==========================================================================
# cosmat_compare.py
# ==========================================================================

COMPARE_TASKS = ["verse", "summarization", "persona", "bias", "factual recall"]

COMPARE_SHAPINGS = ["bs_h", "s_h", "b_last_h", "b_sh"]

NS = [10, 12, 20, 50]

def load(d):
    """{(task, shaping, n): (observed_row, shuffled_row)} from one outdir."""
    path = Path(d) / "cosmat_stats.csv"
    if not path.is_absolute():
        path = REPO / path
    if not path.exists():
        return None
    obs, shuf = {}, {}
    for r in csv.DictReader(open(path)):
        (obs if r["arm"] == "observed" else shuf)[
            (r["task"], r["shaping"], int(r["n"]))] = r
    return {k: (v, shuf.get(k)) for k, v in obs.items()}

def fmt(pair):
    if pair is None or pair[0] is None:
        return f"{'--':>7}{'--':>7}{'--':>7}{'--':>8}"
    o, s = pair
    rho = float(o["svcca5"])
    fl = float(s["svcca5"]) if s else float("nan")
    return (f"{float(o['diag_mean']):>7.2f}{float(o['rowmax_mean']):>7.2f}"
            f"{rho:>7.3f}{rho - fl:>+8.3f}")

def _compare_main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--a", default="cosmat_figs_all", help="left block")
    ap.add_argument("--b", default="cosmat_figs_prompt", help="right block")
    a = ap.parse_args()

    A, B = load(a.a), load(a.b)
    if A is None or B is None:
        print(f"missing cosmat_stats.csv in {a.a if A is None else a.b}")
        return

    la, lb = Path(a.a).name, Path(a.b).name
    print(f"{'':<16}{'':<9}{'':>3}   {la:^29} | {lb:^29}")
    head = f"{'|diag|':>7}{'rowmax':>7}{'rho5':>7}{'excess':>8}"
    print(f"{'task':<16}{'shaping':<9}{'n':>3}   {head} | {head}   deg")
    print("-" * 100)
    for t in COMPARE_TASKS:
        for sh in COMPARE_SHAPINGS:
            for n in NS:
                pa, pb = A.get((t, sh, n)), B.get((t, sh, n))
                if pa is None and pb is None:
                    continue
                src = pa or pb
                deg = "DEG" if src[0]["degenerate"] == "1" else ""
                print(f"{t:<16}{sh:<9}{n:>3}   {fmt(pa)} | {fmt(pb)}   {deg}")
        print()

    # Aggregate: the headline is whether the excess survives the switch to
    # input-only activations, per shaping, over the NON-degenerate points only.
    print("=" * 100)
    print(f"mean excess over non-degenerate points   {la} vs {lb}")
    for sh in COMPARE_SHAPINGS:
        va, vb = [], []
        for t in COMPARE_TASKS:
            for n in NS:
                pa, pb = A.get((t, sh, n)), B.get((t, sh, n))
                for p, acc in ((pa, va), (pb, vb)):
                    if p and p[0] and p[1] and p[0]["degenerate"] != "1":
                        acc.append(float(p[0]["svcca5"]) -
                                   float(p[1]["svcca5"]))
        ma = sum(va) / len(va) if va else float("nan")
        mb = sum(vb) / len(vb) if vb else float("nan")
        print(f"  {sh:<9} {la}: {ma:+.3f} (n={len(va):>2})    "
              f"{lb}: {mb:+.3f} (n={len(vb):>2})    diff {ma - mb:+.3f}")


# ------------------------------------------------------------------- dispatcher
_SUBCOMMANDS = {
    "sweep": _sweep_main,
    "cosmat": _cosmat_main,
    "sweep-n": _sweepn_main,
    "compare": _compare_main,
}


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in _SUBCOMMANDS:
        print(__doc__.split("SUBCOMMANDS")[1].split("WHAT THE MERGE")[0].rstrip())
        print(f"\nusage: geometry.py {{{'|'.join(_SUBCOMMANDS)}}} [options]")
        raise SystemExit(2)
    sub = sys.argv.pop(1)          # leave the rest for the subcommand's argparse
    return _SUBCOMMANDS[sub]()


if __name__ == "__main__":
    main()
