#!/usr/bin/env python3
"""SVCCA robustness sweep: is the shared-manifold result an artefact of r and n?

WHY THIS EXISTS
---------------
`analysis/activation_geometry.py` reports that the free-form and single-token
localizations select activations that are near-linearly-equivalent (mean top-5
canonical correlation 0.969 against a row-shuffle floor of 0.313, excess 0.656,
positive in 21/21 cells) even though the two arms share only ~6% of their units.
That is the paper's central mechanistic claim.

It is measured at ONE setting: rank r=24, n~640 samples. Canonical correlation
inflates when r is large relative to n -- with enough dimensions you can
correlate noise -- so a reviewer will ask whether the result survives other
(r, n). This script answers that directly.

The floor is not a nuisance term to be subtracted and forgotten. It is large
(0.313 of the 0.969) and it MOVES with r and n: measured at n=128 it was 0.678,
at n=640 it was 0.313. Only `excess = rho_mean - rho_perm` is evidence, and this
sweep shows how excess behaves across the grid.

WHAT IS SWEPT
-------------
    rank r  : PCA dimensionality before canonical correlation
    n       : number of (prompt, position) samples, balanced desired/undesired

For each (model, task) cell, activations are collected ONCE at the largest n in
the grid and then subsampled. Forward passes dominate the cost by orders of
magnitude, so the whole sweep is nearly free once a cell is collected. Do not
"optimise" this by re-collecting per grid point.

WHAT COUNTS AS A PASS
---------------------
The claim is robust if `excess` stays clearly positive across the grid and the
cells keep their ordering. Watch for two failure modes:

  * excess rising monotonically with r at fixed n  -> the estimator is being fed
    more dimensions than the data supports and rho is drifting toward the floor
    for both real and permuted data; the r=24 number would be a lucky point.
  * excess collapsing as n grows -> the shared structure is a small-sample
    artefact. This is the one that would actually threaten the paper.

SELF-CONTAINED
--------------
Everything needed is vendored here or lives in this repo: no import from the
analysis/ folder, no hardcoded machine paths. Run it from anywhere; paths
resolve relative to this file. Only `model_handler.py` (this repo) and the usual
runtime deps (torch, nnsight, transformers, numpy) are required.

Cells covered by this repo alone: all 22 of the paper's grid -- verse,
summarization, persona, bias and factual recall over the five models, less the
three that were never localized (Falcon3 bias, Falcon3 factual recall, OLMo-2
factual recall).

Persona, bias and factual recall were originally umang's. Persona had been wired
in as symlinks into `/home/ubuntu/gcm-interp-umang`, so it was silently
unavailable on any machine but the original box; all three tasks have since been
copied in, and nothing external is needed. `--extra_repo` remains as an escape
hatch for a differently laid-out checkout, and understands umang's
`results_with_answers/` tree as well as `results/`.

USAGE
-----
    python svcca_sweep.py --dry_run                  # list runnable cells, no GPU
    python svcca_sweep.py                            # full sweep, all cells
    python svcca_sweep.py --models gemma-3-12b-it    # one model
    python svcca_sweep.py --extra_repo ../gcm-interp-umang
    python svcca_sweep.py --ranks 8,24,64 --n_grid 128,320,640
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


# The umang checkout keeps some fields under results_with_answers/ rather than
# results/, so each root is searched in both layouts, results/ first. This repo
# only uses results/; the second entry matters for --extra_repo.
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
def svcca(X, Y, r):
    """Canonical correlations between row-paired matrices, after PCA to r.

    Returns None if either side is rank-deficient at this r.
    """
    def basis(M, r):
        M = M - M.mean(0, keepdims=True)
        U, s, _ = np.linalg.svd(M, full_matrices=False)
        keep = min(r, int((s > 1e-8).sum()))
        return U[:, :keep]
    Qx, Qy = basis(X, r), basis(Y, r)
    if Qx.shape[1] == 0 or Qy.shape[1] == 0:
        return None
    rho = np.linalg.svd(Qx.T @ Qy, compute_uv=False)
    return np.clip(rho, 0, 1)


def sweep_point(XL, XS, r, n_perm, seed=0):
    """(rho1, rho_top5, floor, excess) at one (r, n); floor from row shuffles."""
    rho = svcca(XL, XS, r)
    if rho is None or len(rho) == 0:
        return None
    ntop = min(5, len(rho))
    obs = float(np.mean(rho[:ntop]))
    rng = np.random.default_rng(seed)
    perm = []
    for _ in range(n_perm):
        rp = svcca(XL, XS[rng.permutation(len(XS))], r)
        if rp is not None and len(rp) >= ntop:
            perm.append(float(np.mean(rp[:ntop])))
    floor = float(np.mean(perm)) if perm else float("nan")
    return float(rho[0]), obs, floor, obs - floor


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


def collect(mh, toks, starts, blocks, D, n_pos, bs, torch):
    """Per-sample activations restricted to `blocks`: [n_samples, len(blocks)*D].

    Samples are (prompt, response-position) pairs -- the positions ATP itself
    differentiates. Identical to activation_geometry.collect() minus the
    mean/variance accumulators, which this script does not need.
    """
    with torch.no_grad():
        X = []
        for i in range(0, toks["input_ids"].shape[0], bs):
            sl = slice(i, i + bs)
            batch = {k: v[sl].to(mh.device) for k, v in toks.items()}
            with mh.model.trace(batch):
                saved = [l.self_attn.o_proj.output.detach().cpu().save()
                         for l in mh.model.model.layers]
            A = torch.stack([s.to(torch.float32) for s in saved])
            am = toks["attention_mask"][sl]
            for j, s_ in enumerate(starts[sl]):
                valid = [t for t in range(int(s_), am.shape[1]) if am[j, t] == 1]
                if not valid:
                    continue
                take = ([valid[int(round(x))] for x in
                         np.linspace(0, len(valid) - 1, min(n_pos, len(valid)))]
                        if len(valid) > 1 else valid)
                for t in sorted(set(take)):
                    v = A[:, j, t, :]
                    X.append(torch.cat([v[l, D * u:D * (u + 1)]
                                        for l, u in blocks]).numpy())
            del A, saved
    return np.stack(X) if X else None


def balanced_subsample(X, n_total, rng):
    """Take n_total rows, half from each polarity. X is [desired; undesired]."""
    half = len(X) // 2
    per = n_total // 2
    if per > half:
        return None
    d = rng.choice(half, per, replace=False)
    u = rng.choice(half, per, replace=False) + half
    return X[np.concatenate([d, u])]


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
        description="SVCCA rank/sample-size robustness sweep")
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--tasks",
                    default="verse,summarization,persona,bias,factual recall",
                    help="all five paper tasks; every one is now in this repo")
    ap.add_argument("--k", type=float, default=0.05,
                    help="localization budget; 0.05 matches activation_geometry.py")
    ap.add_argument("--ranks", default="2,4,8,16,24,32,48,64,96,128")
    ap.add_argument("--n_grid", default="64,128,256,384,512,640",
                    help="sample counts to subsample down to")
    ap.add_argument("--n_items", type=int, default=32,
                    help="prompts per polarity; n_items*n_pos*2 must exceed max n_grid")
    ap.add_argument("--n_pos", type=int, default=12)
    ap.add_argument("--n_perm", type=int, default=5)
    ap.add_argument("--n_boot", type=int, default=3,
                    help="subsample repeats per n (averaged); 1 disables")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--extra_repo", default=None,
                    help="e.g. ../gcm-interp-umang, to reach persona for the other models")
    ap.add_argument("--out", default="svcca_sweep.csv")
    ap.add_argument("--dry_run", action="store_true",
                    help="list runnable cells and exit; loads no model")
    a = ap.parse_args()

    roots = [REPO] + ([Path(a.extra_repo).resolve()] if a.extra_repo else [])
    models = [m for m in a.models.split(",") if m in MODELS]
    tasks = [t.strip() for t in a.tasks.split(",")]
    ranks = [int(x) for x in a.ranks.split(",")]
    ngrid = sorted(int(x) for x in a.n_grid.split(","))

    print(f"repo roots : {[str(r) for r in roots]}")
    print(f"ranks      : {ranks}")
    print(f"n grid     : {ngrid}")
    cells = runnable_cells(models, tasks, roots)
    print(f"\n{len(cells)} runnable cells:")
    for model, task, _, _, _ in cells:
        print(f"    {model:<24}{task}")
    missing = [(m, t) for m in models for t in tasks
               if not any(c[0] == m and c[1] == t for c in cells)]
    if missing:
        print(f"\n  not runnable here ({len(missing)}): "
              + ", ".join(f"{m}/{t}" for m, t in missing))
        print("  (these three were never localized in either checkout)")
    if a.dry_run:
        return
    if not cells:
        print("\nnothing to run")
        return

    # imported late so --dry_run needs no torch/GPU
    import torch
    sys.path.insert(0, str(REPO))
    from model_handler import ModelHandler

    need = max(ngrid)
    rows = []
    by_model = defaultdict(list)
    for c in cells:
        by_model[c[0]].append(c)

    for model, cs in by_model.items():
        first_task = cs[0][1]
        cfg = SimpleNamespace(args=SimpleNamespace(
            model_id=MODELS[model], device=a.device, pyreft=False,
            full_precision=False, source=STEMS[first_task]))
        mh = ModelHandler(cfg)
        D = mh.dim
        print(f"\n{'='*100}\n{model}   block width D={D}   k={a.k}\n{'='*100}",
              flush=True)

        for _, task, F, fd, fu in cs:
            L, S = F[(task, "long")], F[(task, "single")]
            uL, uS = top_units(L, a.k), top_units(S, a.k)
            union = sorted(set(uL) | set(uS))
            pos = {b: i for i, b in enumerate(union)}

            td, sd = prep(mh, fd, a.n_items, torch)
            tu, su = prep(mh, fu, a.n_items, torch)
            Xd = collect(mh, td, sd, union, D, a.n_pos, a.batch_size, torch)
            Xu = collect(mh, tu, su, union, D, a.n_pos, a.batch_size, torch)
            if Xd is None or Xu is None:
                print(f"  {task}: no valid response positions, skipped")
                continue
            m = min(len(Xd), len(Xu))
            X = np.concatenate([Xd[:m], Xu[:m]])       # [desired; undesired]

            def cols(units):
                return np.concatenate([np.arange(pos[b] * D, (pos[b] + 1) * D)
                                       for b in units])
            cL, cS = cols(uL), cols(uS)
            print(f"  {task}: collected {len(X)} samples "
                  f"(need {need}), d_L={len(cL)}, d_S={len(cS)}", flush=True)
            if len(X) < min(ngrid):
                print(f"    too few samples for any grid point, skipped")
                continue

            for n in ngrid:
                if n > len(X):
                    print(f"    n={n:<5} unavailable (only {len(X)} collected)")
                    continue
                reps = 1 if n == len(X) else a.n_boot
                for r in ranks:
                    acc = []
                    for b in range(reps):
                        rng = np.random.default_rng(1000 * b + 7)
                        Xn = X if n == len(X) else balanced_subsample(X, n, rng)
                        if Xn is None:
                            continue
                        res = sweep_point(Xn[:, cL], Xn[:, cS], r, a.n_perm,
                                          seed=b)
                        if res is not None:
                            acc.append(res)
                    if not acc:
                        continue
                    rho1, obs, floor, exc = (float(np.mean([x[i] for x in acc]))
                                             for i in range(4))
                    rows.append(dict(model=model, task=task, k=a.k, rank=r,
                                     n=n, r_over_n=round(r / n, 4),
                                     d_L=len(cL), d_S=len(cS),
                                     n_collected=len(X), reps=len(acc),
                                     rho1=round(rho1, 6),
                                     rho_top5=round(obs, 6),
                                     floor=round(floor, 6),
                                     excess=round(exc, 6)))
                    print(f"    n={n:<5} r={r:<4} rho5={obs:.4f}  "
                          f"floor={floor:.4f}  excess={exc:+.4f}", flush=True)
            # written after every cell so a crash keeps completed work
            if rows:
                with open(REPO / a.out, "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                    w.writeheader()
                    w.writerows(rows)
        del mh
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not rows:
        print("\nno results")
        return
    print(f"\nwrote {REPO / a.out}  ({len(rows)} sweep points)")

    print(f"\n{'='*100}\nEXCESS over the shuffle floor, averaged across cells")
    print("  the claim is robust if these stay clearly positive and flat-ish")
    print(f"{'='*100}")
    hdr = "  " + "n \\ r".ljust(8) + "".join(f"{r:>9}" for r in ranks)
    print(hdr)
    for n in ngrid:
        line = f"  {n:<8}"
        for r in ranks:
            v = [x["excess"] for x in rows if x["n"] == n and x["rank"] == r]
            line += f"{st.mean(v):>9.3f}" if v else f"{'-':>9}"
        print(line)
    print(f"\n{'='*100}\nFLOOR (row-shuffled) -- shows the estimator inflation being removed")
    print(f"{'='*100}")
    print(hdr)
    for n in ngrid:
        line = f"  {n:<8}"
        for r in ranks:
            v = [x["floor"] for x in rows if x["n"] == n and x["rank"] == r]
            line += f"{st.mean(v):>9.3f}" if v else f"{'-':>9}"
        print(line)

    ref = [x for x in rows if x["rank"] == 24]
    if ref:
        print(f"\n  reference point (r=24, the published setting): "
              f"excess {st.mean(x['excess'] for x in ref):+.3f} "
              f"over {len(ref)} sweep points")
    neg = [x for x in rows if x["excess"] <= 0]
    print(f"  sweep points with excess <= 0: {len(neg)} of {len(rows)}")
    if neg:
        for x in neg[:10]:
            print(f"    {x['model']}/{x['task']}  r={x['rank']} n={x['n']}  "
                  f"excess {x['excess']:+.4f}")


if __name__ == "__main__":
    main()
