#!/usr/bin/env python3
"""Are the long-form and single-token localizations on the SAME CONCEPT MANIFOLD,
and is there a GEOMETRIC reason long-form needs fewer units?

Read at the activations, arm against arm. No random or layer-matched control
appears anywhere: every number compares LF's coordinates to ST's coordinates.

THE TRAP THIS SCRIPT AVOIDS
---------------------------
The arms select ~92% disjoint coordinate blocks (unit Jaccard 0.082). So any
direct geometric comparison of the two SUBSPACES -- cosine between their concept
directions, principal angles between the coordinate spans, eigenvector overlap --
is pinned near zero by construction, no matter what the model is doing. That is
what produced the 0.000 cross-arm eigenvector cosines earlier, and it measures
disjointness, not representation.

Two coordinate sets can be disjoint and still carry the SAME information: unit 3
of layer 10 and unit 9 of layer 22 can be near-perfectly correlated across
inputs. The question "same manifold?" is about shared INFORMATION, not shared
axes, and the instrument for that is canonical correlation.

WHAT IS MEASURED, per (model, task) cell
----------------------------------------
Activations are o_proj.output at ATP-matched positions: the full conversation,
sampled over RESPONSE tokens, which is what ATP itself differentiates. Each
sample is one (prompt, position). The same samples index both arms, so the two
activation matrices are row-paired:

    X_L  (n_samples x d_L)   restricted to LF's top-k blocks
    X_S  (n_samples x d_S)   restricted to ST's top-k blocks

1. SAME MANIFOLD?  SVCCA
   PCA each arm to r components, then canonical correlation between the two.
   rho_i in [0,1]: rho near 1 means a direction in LF's coordinates is a near
   deterministic function of ST's coordinates -- the same latent factor written
   in two different bases. rho near 0 means genuinely unrelated information.
   Reported as the mean of the top canonical correlations.

   Crucially this is invariant to WHICH coordinates each arm owns, so disjoint
   selection cannot manufacture either answer.

2. IS THE CONCEPT IN THE SHARED PART?
   Each canonical variate is correlated with the desired/undesired label. If
   both arms' leading canonical variates track the concept, the shared subspace
   is the concept subspace -- the strong form of "same manifold". If the shared
   directions are concept-blind, the arms agree about something else.

3. GEOMETRY OF THE FEWER-HEADS RESULT
   Let d be the per-coordinate effect size across the FULL residual write,

       d_c = ( mean_desired(c) - mean_undesired(c) ) / pooled_sd(c)

   for every coordinate c of every (layer, block). Standardizing per coordinate
   is what makes this a concept measure rather than a magnitude measure: the raw
   mean difference is dominated by a few outlier dims with huge scale, which is
   an artifact of the model's activation statistics, not of the concept. The
   concept energy an arm's coordinates carry is

       energy(arm) = sum over its blocks of ||d_block||^2  /  ||d||^2

   An arm holding a fraction s of all coordinates would carry s of the energy if
   the concept were spread uniformly. So the scale-free quantity is

       density(arm) = energy(arm) / s

   density > 1 means the arm's coordinates are concept-richer than their size.
   density_LF > density_ST is a geometric explanation for fewer heads: LF's
   blocks sit where more of the concept direction lives, so fewer of them span
   the same amount of it. Here s is the arm's own coordinate share -- an exact
   structural constant, not an empirical control arm.

4. HOW MANY DIRECTIONS DOES THE CONCEPT NEED?
   Within each arm, the participation ratio of the per-block concept energy:

       PR = (sum e_b)^2 / sum e_b^2,  e_b = ||d_block||^2

   reported as PR / (number of blocks the arm holds). Low means the arm's
   concept energy is packed into a few of its own blocks. If LF is lower, its
   budget is spent on a shorter list of blocks that matter -- the geometric
   statement of concentration, measured on activations rather than attribution.
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
FROM_RE = re.compile(
    r"^from_(?P<src>.+?)-(?P<loc>long|single)_to_(?P<base>.+?)(?P<old>_old)?$")


# ----------------------------------------------------------------- attribution
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
    """Signed descending, matching the pipeline.

    eval/logits_handler.py:92 selects with flat.topk(k) -- the largest SIGNED
    attributions. Roughly 48% of every field is negative, and ranking by
    |attribution| instead yields a set overlapping the pipeline's by only
    Jaccard 0.43, i.e. a different experiment from the one the paper runs.
    """
    n = max(1, int(round(frac * A.size)))
    idx = np.argsort(-A, axis=None, kind="stable")[:n]
    return [(int(i // A.shape[1]), int(i % A.shape[1])) for i in idx]


# -------------------------------------------------------------------- sampling
def prep(mh, path, limit):
    rows = [json.loads(l) for l in open(path)][:limit]
    text = [mh.tokenizer.apply_chat_template(r["prompt"], add_generation_prompt=False,
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


@torch.no_grad()
def collect(mh, toks, starts, blocks, D, n_pos, bs):
    """Per-sample activations at `blocks`, plus the full-space mean write.

    Returns X (n_samples x len(blocks)*D) and M (n_layers x hidden), the mean
    o_proj.output over exactly the sampled positions.
    """
    nl = len(mh.model.model.layers)
    X, tot, sq, cnt = [], None, None, 0
    for i in range(0, toks["input_ids"].shape[0], bs):
        sl = slice(i, i + bs)
        batch = {k: v[sl].to(mh.device) for k, v in toks.items()}
        with mh.model.trace(batch):
            saved = [l.self_attn.o_proj.output.detach().cpu().save()
                     for l in mh.model.model.layers]
        A = torch.stack([s.to(torch.float32) for s in saved])   # (nl, b, seq, h)
        am = toks["attention_mask"][sl]
        for j, s_ in enumerate(starts[sl]):
            valid = [t for t in range(int(s_), am.shape[1]) if am[j, t] == 1]
            if not valid:
                continue
            take = ([valid[int(round(x))] for x in
                     np.linspace(0, len(valid) - 1, min(n_pos, len(valid)))]
                    if len(valid) > 1 else valid)
            take = sorted(set(take))
            for t in take:
                v = A[:, j, t, :]                                # (nl, h)
                X.append(torch.cat([v[l, D * u:D * (u + 1)]
                                    for l, u in blocks]).numpy())
                tot = v.clone() if tot is None else tot + v
                sq = v.pow(2) if sq is None else sq + v.pow(2)
                cnt += 1
        del A, saved
    if not X or cnt == 0:
        return None, None, None
    mean = (tot / cnt).numpy()
    var = np.maximum((sq / cnt).numpy() - mean ** 2, 0.0)
    return np.stack(X), mean, var


# -------------------------------------------------------------------- geometry
def svcca(X, Y, r):
    """Canonical correlations between two row-paired matrices, after PCA to r."""
    def basis(M, r):
        M = M - M.mean(0, keepdims=True)
        U, s, _ = np.linalg.svd(M, full_matrices=False)
        keep = min(r, int((s > 1e-8).sum()))
        return U[:, :keep]
    Qx, Qy = basis(X, r), basis(Y, r)
    if Qx.shape[1] == 0 or Qy.shape[1] == 0:
        return None
    Ux, rho, Vt = np.linalg.svd(Qx.T @ Qy, full_matrices=False)
    return np.clip(rho, 0, 1), Qx @ Ux, Qy @ Vt.T


def effect_size(m_des, v_des, m_und, v_und):
    """Per-coordinate discriminability (mean gap / pooled sd), full space.

    Standardizing per coordinate is essential: raw mean differences are
    dominated by whichever dims have the largest scale, so an unstandardized
    version measures activation magnitude rather than concept.
    """
    pooled = np.sqrt(np.maximum((v_des + v_und) / 2.0, 1e-12))
    return (m_des - m_und) / pooled


def energy(d, blocks, D):
    """Fraction of total squared effect size sitting inside `blocks`."""
    tot = float((d ** 2).sum())
    e = np.array([float((d[l, D * u:D * (u + 1)] ** 2).sum()) for l, u in blocks])
    return e, (e.sum() / tot if tot > 0 else float("nan"))


def pr_frac(e):
    """Participation ratio of per-block energies, as a fraction of block count."""
    s = e.sum()
    return float(s ** 2 / (e ** 2).sum() / len(e)) if s > 0 else float("nan")


# ------------------------------------------------------------------- reporting
def boot(d, n=10000, seed=0):
    if len(d) < 2:
        return float("nan"), float("nan")
    r = np.random.default_rng(seed)
    m = np.sort(r.choice(np.asarray(d, float), (n, len(d))).mean(1))
    return float(m[int(.025 * n)]), float(m[int(.975 * n)])


def sign_p(d, tol=1e-12):
    pos = sum(1 for x in d if x > tol)
    neg = sum(1 for x in d if x < -tol)
    n = pos + neg
    if not n:
        return pos, neg, 1.0
    k = min(pos, neg)
    return pos, neg, min(sum(comb(n, i) for i in range(k + 1)) / 2 ** n * 2, 1.0)


def line(label, d, direction="LF higher"):
    if not d:
        print(f"  {label:<34}  no cells")
        return
    lo, hi = boot(d)
    pos, neg, p = sign_p(d)
    hits = neg if "lower" in direction else pos
    print(f"  {label:<34}n={len(d):>3}  {st.mean(d):+8.4f} [{lo:+.4f},{hi:+.4f}]"
          f"   {direction} {hits}/{pos+neg}   p={p:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--tasks", default="verse,summarization,bias,factual recall,persona")
    ap.add_argument("--k", type=float, default=0.05)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--n_items", type=int, default=16)
    ap.add_argument("--n_pos", type=int, default=12)
    ap.add_argument("--rank", type=int, default=24)
    ap.add_argument("--n_perm", type=int, default=5)
    ap.add_argument("--out", default="activation_geometry.csv")
    a = ap.parse_args()
    sys.path.insert(0, str(AYUSHI))
    from model_handler import ModelHandler

    rows = []
    for model in [m for m in a.models.split(",") if m in MODELS]:
        F = load_fields(model)
        tasks = [t for t in a.tasks.split(",")
                 if (t, "long") in F and (t, "single") in F]
        if not tasks:
            continue
        cfg = SimpleNamespace(args=SimpleNamespace(
            model_id=MODELS[model], device=a.device, pyreft=False,
            full_precision=False, source=STEMS[tasks[0]][0]))
        mh = ModelHandler(cfg)
        D = mh.dim
        print(f"\n{'='*104}\n{SHORT[model]}   k={a.k}   block width D={D}\n{'='*104}",
              flush=True)
        for task in tasks:
            src, _ = STEMS[task]
            d = REPOS[OWNER[task]] / "data" / model / f"{src}-long"
            fd = d / f"{src}-long-desired-all.jsonl"
            fu = d / f"{src}-long-undesired-all.jsonl"
            if not (fd.exists() and fu.exists()):
                continue
            L, S = F[(task, "long")], F[(task, "single")]
            nblocks = L.size
            uL = top_units(L, a.k)
            uS = top_units(S, a.k)
            union = sorted(set(uL) | set(uS))
            pos = {b: i for i, b in enumerate(union)}

            td, sd = prep(mh, fd, a.n_items)
            tu, su = prep(mh, fu, a.n_items)
            Xd, Md, Vd = collect(mh, td, sd, union, D, a.n_pos, a.batch_size)
            Xu, Mu, Vu = collect(mh, tu, su, union, D, a.n_pos, a.batch_size)
            if Xd is None or Xu is None:
                print(f"  {task}: no valid response positions, skipped")
                continue
            n = min(len(Xd), len(Xu))
            X = np.concatenate([Xd[:n], Xu[:n]])
            label = np.concatenate([np.ones(n), np.zeros(n)])

            def cols(units):
                return np.concatenate([np.arange(pos[b] * D, (pos[b] + 1) * D)
                                       for b in units])
            XL, XS = X[:, cols(uL)], X[:, cols(uS)]

            res = svcca(XL, XS, a.rank)
            if res is None:
                print(f"  {task}: degenerate activations, skipped")
                continue
            rho, CL, CS = res
            ntop = min(5, len(rho))
            # inflation baseline: same matrices, row correspondence destroyed
            rng = np.random.default_rng(0)
            perm = []
            for _ in range(a.n_perm):
                rp = svcca(XL, XS[rng.permutation(len(XS))], a.rank)
                if rp is not None:
                    perm.append(float(np.mean(rp[0][:ntop])))
            rho_perm = float(np.mean(perm)) if perm else float("nan")
            # how strongly each arm's canonical variates track the concept
            def lab_corr(C):
                out = []
                for i in range(ntop):
                    c = C[:, i]
                    out.append(abs(float(np.corrcoef(c, label)[0, 1]))
                               if c.std() > 0 else 0.0)
                return out
            aL, aS = lab_corr(CL), lab_corr(CS)

            dvec = effect_size(Md, Vd, Mu, Vu)   # concept direction, standardized
            eL, enL = energy(dvec, uL, D)
            eS, enS = energy(dvec, uS, D)
            sL, sS = len(uL) / nblocks, len(uS) / nblocks

            rows.append(dict(
                model=model, task=task, n_samples=len(X),
                d_L=XL.shape[1], d_S=XS.shape[1],
                rho_mean=float(np.mean(rho[:ntop])), rho1=float(rho[0]),
                rho_perm=rho_perm,
                rho_excess=float(np.mean(rho[:ntop])) - rho_perm,
                lab_L=max(aL), lab_S=max(aS),
                energy_L=enL, energy_S=enS, share_L=sL, share_S=sS,
                density_L=enL / sL, density_S=enS / sS,
                prfrac_L=pr_frac(eL), prfrac_S=pr_frac(eS),
                jaccard=len(set(uL) & set(uS)) / len(set(uL) | set(uS))))
            r = rows[-1]
            print(f"  {task:<16} n={len(X):<5} rho(top{ntop})={r['rho_mean']:.3f} "
                  f"shuffled={r['rho_perm']:.3f} excess={r['rho_excess']:+.3f}  "
                  f"concept|canon L={r['lab_L']:.2f} S={r['lab_S']:.2f}  "
                  f"density L={r['density_L']:.2f} S={r['density_S']:.2f}",
                  flush=True)
        del mh
        torch.cuda.empty_cache()

    if not rows:
        print("no cells scored")
        return
    with open(HERE / a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print("\n" + "=" * 104)
    print("1. SAME CONCEPT MANIFOLD?  canonical correlation between the arms")
    print("=" * 104)
    for f, lbl in (("rho1", "top canonical correlation"),
                   ("rho_mean", "mean of top-5"),
                   ("rho_perm", "same, row-shuffled (inflation)"),
                   ("rho_excess", "EXCESS over shuffle"),
                   ("jaccard", "unit Jaccard (for scale)")):
        v = [r[f] for r in rows]
        lo, hi = boot(v)
        print(f"  {lbl:<34}n={len(v):>3}  {st.mean(v):.4f} [{lo:.4f},{hi:.4f}]"
              f"   min {min(v):.3f}  max {max(v):.3f}")
    print("\n  the shuffled row is what the estimator produces from dimensionality")
    print("  alone; only the EXCESS over it is evidence of shared information.")
    print("\n  rho near 1 with Jaccard near 0 -> disjoint coordinates carrying the")
    print("  SAME latent factors: one manifold, two bases.")
    print("  rho near 0 -> genuinely different information: different manifolds.")

    print("\n" + "=" * 104)
    print("2. IS THE SHARED SUBSPACE THE CONCEPT SUBSPACE?")
    print("   |corr| between an arm's leading canonical variates and the")
    print("   desired/undesired label")
    print("=" * 104)
    for f, lbl in (("lab_L", "long-form canonical vs concept"),
                   ("lab_S", "single-token canonical vs concept")):
        v = [r[f] for r in rows]
        lo, hi = boot(v)
        print(f"  {lbl:<34}n={len(v):>3}  {st.mean(v):.4f} [{lo:.4f},{hi:.4f}]")

    print("\n" + "=" * 104)
    print("3. GEOMETRIC EXPLANATION FOR FEWER HEADS")
    print("   density = share of the concept direction's energy an arm holds,")
    print("   divided by its share of coordinates. >1 = concept-richer than size.")
    print("=" * 104)
    for f, lbl in (("energy_L", "LF concept energy"),
                   ("energy_S", "ST concept energy"),
                   ("density_L", "LF density"), ("density_S", "ST density")):
        v = [r[f] for r in rows]
        lo, hi = boot(v)
        print(f"  {lbl:<34}n={len(v):>3}  {st.mean(v):.4f} [{lo:.4f},{hi:.4f}]")
    print()
    line("paired density, LF - ST", [r["density_L"] - r["density_S"] for r in rows])
    line("paired PR/blocks, LF - ST",
         [r["prfrac_L"] - r["prfrac_S"] for r in rows], "LF lower")

    print("\n" + "=" * 104)
    print("PER-CELL")
    print("=" * 104)
    print(f"  {'model':<14}{'task':<16}{'rho5':>7}{'shuf':>7}{'excess':>8}"
          f"{'labL':>7}{'labS':>7}{'densL':>8}{'densS':>8}{'jac':>7}")
    for r in rows:
        print(f"  {SHORT[r['model']]:<14}{r['task']:<16}{r['rho_mean']:>7.3f}"
              f"{r['rho_perm']:>7.3f}{r['rho_excess']:>+8.3f}"
              f"{r['lab_L']:>7.2f}{r['lab_S']:>7.2f}"
              f"{r['density_L']:>8.2f}{r['density_S']:>8.2f}{r['jaccard']:>7.3f}")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
