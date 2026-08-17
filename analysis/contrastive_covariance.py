#!/usr/bin/env python3
"""Contrastive covariance: concept structure that mean-shifts cannot see.

eta^2 compares GROUP MEANS. A unit that carries the concept without shifting its
block mean -- in the covariance structure, or along a direction that cancels under
averaging -- scores eta^2_concept ~ 0 while encoding the concept perfectly. Second
moments catch that.

PART A -- HOW MUCH CONDITION-SPECIFIC STRUCTURE DOES EACH ARM HOLD?
Within an arm's selected coordinates, with activations centred inside each
condition,

    C_s = cov(activations | source-instruction prompts)
    C_b = cov(activations | base-instruction prompts)
    contrastive spectrum = eigenvalues of  C_s - C_b

lambda_max(C_s - C_b) is the direction whose variance is most inflated by the
concept. Reported scale-free as

    contrast ratio = lambda_max(C_s - C_b) / ( tr(C_s) + tr(C_b) )

so it is comparable across arms, models and unit counts. Scored against random
units drawn from the arm's OWN per-layer histogram, since deeper coordinates carry
different variance before localization is involved.

C_s - C_b is never formed explicitly: with m = |units| * D running to ~10k
dimensions and only ~60 prompts per condition it would be rank-deficient and
wasteful. Power iteration on matrix-vector products gives the top eigenpair
directly.

PART B -- A SHARED REFERENCE FRAME, WHICH IS THE POINT
Cross-arm eigenvector cosines are structurally pinned at 0: the two arms select
~94% disjoint coordinate blocks, so their principal directions cannot overlap no
matter what the underlying answer is. That comparison is uninformative here and no
amount of data fixes it.

The way around it is a common reference. Compute the concept direction on the FULL
attention write at each layer -- all d_model coordinates, no unit selection -- as

    u_l = top eigenvector of ( C_s^full(l) - C_b^full(l) )     [second moment]
    m_l = mean(source) - mean(base) at layer l                 [first moment]

then ask what fraction of each reference direction's squared mass falls inside each
arm's selected blocks:

    capture(arm, l) = || P_arm u_l ||^2  /  || u_l ||^2

Both arms are now measured against the SAME vector, so the comparison is not
forced to zero by disjointness. Chance capture is |blocks in that layer| / n_units,
and the layer-matched null pins it empirically.
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


def templated(path, tok, limit):
    rows = [json.loads(l) for l in open(path)][:limit]
    out = []
    for r in rows:
        msgs = r["prompt"]
        if any(m["role"] == "assistant" for m in msgs):
            msgs = msgs[:-1]
        out.append(tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False))
    return out


@torch.no_grad()
def writes(mh, prompts, bs):
    layers = mh.model.model.layers
    chunks = []
    for i in range(0, len(prompts), bs):
        t = mh.tokenizer(prompts[i:i + bs], padding=True, truncation=False,
                         return_tensors="pt")
        t = {k: v.to(mh.device) for k, v in t.items()}
        with mh.model.trace(t) as _:
            s = [l.self_attn.o_proj.output[:, -1, :].detach().cpu().save() for l in layers]
        chunks.append(torch.stack([x.to(torch.float32) for x in s]).permute(1, 0, 2))
    return torch.cat(chunks, 0).numpy()


def top_contrastive(Xs, Xb, iters=200, seed=0):
    """Top eigenpair of C_s - C_b without forming it. Xs, Xb are [n, m], centred."""
    m = Xs.shape[1]
    rng = np.random.default_rng(seed)
    v = rng.normal(size=m)
    v /= np.linalg.norm(v)
    ns, nb = max(len(Xs) - 1, 1), max(len(Xb) - 1, 1)
    # shift to keep power iteration on the largest-magnitude eigenvalue positive
    shift = (np.einsum("ij,ij->", Xs, Xs) / ns + np.einsum("ij,ij->", Xb, Xb) / nb) / m
    for _ in range(iters):
        w = Xs.T @ (Xs @ v) / ns - Xb.T @ (Xb @ v) / nb + shift * v
        n = np.linalg.norm(w)
        if n < 1e-12:
            break
        v = w / n
    lam = float(v @ (Xs.T @ (Xs @ v) / ns - Xb.T @ (Xb @ v) / nb))
    tr = float(np.einsum("ij,ij->", Xs, Xs) / ns + np.einsum("ij,ij->", Xb, Xb) / nb)
    return lam, v, tr


def boot(d, n=10000, seed=0):
    if len(d) < 2:
        return float("nan"), float("nan")
    r = np.random.default_rng(seed)
    a = np.asarray(d, float)
    m = np.sort(r.choice(a, (n, len(a))).mean(1))
    return float(m[int(.025 * n)]), float(m[int(.975 * n)])


def sign_p(d):
    pos = sum(1 for x in d if x > 0); neg = sum(1 for x in d if x < 0)
    n = pos + neg
    if not n:
        return pos, neg, 1.0
    k = min(pos, neg)
    return pos, neg, min(sum(comb(n, i) for i in range(k + 1)) / 2 ** n * 2, 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--tasks", default="verse,summarization,bias,factual recall,persona")
    ap.add_argument("--k", type=float, default=0.05)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch_size", type=int, default=8)
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
            print(f"{model}: no task with both arms"); continue
        cfg = SimpleNamespace(args=SimpleNamespace(
            model_id=MODELS[model], device=a.device, pyreft=False,
            full_precision=False, source=STEMS[tasks[0]][0]))
        mh = ModelHandler(cfg)
        D = mh.dim
        print(f"\n{'='*104}\n{SHORT[model]}  D={D}  k={a.k}\n{'='*104}", flush=True)
        for task in tasks:
            src, base = STEMS[task]
            d = REPOS[OWNER[task]] / "data" / model / f"{src}-long"
            fs = d / f"{src}-long-desired-all.jsonl"
            fb = next((c for c in (d / f"{base}-desired-all.jsonl",
                                   d / f"{base}-long-desired-all.jsonl") if c.exists()), None)
            if not (fs.exists() and fb):
                print(f"  {task}: prompts missing -- skipped"); continue
            Ws = writes(mh, templated(fs, mh.tokenizer, a.n_items), a.batch_size)
            Wb = writes(mh, templated(fb, mh.tokenizer, a.n_items), a.batch_size)
            nl, dm = Ws.shape[1], Ws.shape[2]
            nu = dm // D
            LF, ST = topk_units(F[(task, "long")], a.k), topk_units(F[(task, "single")], a.k)
            groups = {"LF-only": [x for x in LF if x not in set(ST)],
                      "ST-only": [x for x in ST if x not in set(LF)]}

            def gather(units, W):
                return np.concatenate([W[:, l, D*u:D*(u+1)] for l, u in units], 1)

            print(f"\n  {task}:  {len(Ws)}+{len(Wb)} prompts   "
                  f"LF-only {len(groups['LF-only'])}, ST-only {len(groups['ST-only'])} units")
            print(f"    {'group':<10}{'contrast ratio':>16}{'layer-matched null':>21}{'excess':>9}")
            for g, units in groups.items():
                Xs = gather(units, Ws); Xb = gather(units, Wb)
                Xs = Xs - Xs.mean(0); Xb = Xb - Xb.mean(0)
                lam, _, tr = top_contrastive(Xs, Xb)
                ratio = lam / tr if tr > 0 else float("nan")
                ctl = []
                for s in range(a.n_control):
                    ru = layer_matched(units, nu)
                    Ys = gather(ru, Ws); Yb = gather(ru, Wb)
                    Ys = Ys - Ys.mean(0); Yb = Yb - Yb.mean(0)
                    l2, _, t2 = top_contrastive(Ys, Yb, seed=s)
                    ctl.append(l2 / t2 if t2 > 0 else np.nan)
                rows.append(dict(model=model, task=task, group=g, part="A_contrast",
                                 value=ratio, control=float(np.nanmean(ctl)),
                                 excess=ratio - float(np.nanmean(ctl))))
                print(f"    {g:<10}{ratio:>16.4f}{np.nanmean(ctl):>17.4f} "
                      f"+-{np.nanstd(ctl):.4f}{ratio-np.nanmean(ctl):>9.4f}")

            # ---- Part B: capture of a COMMON reference direction ---------------
            cap = defaultdict(list)
            for l in range(nl):
                Xs = Ws[:, l, :] - Ws[:, l, :].mean(0)
                Xb = Wb[:, l, :] - Wb[:, l, :].mean(0)
                _, u, _ = top_contrastive(Xs, Xb)          # second-moment reference
                mdir = Ws[:, l, :].mean(0) - Wb[:, l, :].mean(0)  # first-moment reference
                mdir = mdir / (np.linalg.norm(mdir) + 1e-12)
                for g, units in groups.items():
                    blocks = [uu for ll, uu in units if ll == l]
                    if not blocks:
                        continue
                    mask = np.zeros(dm, bool)
                    for uu in blocks:
                        mask[D*uu:D*(uu+1)] = True
                    cap[(g, "2nd")].append((float((u[mask]**2).sum()), mask.sum()/dm))
                    cap[(g, "1st")].append((float((mdir[mask]**2).sum()), mask.sum()/dm))
            print(f"    {'group':<10}{'capture of 2nd-moment dir':>27}{'chance':>9}"
                  f"{'capture of 1st-moment dir':>27}{'chance':>9}")
            for g in groups:
                for mom in ("2nd", "1st"):
                    v = cap[(g, mom)]
                    if not v:
                        continue
                    obs = st.mean(x for x, _ in v); ch = st.mean(c for _, c in v)
                    rows.append(dict(model=model, task=task, group=g,
                                     part=f"B_capture_{mom}", value=obs, control=ch,
                                     excess=obs - ch))
                v2 = cap[(g, "2nd")]; v1 = cap[(g, "1st")]
                if v2 and v1:
                    print(f"    {g:<10}{st.mean(x for x,_ in v2):>27.4f}"
                          f"{st.mean(c for _,c in v2):>9.4f}"
                          f"{st.mean(x for x,_ in v1):>27.4f}"
                          f"{st.mean(c for _,c in v1):>9.4f}")
        del mh
        torch.cuda.empty_cache()

    if not rows:
        return
    with open(HERE / "contrastive_covariance.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    idx = {(r["model"], r["task"], r["group"], r["part"]): r for r in rows}
    cells = sorted({(r["model"], r["task"]) for r in rows})
    print("\n" + "=" * 104)
    print(f"PAIRED ST - LF across {len(cells)} cells")
    print("=" * 104)
    for part, lbl in (("A_contrast", "condition-specific structure (excess over null)"),
                      ("B_capture_2nd", "capture of the 2nd-moment concept direction (excess over chance)"),
                      ("B_capture_1st", "capture of the 1st-moment concept direction (excess over chance)")):
        d = [idx[(m, t, "ST-only", part)]["excess"] - idx[(m, t, "LF-only", part)]["excess"]
             for m, t in cells
             if (m, t, "ST-only", part) in idx and (m, t, "LF-only", part) in idx]
        if not d:
            continue
        lo, hi = boot(d); pos, neg, p = sign_p(d)
        print(f"  {lbl:<62}n={len(d):>2}  {st.mean(d):+.4f} [{lo:+.4f},{hi:+.4f}]  "
              f"ST higher {pos}/{len(d)}  p={p:.3f}")
    print("\nwrote contrastive_covariance.csv")


if __name__ == "__main__":
    main()
