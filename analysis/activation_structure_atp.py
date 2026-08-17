#!/usr/bin/env python3
"""ATP-MATCHED activation structure: full conversations, response positions.

Identical to activation_structure.py except for what it reads. See writes_atp().

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


def eta2(X, labels):
    """multivariate SS_between / SS_total for one factor."""
    xbar = X.mean(0)
    sst = ((X - xbar) ** 2).sum()
    ssb = 0.0
    for g in set(labels):
        m = np.array([l == g for l in labels])
        ssb += m.sum() * ((X[m].mean(0) - xbar) ** 2).sum()
    return float(ssb / sst) if sst > 0 else float("nan")


def spectrum(X):
    Xc = X - X.mean(0, keepdims=True)
    s = np.linalg.svd(Xc, compute_uv=False)
    ev = s ** 2
    ev = ev[ev > 1e-12]
    if ev.size == 0:
        return dict(pr=float("nan"), top1=float("nan"), top5=float("nan")), None
    return dict(pr=float(ev.sum() ** 2 / (ev ** 2).sum()),
                top1=float(ev[0] / ev.sum()),
                top5=float(ev[:5].sum() / ev.sum())), Xc


def top_evecs(Xc, r=3):
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Vt[:r]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="gemma-3-12b-it")
    ap.add_argument("--tasks", default="verse,summarization,bias,factual recall")
    ap.add_argument("--k", type=float, default=0.05)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--n_items", type=int, default=40)
    a = ap.parse_args()

    sys.path.insert(0, str(AYUSHI))
    from model_handler import ModelHandler
    rows, spec = [], []
    for model in [m for m in a.models.split(",") if m in MODELS]:
        F = load_fields(model)
        tasks = [t for t in a.tasks.split(",") if (t, "long") in F and (t, "single") in F]
        if not tasks:
            continue
        cfg = SimpleNamespace(args=SimpleNamespace(
            model_id=MODELS[model], device=a.device, pyreft=False,
            full_precision=False, source=STEMS[tasks[0]][0]))
        mh = ModelHandler(cfg)
        D = mh.dim
        print(f"\n{'='*108}\n{SHORT[model]}   D={D}   k={a.k}\n{'='*108}")

        for task in tasks:
            src, base = STEMS[task]
            paths = {}
            for fmt in ("long", "single"):
                d = REPOS[OWNER[task]] / "data" / model / f"{src}-{fmt}"
                cand = [(d / f"{src}-{fmt}-desired-all.jsonl", "source"),
                        (d / f"{base}-desired-all.jsonl", "base"),
                        (d / f"{base}-{fmt}-desired-all.jsonl", "base")]
                for p, lbl in cand:
                    if p.exists() and (fmt, lbl) not in paths:
                        paths[(fmt, lbl)] = p
            if len(paths) < 4:
                print(f"\n  {task}: need all four 2x2 prompt sets, found "
                      f"{sorted(paths)} -- skipped")
                continue
            X, concept, fmtlab = [], [], []
            for (fmt, lbl), p in sorted(paths.items()):
                texts = full_convs(p, mh.tokenizer, a.n_items)
                W_ = writes_atp(mh, texts, a.batch_size)
                X.append(W_); concept += [lbl] * len(texts); fmtlab += [fmt] * len(texts)
            W = np.concatenate(X, 0)
            nl, dm = W.shape[1], W.shape[2]
            LF = topk_units(F[(task, "long")], a.k)
            ST = topk_units(F[(task, "single")], a.k)
            groups = {"LF-only": [x for x in LF if x not in set(ST)],
                      "ST-only": [x for x in ST if x not in set(LF)]}
            print(f"\n  {task}:  {len(W)} items over the 2x2   "
                  f"LF-only {len(groups['LF-only'])}, ST-only {len(groups['ST-only'])} units")
            print(f"    {'group':<10}{'eta2 concept':>14}{'eta2 format':>13}"
                  f"{'format - concept':>18}")
            for g, units in groups.items():
                ec, ef = [], []
                for (l, u) in units:
                    B = W[:, l, D * u:D * (u + 1)]
                    ec.append(eta2(B, concept)); ef.append(eta2(B, fmtlab))
                # LAYER-MATCHED CONTROL. LF sits ~3.5 layers earlier than ST, so a
                # depth-dependent difference in activation variance would masquerade
                # as a difference in concept encoding. Each arm is therefore scored
                # against random units drawn from its OWN per-layer histogram.
                cc, cf = [], []
                for s_ in range(3):
                    ru = layer_matched(units, W.shape[2] // D)
                    cc.append(st.mean(eta2(W[:, l, D*u:D*(u+1)], concept) for l, u in ru))
                    cf.append(st.mean(eta2(W[:, l, D*u:D*(u+1)], fmtlab) for l, u in ru))
                rows.append(dict(model=model, task=task, group=g, n_units=len(units),
                                 eta2_concept=st.mean(ec), eta2_format=st.mean(ef),
                                 ctl_concept=st.mean(cc), ctl_format=st.mean(cf),
                                 exc_concept=st.mean(ec)-st.mean(cc),
                                 exc_format=st.mean(ef)-st.mean(cf)))
                print(f"    {g:<10}{st.mean(ec):>14.3f}{st.mean(ef):>13.3f}"
                      f"{st.mean(ef)-st.mean(ec):>18.3f}"
                      f"   ctl {st.mean(cc):.3f}/{st.mean(cf):.3f}"
                      f"  exc {st.mean(ec)-st.mean(cc):+.4f}/{st.mean(ef)-st.mean(cf):+.4f}")

            # ---- spectra, on the long-form prompts only (fixed format) --------
            keep = np.array([f == "long" for f in fmtlab])
            def flat(units):
                out = np.zeros((keep.sum(), nl * dm), dtype=np.float32)
                for (l, u) in units:
                    out[:, l * dm + D * u: l * dm + D * (u + 1)] = W[keep][:, l, D*u:D*(u+1)]
                return out
            sp = {}
            for g, units in groups.items():
                s, Xc = spectrum(flat(units))
                sp[g] = (s, Xc)
                spec.append(dict(model=model, task=task, group=g, **s))
            print(f"    {'group':<10}{'eff. dim (PR)':>15}{'top-1 var':>11}{'top-5 var':>11}")
            for g in groups:
                s = sp[g][0]
                print(f"    {g:<10}{s['pr']:>15.2f}{s['top1']:>11.3f}{s['top5']:>11.3f}")
            # cross-arm principal-direction cosine, against a layer-matched null
            vL, vS = top_evecs(sp["LF-only"][1]), top_evecs(sp["ST-only"][1])
            obs = float(np.abs(vL @ vS.T).max())
            null = []
            for _ in range(5):
                a1 = layer_matched(groups["LF-only"], dm // D)
                a2 = layer_matched(groups["ST-only"], dm // D)
                n1, n2 = spectrum(flat(a1))[1], spectrum(flat(a2))[1]
                null.append(float(np.abs(top_evecs(n1) @ top_evecs(n2).T).max()))
            print(f"    top-3 eigenvector cosine LF vs ST: {obs:.3f}   "
                  f"layer-matched null {np.mean(null):.3f} +-{np.std(null):.3f}")
        del mh
        torch.cuda.empty_cache()

    if rows:
        with open(HERE / "activation_structure_atp.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print("\n" + "=" * 108)
        print("PAIRED across cells: is the format effect larger for ST-only units?")
        print("=" * 108)
        idx = {(r["model"], r["task"], r["group"]): r for r in rows}
        cells = sorted({(r["model"], r["task"]) for r in rows})
        for f_ in ("eta2_concept", "eta2_format"):
            d = [idx[(m, t, "ST-only")][f_] - idx[(m, t, "LF-only")][f_]
                 for m, t in cells if (m, t, "ST-only") in idx and (m, t, "LF-only") in idx]
            pos = sum(1 for x in d if x > 0)
            print(f"  {f_:<14} ST - LF: n={len(d)}  mean {st.mean(d):+.4f}  "
                  f"ST higher in {pos}/{len(d)}")
        d = [(idx[(m, t, "ST-only")]["eta2_format"] - idx[(m, t, "ST-only")]["eta2_concept"])
             - (idx[(m, t, "LF-only")]["eta2_format"] - idx[(m, t, "LF-only")]["eta2_concept"])
             for m, t in cells]
        pos = sum(1 for x in d if x > 0)
        print(f"  {'format-bias':<14} ST - LF: n={len(d)}  mean {st.mean(d):+.4f}  "
              f"ST more format-driven in {pos}/{len(d)}")
        print("\nwrote activation_structure.csv")


if __name__ == "__main__":
    main()
