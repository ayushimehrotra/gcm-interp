#!/usr/bin/env python3
"""Necessity as a DOSE-RESPONSE curve.

The earlier version ran only k=0.05 and found nothing for any arm. A smoke test
explains why: ablating 25% of all units destroys verse outright (argmax changes at
68% of positions), 50% degrades further, 100% gives gibberish -- but at k=0.05
nothing moves for LF, ST, or layer-matched random alike. The whole comparison sat
below the threshold where ablation does anything, so the null was about dose, not
about selection.

k=0.05 was inherited from the steering analyses, where it is the right range
because steering ADDS a unit-norm vector scaled by N. Ablation removes ~1% of the
residual stream at that budget. The two interventions do not share an operating
range, and the dose has to be swept rather than assumed.

So sweep it: k in {0.05, 0.1, 0.2, 0.3, 0.5} for each arm and its layer-matched
control, and find (a) where the behaviour breaks and (b) whether the selection
matters anywhere along the curve.

Everything measured so far is SUFFICIENCY -- add a steering vector at these units,
get the behaviour. That is a weak test here, because `_steering_vector` normalises
each block to unit norm before injecting, so the direction injected is set by the
task vector and the unit set mostly decides dosage. Ablation has no such escape
hatch: if a unit set is not carrying the behaviour, removing it costs nothing.

THE MEASURE
-----------
No judge and no generation. On the long-form data, prompt p has two committed
continuations -- the behaviour-consistent one (e.g. the verse reply) and the
contrasting one (the prose reply). Define the behaviour margin

    L(p) = log P(desired continuation | p) - log P(undesired continuation | p)

summed over continuation tokens, exactly the quantity ATP differentiates. Ablating
a unit set U replaces those coordinate blocks with zeros during the forward pass:

    a_l[u*D:(u+1)*D] <- 0    for every (l,u) in U

and the necessity of U is the damage it does,  dL(U) = L_ablated - L_baseline,
negative meaning the behaviour margin collapsed.

THE CONTRAST THAT MATTERS
-------------------------
    dL(LF) vs dL(ST)                which set is the model actually using
    dL(.)  vs dL(layer-matched random of the same size and layer profile)

The layer-matched control is essential: LF and ST sit at different depths, and
ablating deeper or shallower coordinates costs different amounts on its own. Each
arm is therefore scored against random units drawn inside ITS OWN layer histogram.

A dissociation -- LF necessary, ST not -- would say ST localization finds a causal
pathway into the logits rather than the pathway generation actually runs through.
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


def layer_matched(units, n_units, nl):
    """Random units with the same per-layer counts."""
    h = defaultdict(int)
    for l, _ in units:
        h[l] += 1
    return [(l, int(u)) for l, n in h.items()
            for u in RNG.choice(n_units, n, replace=False)]


def by_layer(units):
    d = defaultdict(list)
    for l, u in units:
        d[l].append(u)
    return d


# ------------------------------------------------------------------ scoring --
@torch.no_grad()
def margin(mh, toks_des, toks_und, starts_des, starts_und, ablate, D, batch_size):
    """mean over items of  (1/|d|) logP(desired) - (1/|u|) logP(undesired).

    PER-TOKEN, not summed. Summing over the continuation makes the metric a length
    ruler: each extra token costs about one nat of total log-probability, so a
    contrast whose two sides differ in length is scored mostly on that difference.
    Measured on the committed data, corr(length difference, summed margin) = +0.535
    with a slope of +1.01 nats/token -- and summarization's desired continuation runs
    100-165 tokens longer than its undesired one, which is why its summed baselines
    came out negative. Dividing by the token count removes it.
    """
    def ll(toks, starts):
        out = []
        n = toks["input_ids"].shape[0]
        for i in range(0, n, batch_size):
            sl = slice(i, i + batch_size)
            batch = {k: v[sl].to(mh.device) for k, v in toks.items()}
            with mh.model.trace(batch) as _:
                if ablate:
                    for l, us in ablate.items():
                        for u in us:
                            mh.model.model.layers[l].self_attn.o_proj.output[
                                ..., D * u:D * (u + 1)] = 0.0
                logits = mh.model.lm_head.output.detach().cpu().save()
            lp = torch.log_softmax(logits.to(torch.float32), -1)
            ids = toks["input_ids"][sl]
            am = toks["attention_mask"][sl]
            for j, st_ in enumerate(starts[sl]):
                tgt = ids[j, st_ + 1:]
                keep = am[j, st_ + 1:].bool()          # ignore right-side padding
                if keep.sum() == 0:
                    continue
                tok_lp = lp[j, st_:-1, :].gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
                out.append(float(tok_lp[keep].mean()))  # PER-TOKEN
        return out
    d = ll(toks_des, starts_des)
    u = ll(toks_und, starts_und)
    return float(np.mean([x - y for x, y in zip(d, u)]))


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="gemma-3-12b-it")
    ap.add_argument("--tasks", default="verse,summarization,bias")
    ap.add_argument("--ks", default="0.05,0.1,0.2,0.3,0.5")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--n_items", type=int, default=40)
    ap.add_argument("--n_control", type=int, default=3)
    ap.add_argument("--margin", default="long", choices=["long", "single"],
                    help="which behaviour margin to score. 'long' uses the long-form "
                         "continuation pair; 'single' uses the single-token answer pair. "
                         "Measuring only on 'long' is biased toward the LF arm, which "
                         "was localized on that data -- run both.")
    a = ap.parse_args()
    ks = [float(x) for x in a.ks.split(",")]

    sys.path.insert(0, str(AYUSHI))
    from model_handler import ModelHandler

    rows = []
    for model in [m for m in a.models.split(",") if m in MODELS]:
        F = load_fields(model)
        tasks = [t for t in a.tasks.split(",")
                 if (t, "long") in F and (t, "single") in F]
        if not tasks:
            print(f"{model}: no task with both arms")
            continue
        cfg = SimpleNamespace(args=SimpleNamespace(
            model_id=MODELS[model], device=a.device, pyreft=False,
            full_precision=False, source=STEMS[tasks[0]][0]))
        mh = ModelHandler(cfg)
        D = mh.dim
        print(f"\n{'='*104}\n{SHORT[model]}  D={D}\n{'='*104}")

        for task in tasks:
            src, base = STEMS[task]
            mm = a.margin
            d = REPOS[OWNER[task]] / "data" / model / f"{src}-{mm}"
            fd = d / f"{src}-{mm}-desired-all.jsonl"
            fu = d / f"{src}-{mm}-undesired-all.jsonl"
            if not (fd.exists() and fu.exists()):
                print(f"  {task}: {mm} contrast files missing -- skipped")
                continue
            td, sd = prep(mh, fd, a.n_items)
            tu, su = prep(mh, fu, a.n_items)
            base_margin = margin(mh, td, tu, sd, su, None, D, a.batch_size)
            nl, nu = F[(task, "long")].shape
            print(f"\n  {task} [{mm} margin]: baseline {base_margin:+.2f} nats "
                  f"({a.n_items} items)")
            print(f"    {'k':>6}  {'arm':<8}{'ablated dL':>13}{'layer-matched control':>24}"
                  f"{'arm - control':>15}")
            for k in ks:
                for arm in ("long", "single"):
                    units = topk_units(F[(task, arm)], k)
                    dl = margin(mh, td, tu, sd, su, by_layer(units), D, a.batch_size) - base_margin
                    ctl = [margin(mh, td, tu, sd, su,
                                  by_layer(layer_matched(units, nu, nl)), D, a.batch_size)
                           - base_margin for _ in range(a.n_control)]
                    rows.append(dict(model=model, task=task, margin=mm, topk=k, arm=arm,
                                     n_units=len(units), baseline=base_margin,
                                     d_ablate=dl, d_control=float(np.mean(ctl)),
                                     control_sd=float(np.std(ctl)),
                                     excess=dl - float(np.mean(ctl))))
                    print(f"    {k:>6}  {arm:<8}{dl:>13.2f}"
                          f"{np.mean(ctl):>18.2f} +-{np.std(ctl):.2f}"
                          f"{dl-np.mean(ctl):>15.2f}")
        del mh
        torch.cuda.empty_cache()

    if not rows:
        return
    out = HERE / f"necessity_{a.margin}.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    print("\n" + "=" * 104)
    print(f"PAIRED on the {a.margin.upper()} margin: is the LF set more necessary than the ST set?")
    print("(excess = damage beyond a layer-matched random set of the same size)")
    print("=" * 104)
    idx = {(r["model"], r["task"], r["topk"], r["arm"]): r for r in rows}
    for field in ("d_ablate", "excess"):
        d = [idx[(m, t, k, "long")][field] - idx[(m, t, k, "single")][field]
             for (m, t, k, arm) in idx if arm == "long" and (m, t, k, "single") in idx]
        if not d:
            continue
        pos = sum(1 for x in d if x > 0); neg = sum(1 for x in d if x < 0)
        n = pos + neg
        p = min(sum(comb(n, i) for i in range(min(pos, neg) + 1)) / 2 ** n * 2, 1.0) if n else 1.0
        print(f"  {field:<12} LF - ST: n={len(d):>2}  mean {st.mean(d):+.3f}  "
              f"LF more damaging in {neg}/{len(d)}  p={p:.3f}")
    print(f"\nwrote {out.name}")


if __name__ == "__main__":
    main()
