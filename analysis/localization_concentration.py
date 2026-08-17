#!/usr/bin/env python3
"""WHY does long-form localization need fewer units? Answered from the
LOCALIZATIONS THEMSELVES -- no steering, no ablation, no new intervention.

The behavioural fact needing an explanation (fewer_heads_claim.py): at a
demanding bar, LF reaches the behaviour at a smaller budget than ST
(-1.167 grid steps at B=0.7, 14 of 24 cells, p=0.004), and at k=0.03 LF is
+0.093 accuracy ahead.

THE HYPOTHESIS
--------------
ATP assigns every (layer, unit) block a signed attribution. Localization keeps
the top-k of that field. If LF's attributed effect is packed into FEWER blocks
than ST's, then the same budget k captures a larger share of the total effect,
and the behaviour arrives sooner. "Fewer heads" would then be a property of the
attribution field's shape -- nothing about which units, only how concentrated.

This is a statement purely about the localizations, so it is measured purely
from them. Both arms' fields are read off disk and compared to each other
directly; no random or layer-matched arm appears anywhere in this script.

WHAT IS MEASURED, per (model, task) cell and per arm
----------------------------------------------------
Let a = |attribution| over all N = n_layers x n_units blocks, p = a / sum(a).

  capture(k)   sum of the top k*N entries of a, over sum(a)
               -- the share of total attributed effect a budget k buys.
               This is the quantity the hypothesis is about.

  PR           participation ratio (sum a)^2 / sum a^2, reported as PR/N
               -- the effective FRACTION of blocks carrying the effect.
               Low = concentrated. Insensitive to the choice of k.

  entropy      -sum p log p / log N, normalised to [0,1]. Low = concentrated.

  gini         standard Gini of a. High = concentrated.

  top1_share   a_max / sum(a) -- how much rides on the single largest block.

All four are scale-free, so LF and ST are comparable even though their raw
attribution magnitudes differ by an order of magnitude (ATP on long-form
sequences sums over far more positions).

THE LINK TEST -- does concentration actually explain the budget difference?
--------------------------------------------------------------------------
A concentration difference is only an explanation if the cells where LF is more
concentrated are the cells where LF needs a smaller budget. So, across cells:

    corr( concentration advantage of LF ,  budget advantage of LF )

The budget advantage comes from the existing judged evaluations already on disk
(the same accuracy sweeps fewer_heads_claim.py reads) -- min-k to reach an
absolute behaviour bar B, ST minus LF, positive meaning LF got there sooner.
No new steering is run; the behavioural numbers are the paper's own.

A positive correlation means the shape of the attribution field predicts the
budget saving, and "fewer heads" is explained by concentration. A flat one rules
concentration out and the mechanism stays open.

SAME OR DIFFERENT LOCALIZATION?
-------------------------------
Reported alongside, arm against arm, from the same fields:
  unit Jaccard of the top-k sets, layer Jaccard, Spearman rank correlation over
  all blocks, and cosine between the two flattened fields.
Near-zero agreement at unit level with substantial agreement at layer level
means the arms localize to the same REGION by different BLOCKS.
"""
import argparse
import csv
import json
import re
import statistics as st
import subprocess
from collections import defaultdict
from math import comb
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
AYUSHI = Path("/home/ubuntu/gcm-interp")
UMANG = Path("/home/ubuntu/gcm-interp-umang")
REPOS = {"ayushi": AYUSHI, "umang": UMANG}
OWNER = {"verse": "ayushi", "summarization": "ayushi",
         "bias": "umang", "factual recall": "umang", "persona": "umang"}
TASKNAME = {"verse": "verse", "paragraph": "summarization", "female": "bias",
            "lying": "factual recall", "extraversion": "persona"}
SHORT = {"gemma-3-12b-it": "Gemma-3-12B", "Qwen1.5-14B-Chat": "Qwen1.5-14B",
         "Falcon3-10B-Instruct": "Falcon3-10B", "OLMo-2-1124-13B-DPO": "OLMo-2-13B",
         "Qwen1.5-32B-Chat": "Qwen1.5-32B"}
FROM_RE = re.compile(
    r"^from_(?P<src>.+?)-(?P<loc>long|single)_to_(?P<base>.+?)(?P<old>_old)?$")
ACC_ROOTS = {"ayushi": AYUSHI / "judge-evals" / "accuracy",
             "umang": UMANG / "results_pipeline"}
FN_RE = re.compile(
    r"^(?P<N>\d+)_(?P<reps>random|targeted)_(?P<method>steer|mean)_topk_"
    r"(?P<topk>[\d.]+)_gen_accuracy_"
    r"(?P<metric>w_rf|wo_rf|comb|flu|rel|judge_3|judge_4|judge_5|mcqa)"
    r"\.json\.accuracy\.json$")
# long-form evals are judged (+fluency/relevance); single-token evals are token-matched
MET = {("ayushi", "long"): "w_rf", ("ayushi", "single"): "w_rf",
       ("umang", "long"): "comb", ("umang", "single"): "w_rf"}
OLD = {"Llama-2-13b-chat-hf", "SOLAR-10.7B-Instruct-v1.0", "vicuna-13b-v1.5",
       "phi-4"}
KGRID = [0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.5]


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
    """(task, arm) -> attribution field, newest run when a tree holds several."""
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


# --------------------------------------------------------------- concentration
def capture(a_sorted, total, frac):
    """Share of total |attribution| held by the top `frac` of blocks."""
    n = max(1, int(round(frac * a_sorted.size)))
    return float(a_sorted[:n].sum() / total)


def concentration(A):
    """Scale-free shape statistics of the field the pipeline actually selects from.

    Positive part only, because eval/logits_handler.py:92 takes the largest
    SIGNED values; blocks with negative attribution are never selected at any
    budget, so they are not part of what a budget buys.
    """
    a = np.maximum(A, 0.0).ravel().astype(float)
    N = a.size
    total = a.sum()
    if total <= 0:
        return None
    srt = np.sort(a)[::-1]
    p = a / total
    nz = p[p > 0]
    ent = float(-(nz * np.log(nz)).sum() / np.log(N))
    pr = float(total ** 2 / (a ** 2).sum())          # effective block count
    idx = np.arange(1, N + 1)
    gini = float((2 * (idx * np.sort(a)).sum()) / (N * total) - (N + 1) / N)
    out = {"N": N, "pr_frac": pr / N, "entropy": ent, "gini": gini,
           "top1_share": float(srt[0] / total)}
    for k in KGRID:
        out[f"cap{k}"] = capture(srt, total, k)
    return out


def top_units(A, frac):
    """Signed descending, matching eval/logits_handler.py:92 (flat.topk)."""
    n = max(1, int(round(frac * A.size)))
    idx = np.argsort(-A, axis=None, kind="stable")[:n]
    return {(int(i // A.shape[1]), int(i % A.shape[1])) for i in idx}


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


# ------------------------------------------------------------------ behavioural
def strip_mode(s):
    for m in ("long", "single"):
        if s.endswith("-" + m):
            return s[: -len(m) - 1], m
    return s, None


def harvest_budgets():
    """(model, task, arm) -> {k: accuracy}, long-form eval only, atp arms only.

    Long-form eval only because that is the setting the claim is about: users
    write free-form, so the single-token eval is not the target behaviour.
    """
    g = defaultdict(dict)
    for repo, root in ACC_ROOTS.items():
        if not root.exists():
            continue
        for p in root.rglob("*.accuracy.json"):
            parts = p.relative_to(root).parts
            m = FN_RE.match(parts[-1])
            if not m or any(c.startswith("random") for c in parts) \
                    or "atp" not in parts:
                continue
            srcbase = evald = steerd = model = None
            for i, c in enumerate(parts[:-1]):
                if c.startswith("from_") and "_to_" in c:
                    srcbase, model = c, parts[i - 1] if i else None
                elif c.endswith("_eval"):
                    evald = c
                elif c.endswith("_steer"):
                    steerd = c
            if not (srcbase and evald and steerd and model):
                continue
            if srcbase.endswith("_old") or model in OLD:
                continue
            if m.group("reps") != "targeted" or m.group("method") != "steer":
                continue
            sb = re.match(r"^from_(?P<src>.+?)_to_(?P<base>.+)$", srcbase)
            if not sb:
                continue
            task, arm = strip_mode(sb.group("src"))
            task = TASKNAME.get(task)
            _, ev = strip_mode(evald[:-5])
            _, stm = strip_mode(steerd[:-6])
            if task is None or arm is None or ev != "long" or stm != ev:
                continue
            if MET.get((repo, ev)) != m.group("metric"):
                continue
            try:
                v = json.load(open(p)).get("q1")
            except Exception:
                continue
            if v is None:
                continue
            k = float(m.group("topk"))
            g[(model, task, arm)].setdefault(k, []).append(float(v))
    return {key: {k: max(vs) for k, vs in d.items()} for key, d in g.items()}


def min_k(curve, B):
    """Smallest budget whose accuracy reaches B; None if it never does."""
    for k in sorted(curve):
        if curve[k] >= B:
            return k
    return None


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


def line(label, d, direction="LF higher", fmt="{:+9.4f}"):
    """`direction` names which sign is being counted: 'LF lower' counts d<0."""
    if not d:
        print(f"  {label:<26}  no cells")
        return
    lo, hi = boot(d)
    pos, neg, p = sign_p(d)
    hits = neg if "lower" in direction else pos
    print(f"  {label:<26}n={len(d):>3} {fmt.format(st.mean(d))} "
          f"[{lo:+.4f},{hi:+.4f}]  {direction} {hits}/{pos+neg}  p={p:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(SHORT))
    ap.add_argument("--k", type=float, default=0.05,
                    help="budget for the set-overlap statistics")
    ap.add_argument("--B", type=float, default=0.7,
                    help="behaviour bar for the link test")
    a = ap.parse_args()

    rows, sim = [], []
    for model in [m for m in a.models.split(",") if m in SHORT]:
        F = load_fields(model)
        tasks = sorted({t for (t, arm) in F})
        for task in tasks:
            if (task, "long") not in F or (task, "single") not in F:
                continue
            L, S = F[(task, "long")], F[(task, "single")]
            if L.shape != S.shape:
                continue
            for arm, A in (("long", L), ("single", S)):
                c = concentration(A)
                if c:
                    rows.append(dict(model=model, task=task, arm=arm, **c))
            uL, uS = top_units(L, a.k), top_units(S, a.k)
            lL = {l for l, _ in uL}
            lS = {l for l, _ in uS}
            fl, fs = L.ravel(), S.ravel()
            sim.append(dict(
                model=model, task=task,
                unit_jaccard=len(uL & uS) / max(1, len(uL | uS)),
                layer_jaccard=len(lL & lS) / max(1, len(lL | lS)),
                spearman=float(np.corrcoef(rank(fl), rank(fs))[0, 1]),
                cosine=float(fl @ fs / ((np.linalg.norm(fl) *
                                         np.linalg.norm(fs)) + 1e-12))))
            # NOTE: Spearman/cosine here are on SIGNED attribution, so they
            # answer "do the arms agree on the ordering the pipeline uses".

    if not rows:
        print("no localizations found")
        return
    for name, data in (("localization_concentration.csv", rows),
                       ("localization_similarity.csv", sim)):
        with open(HERE / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0].keys()))
            w.writeheader()
            w.writerows(data)

    idx = {(r["model"], r["task"], r["arm"]): r for r in rows}
    cells = sorted({(r["model"], r["task"]) for r in rows}
                   & {(r["model"], r["task"]) for r in rows})
    cells = [c for c in cells if (c[0], c[1], "long") in idx
             and (c[0], c[1], "single") in idx]
    print(f"{len(cells)} (model, task) cells | "
          f"{len({c[0] for c in cells})} models | overlap budget k={a.k}\n")

    print("=" * 104)
    print("1. IS THE LONG-FORM ATTRIBUTION FIELD MORE CONCENTRATED?")
    print("   paired LF - ST over cells; concentrated = low PR, low entropy,")
    print("   high gini, high capture")
    print("=" * 104)

    def pair(field):
        return [idx[(m, t, "long")][field] - idx[(m, t, "single")][field]
                for m, t in cells]

    line("PR / N  (lower=conc.)", pair("pr_frac"), "LF lower")
    line("entropy (lower=conc.)", pair("entropy"), "LF lower")
    line("gini    (higher=conc.)", pair("gini"), "LF higher")
    line("top-1 share", pair("top1_share"), "LF higher")

    print("\n" + "=" * 104)
    print("2. CAPTURE CURVE: share of total attributed effect bought by budget k")
    print("=" * 104)
    print(f"  {'k':>6}{'LF':>9}{'ST':>9}{'LF-ST':>9}{'95% CI':>22}"
          f"{'LF higher':>12}{'p':>9}")
    for k in KGRID:
        f = f"cap{k}"
        d = pair(f)
        lo, hi = boot(d)
        pos, neg, p = sign_p(d)
        mL = st.mean(idx[(m, t, "long")][f] for m, t in cells)
        mS = st.mean(idx[(m, t, "single")][f] for m, t in cells)
        print(f"  {k:>6}{mL:>9.4f}{mS:>9.4f}{st.mean(d):>+9.4f}"
              f"   [{lo:+.4f},{hi:+.4f}]{f'{pos}/{pos+neg}':>12}{p:>9.4f}")

    print("\n" + "=" * 104)
    print("3. SAME LOCALIZATION OR DIFFERENT? arm vs arm, no control")
    print("=" * 104)
    for f, lbl in (("unit_jaccard", f"unit Jaccard @ k={a.k}"),
                   ("layer_jaccard", "layer Jaccard"),
                   ("spearman", "Spearman over all blocks"),
                   ("cosine", "cosine of |fields|")):
        d = [s[f] for s in sim]
        lo, hi = boot(d)
        print(f"  {lbl:<28}n={len(d):>3}  mean {st.mean(d):+.4f} "
              f"[{lo:+.4f},{hi:+.4f}]  min {min(d):+.3f}  max {max(d):+.3f}")

    print("\n" + "=" * 104)
    print(f"4. LINK TEST: does concentration predict the budget saving?  (B={a.B})")
    print("   budget advantage = min-k(ST) - min-k(LF) on the LONG-FORM eval;")
    print("   positive = LF reached the bar at a smaller budget")
    print("=" * 104)
    budgets = harvest_budgets()
    pts = []
    for m, t in cells:
        cL, cS = budgets.get((m, t, "long")), budgets.get((m, t, "single"))
        if not cL or not cS:
            continue
        kL, kS = min_k(cL, a.B), min_k(cS, a.B)
        if kL is None or kS is None:
            continue
        pts.append(dict(model=m, task=t, adv=kS - kL,
                        d_pr=idx[(m, t, "single")]["pr_frac"]
                        - idx[(m, t, "long")]["pr_frac"],
                        d_cap=idx[(m, t, "long")][f"cap{a.k}"]
                        - idx[(m, t, "single")][f"cap{a.k}"]))
    if len(pts) < 3:
        print(f"  only {len(pts)} cells reach B={a.B} on both arms -- "
              "cannot correlate. Try a lower --B.")
    else:
        print(f"  {len(pts)} cells reach B={a.B} on both arms")
        print(f"  {'model':<14}{'task':<16}{'budget adv':>12}"
              f"{'conc adv (PR)':>15}{'capture adv':>13}")
        for r in sorted(pts, key=lambda r: -r["adv"]):
            print(f"  {SHORT[r['model']]:<14}{r['task']:<16}{r['adv']:>+12.3f}"
                  f"{r['d_pr']:>+15.4f}{r['d_cap']:>+13.4f}")
        y = np.array([r["adv"] for r in pts])
        for lbl, key in (("PR concentration", "d_pr"), ("capture", "d_cap")):
            x = np.array([r[key] for r in pts])
            if x.std() > 0 and y.std() > 0:
                rp = float(np.corrcoef(x, y)[0, 1])
                rs = float(np.corrcoef(rank(x), rank(y))[0, 1])
                print(f"\n  {lbl:<18} Pearson r={rp:+.3f}   Spearman rho={rs:+.3f}")
            else:
                print(f"\n  {lbl:<18} no spread")
        print("\n  positive -> cells where LF's field is more concentrated are the")
        print("              cells where LF needs a smaller budget: concentration")
        print("              explains the fewer-heads result")
        print("  flat     -> concentration is not the mechanism")

    print("\n" + "=" * 104)
    print("5. CONTINUOUS LINK TEST: accuracy gap at a FIXED budget vs capture gap")
    print("   min-k is quantized onto the coarse eval grid, so most cells tie at")
    print("   0.000 and one outlier drives the correlation. Accuracy at a fixed k")
    print("   is continuous and tests the same claim with far more power.")
    print("=" * 104)
    print(f"  {'k':>6}{'n':>5}{'mean acc gap':>14}{'mean cap gap':>14}"
          f"{'Pearson':>10}{'Spearman':>10}")
    for k in (0.01, 0.03, 0.05, 0.07, 0.1):
        xs, ys = [], []
        for m, t in cells:
            cL, cS = budgets.get((m, t, "long")), budgets.get((m, t, "single"))
            if not cL or not cS or k not in cL or k not in cS:
                continue
            capk = min(KGRID, key=lambda g: abs(g - k))
            xs.append(idx[(m, t, "long")][f"cap{capk}"]
                      - idx[(m, t, "single")][f"cap{capk}"])
            ys.append(cL[k] - cS[k])
        if len(xs) < 3:
            print(f"  {k:>6}{len(xs):>5}   too few cells")
            continue
        x, y = np.array(xs), np.array(ys)
        rp = float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 and y.std() > 0 else float("nan")
        rs = float(np.corrcoef(rank(x), rank(y))[0, 1]) if x.std() > 0 and y.std() > 0 else float("nan")
        print(f"  {k:>6}{len(xs):>5}{y.mean():>+14.4f}{x.mean():>+14.4f}"
              f"{rp:>+10.3f}{rs:>+10.3f}")
    print("\n  acc gap = accuracy(LF) - accuracy(ST) at that budget, long-form eval")
    print("  cap gap = share of total attributed effect captured, LF - ST")

    print("\nwrote localization_concentration.csv, localization_similarity.csv")


if __name__ == "__main__":
    main()
