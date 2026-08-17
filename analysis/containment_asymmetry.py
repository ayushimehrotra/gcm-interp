#!/usr/bin/env python3
"""Is single-token localization a NOISIER VERSION of long-form localization, or a
genuinely different target?

Jaccard says the two arms share ~8% of their selected blocks. That is a symmetric
statistic: it cannot tell "two unrelated selections" from "one selection plus
noise around the other". Those are different mechanistic claims and they predict
different things about why LF needs fewer units.

The asymmetric question is where each arm's picks sit in the OTHER arm's full
ranking:

    if LF's top blocks rank HIGH in ST's field, but ST's top blocks rank
    ORDINARY in LF's field, then ST's field contains LF's signal plus extra
    mass elsewhere -- ST is LF blurred, and LF's advantage is denoising.

    if both sit at chance in the other's ranking, the arms are localizing
    genuinely different things and no denoising story applies.

SELECTION MATCHES THE PIPELINE
------------------------------
Blocks are ranked by SIGNED attribution, descending, because that is what the
eval pipeline does: eval/logits_handler.py:92 calls flat.topk(k), which takes
the largest signed values, not the largest magnitudes. This matters a lot --
about 48% of every field is negative, and ranking by |attribution| instead
produces a selection that overlaps the pipeline's by only Jaccard 0.43. Any
statement about "the units localization picks" has to use the signed order.

MEASURES, per (model, task)
---------------------------
  pct_in_other   mean percentile of one arm's top-k inside the other arm's full
                 ranking. 1.0 = at the very top, 0.5 = exactly chance.

  AUC            P(a randomly chosen selected block outranks a randomly chosen
                 unselected block) under the other arm's scores. 0.5 = chance.
                 Rank-based, so insensitive to the attribution scale, which
                 differs by an order of magnitude between arms.

  recall(m)      fraction of one arm's top-k that falls inside the other arm's
                 top-m, as m grows. The curve that rises faster is the arm whose
                 field already contains the other's picks.

  asymmetry      AUC_S(U_L) - AUC_L(U_S). Positive = LF's picks are better
                 represented in ST's field than the reverse = ST looks like a
                 blurred LF.

Chance is exactly 0.5 by construction here -- a structural constant, not an
empirical control arm, so no random baseline is involved anywhere.
"""
import argparse
import csv
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
MGRID = [0.05, 0.1, 0.2, 0.3, 0.5]


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


def order(A):
    """Block indices in the pipeline's selection order: signed, descending."""
    return np.argsort(-A, axis=None, kind="stable")


def topk_idx(A, k):
    return order(A)[:max(1, int(round(k * A.size)))]


def percentile_in(other_rank, sel):
    """Mean percentile of `sel` inside another field's ranking. 0.5 = chance."""
    N = len(other_rank)
    return float(np.mean(1.0 - other_rank[sel] / (N - 1)))


def auc_of(other_rank, sel, N):
    """P(selected outranks unselected) under the other arm's order."""
    mask = np.zeros(N, bool)
    mask[sel] = True
    # rank 0 = best; convert to "score" where higher is better
    score = (N - 1) - other_rank
    sp = np.sort(score[mask])
    su = np.sort(score[~mask])
    # rank-sum (Mann-Whitney) without scipy
    allv = np.concatenate([sp, su])
    r = np.empty(len(allv))
    o = allv.argsort()
    r[o] = np.arange(len(allv))
    for v in np.unique(allv):
        m = allv == v
        if m.sum() > 1:
            r[m] = r[m].mean()
    R1 = r[:len(sp)].sum()
    n1, n2 = len(sp), len(su)
    return float((R1 - n1 * (n1 - 1) / 2) / (n1 * n2))


def boot(d, n=10000, seed=0):
    if len(d) < 2:
        return float("nan"), float("nan")
    r = np.random.default_rng(seed)
    m = np.sort(r.choice(np.asarray(d, float), (n, len(d))).mean(1))
    return float(m[int(.025 * n)]), float(m[int(.975 * n)])


def sign_p(d, ref=0.0):
    pos = sum(1 for x in d if x > ref)
    neg = sum(1 for x in d if x < ref)
    n = pos + neg
    if not n:
        return pos, neg, 1.0
    k = min(pos, neg)
    return pos, neg, min(sum(comb(n, i) for i in range(k + 1)) / 2 ** n * 2, 1.0)


def line(label, d, ref=0.0, direction="above"):
    lo, hi = boot(d)
    pos, neg, p = sign_p(d, ref)
    print(f"  {label:<40}n={len(d):>3}  {st.mean(d):+.4f} [{lo:+.4f},{hi:+.4f}]"
          f"   {direction} {pos}/{pos+neg}   p={p:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(SHORT))
    ap.add_argument("--k", type=float, default=0.05)
    a = ap.parse_args()

    rows = []
    for model in [m for m in a.models.split(",") if m in SHORT]:
        F = load_fields(model)
        for task in sorted({t for (t, _) in F}):
            if (task, "long") not in F or (task, "single") not in F:
                continue
            L, S = F[(task, "long")], F[(task, "single")]
            if L.shape != S.shape:
                continue
            N = L.size
            rL = np.empty(N, int)
            rL[order(L)] = np.arange(N)          # rank of each block in LF
            rS = np.empty(N, int)
            rS[order(S)] = np.arange(N)
            uL, uS = topk_idx(L, a.k), topk_idx(S, a.k)
            row = dict(
                model=model, task=task, n_blocks=N, k_units=len(uL),
                pct_L_in_S=percentile_in(rS, uL),
                pct_S_in_L=percentile_in(rL, uS),
                auc_L_in_S=auc_of(rS, uL, N),
                auc_S_in_L=auc_of(rL, uS, N),
                jaccard=len(set(uL.tolist()) & set(uS.tolist()))
                / len(set(uL.tolist()) | set(uS.tolist())))
            row["asymmetry"] = row["auc_L_in_S"] - row["auc_S_in_L"]
            for m in MGRID:
                topL = set(order(L)[:int(round(m * N))].tolist())
                topS = set(order(S)[:int(round(m * N))].tolist())
                row[f"recall_L_in_S{m}"] = len(set(uL.tolist()) & topS) / len(uL)
                row[f"recall_S_in_L{m}"] = len(set(uS.tolist()) & topL) / len(uS)
            rows.append(row)

    if not rows:
        print("no cells")
        return
    with open(HERE / "containment_asymmetry.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"{len(rows)} (model, task) cells | k={a.k} | "
          f"selection = signed top-k, matching eval/logits_handler.py:92\n")

    print("=" * 100)
    print("1. WHERE DOES EACH ARM'S SELECTION SIT IN THE OTHER'S RANKING?")
    print("   0.5 = chance. Above chance means the other arm's field already")
    print("   ranks these blocks highly.")
    print("=" * 100)
    line("LF's picks, percentile in ST's field",
         [r["pct_L_in_S"] for r in rows], 0.5, "above chance")
    line("ST's picks, percentile in LF's field",
         [r["pct_S_in_L"] for r in rows], 0.5, "above chance")
    print()
    line("LF's picks, AUC under ST's scores",
         [r["auc_L_in_S"] for r in rows], 0.5, "above chance")
    line("ST's picks, AUC under LF's scores",
         [r["auc_S_in_L"] for r in rows], 0.5, "above chance")

    print("\n" + "=" * 100)
    print("2. THE ASYMMETRY   AUC(LF in ST) - AUC(ST in LF)")
    print("   positive -> ST's field contains LF's picks better than the reverse,")
    print("               i.e. ST looks like a blurred LF and LF's edge is denoising")
    print("   zero     -> symmetric: the arms localize different things")
    print("=" * 100)
    line("asymmetry", [r["asymmetry"] for r in rows], 0.0, "LF better contained")

    print("\n" + "=" * 100)
    print("3. RECALL CURVES: how much of one arm's top-k sits in the other's top-m")
    print("=" * 100)
    print(f"  {'m':>6}{'LF picks in ST top-m':>24}{'ST picks in LF top-m':>24}"
          f"{'difference':>13}")
    for m in MGRID:
        x = [r[f"recall_L_in_S{m}"] for r in rows]
        y = [r[f"recall_S_in_L{m}"] for r in rows]
        d = [p - q for p, q in zip(x, y)]
        print(f"  {m:>6}{st.mean(x):>24.3f}{st.mean(y):>24.3f}{st.mean(d):>+13.3f}")
    print(f"\n  for reference, chance recall at m equals m itself "
          f"({', '.join(str(m) for m in MGRID)})")

    print("\n" + "=" * 100)
    print("PER-CELL")
    print("=" * 100)
    print(f"  {'model':<14}{'task':<16}{'pctL>S':>9}{'pctS>L':>9}{'aucL>S':>9}"
          f"{'aucS>L':>9}{'asym':>9}{'jac':>7}")
    for r in sorted(rows, key=lambda r: -r["asymmetry"]):
        print(f"  {SHORT[r['model']]:<14}{r['task']:<16}{r['pct_L_in_S']:>9.3f}"
              f"{r['pct_S_in_L']:>9.3f}{r['auc_L_in_S']:>9.3f}"
              f"{r['auc_S_in_L']:>9.3f}{r['asymmetry']:>+9.3f}{r['jaccard']:>7.3f}")
    print("\nwrote containment_asymmetry.csv")


if __name__ == "__main__":
    main()
