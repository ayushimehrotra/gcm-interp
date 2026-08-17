#!/usr/bin/env python3
"""Layer-wise mechanics of the two localizations, with the arithmetic spelled out.

Section 1a'' showed long-form localization is effectively a LAYER selector: fix its
per-layer histogram, fill those layers with random units, and accuracy is unchanged
(-0.0001). So the layer histogram is the object that carries long-form's signal,
and this script characterises it.

THE OBJECTS
-----------
For a model with L layers and H units per layer, the attribution field is

    A in R^{L x H},   A[l,u] = mean over examples of the ATP score for unit (l,u)

`logits_handler.get_top_k_layer_and_head` selects by taking the largest SIGNED
values (not |A|), so the selected set at budget k is

    S(k) = the first  m = floor(k * L * H)  entries of argsort(-A, axis=None)

From S(k) the per-layer histogram and its normalisation are

    n(l) = |{ u : (l,u) in S(k) }|            counts, sum_l n(l) = m
    p(l) = n(l) / m                           a probability distribution over layers

Every quantity below is a functional of p:

    mean depth      d   = sum_l  l * p(l)                     (in layers)
    relative depth  d/(L-1)                                   (0 = first layer)
    spread          Hn  = -sum_l p(l) log p(l) / log L        (0 = one layer, 1 = uniform)
    layers used     |{ l : n(l) > 0 }|
    concentration   top-3 layer mass = sum of the 3 largest p(l)

COMPARING TWO DISTRIBUTIONS
---------------------------
Mean depth compares only the first moment. Two arms could share a mean and differ
in shape, so the profiles are also compared by 1-Wasserstein distance on relative
depth, which is the mass-transport cost and is sensitive to shape as well as
location:

    W1(p, q) = sum over the depth axis of |CDF_p - CDF_q| * (bin width)

W1 is scale-free once depth is normalised to [0,1], so it is comparable across
models with different L. It is read against a CROSS-TASK null -- W1 between
long(task A) and single(task B) in the same model -- which absorbs any model-wide
tendency for the two methods to sit at different depths regardless of task.
"""
import csv
import re
import statistics as st
import subprocess
from collections import defaultdict
from math import comb, log
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
K = 0.1
NBIN = 20

# ---------------------------------------------------------------------------
# Self-contained: attribution fields are read straight from the two checkouts.
#
# `numerator_1_targeted_1.0.csv` is the k=1.0 selection, i.e. EVERY unit with its
# signed attribution, so it is the complete field rather than a top-k slice.
#
# Two things have to be right or the grid is silently wrong:
#   OWNERSHIP  both repos contain a Qwen1.5-14B `paragraph` localization. Keying a
#              field store by (model, task, arm) alone lets one overwrite the other.
#              unified_results.csv settles it: every paper cell is contributed by
#              exactly one repo -- verse/summarization by ayushi, the rest by umang.
#   VERSION    some trees hold two attribution runs, split across the eval/steer
#              subdirectories by commit date (the verse-single dataset was
#              regenerated on 2026-08-03). Take the git-newest; averaging two runs
#              produces a field nobody ever ran.
# ---------------------------------------------------------------------------
REPOS = {"ayushi": Path("/home/ubuntu/gcm-interp"),
         "umang": Path("/home/ubuntu/gcm-interp-umang")}
OWNER = {"verse": "ayushi", "summarization": "ayushi",
         "bias": "umang", "factual recall": "umang", "persona": "umang"}
TASKNAME = {"verse": "verse", "paragraph": "summarization", "female": "bias",
            "lying": "factual recall", "extraversion": "persona"}
TASKS = ["verse", "summarization", "bias", "factual recall", "persona"]
PAPER_MODELS = {"gemma-3-12b-it", "Falcon3-10B-Instruct", "Qwen1.5-14B-Chat",
                "Qwen1.5-32B-Chat", "OLMo-2-1124-13B-DPO"}
SHORT = {"Falcon3-10B-Instruct": "Falcon3-10B", "OLMo-2-1124-13B-DPO": "OLMo-2-13B",
         "Qwen1.5-14B-Chat": "Qwen1.5-14B", "Qwen1.5-32B-Chat": "Qwen1.5-32B",
         "gemma-3-12b-it": "Gemma-3-12B"}
FROM_RE = re.compile(r"^from_(?P<src>.+?)-(?P<loc>long|single)_to_(?P<base>.+?)(?P<old>_old)?$")


def _read_field(path):
    rows = [(int(r["layer"]), int(r["neuron"]), float(r["value"]))
            for r in csv.DictReader(open(path))]
    if not rows:
        return None
    nl = max(r[0] for r in rows) + 1
    nu = max(r[1] for r in rows) + 1
    if len(rows) != nl * nu:          # not a complete field
        return None
    a = np.zeros((nl, nu))
    for l, u, v in rows:
        a[l, u] = v
    return a


def _git_date(repo, path):
    out = subprocess.run(["git", "-C", str(repo), "log", "-1", "--format=%ad",
                          "--date=short", "--", str(Path(path).relative_to(repo))],
                         capture_output=True, text=True).stdout.strip()
    return out or "0000-00-00"


def load_fields():
    """(model, task, arm) -> [n_layers, n_units] signed attribution field."""
    found = defaultdict(dict)
    for repo, root in REPOS.items():
        for p in root.rglob("numerator_1_targeted_1.0.csv"):
            parts = p.parts
            frm = next((c for c in parts if c.startswith("from_") and "_to_" in c), None)
            if frm is None:
                continue
            m = FROM_RE.match(frm)
            if not m or m.group("old"):
                continue
            model = parts[parts.index(frm) - 1]
            task = TASKNAME.get(m.group("src"))
            if model not in PAPER_MODELS or task is None:
                continue
            f = _read_field(p)
            if f is not None:
                found[(model, task, m.group("loc"))][p] = (repo, root, f)
    out = {}
    for key, cands in found.items():
        # OWNER breaks ties when a cell exists in both checkouts; when it exists
        # in only one, use that one rather than discarding the cell
        # (Falcon3-10B persona lives in ayushi's repo, not umang's)
        pref = {p: x for p, x in cands.items() if x[0] == OWNER.get(key[1])}
        use = pref or cands
        newest = max(use, key=lambda p: (_git_date(use[p][1], p), str(p)))
        out[key] = use[newest][2]
    return out


def selected(A, k=K):
    """S(k) as (layer, unit) index arrays -- the top signed-value entries."""
    m = int(k * A.size)
    idx = np.argsort(-A, axis=None, kind="stable")[:m]
    return idx // A.shape[1], idx % A.shape[1]


def layer_hist(A, k=K):
    L = A.shape[0]
    layers, _ = selected(A, k)
    n = np.bincount(layers, minlength=L).astype(float)
    return n, n / n.sum()


def stats(A, k=K):
    L = A.shape[0]
    n, p = layer_hist(A, k)
    nz = p[p > 0]
    return dict(
        n_layers_total=L,
        mean_depth=float((np.arange(L) * p).sum()),
        rel_depth=float((np.arange(L) * p).sum() / (L - 1)),
        entropy=float(-(nz * np.log(nz)).sum() / log(L)),
        layers_used=int((n > 0).sum()),
        top3_mass=float(np.sort(p)[::-1][:3].sum()),
        peak_layer=int(np.argmax(n)),
    )


def profile(A, k=K, nbin=NBIN):
    """p re-binned onto a common relative-depth axis so models with different L
    can be pooled."""
    L = A.shape[0]
    n, p = layer_hist(A, k)
    rel = np.arange(L) / (L - 1)
    out = np.zeros(nbin)
    edges = np.linspace(0, 1, nbin + 1)
    for i in range(nbin):
        lo, hi = edges[i], edges[i + 1]
        m = (rel >= lo) & (rel < hi if i < nbin - 1 else rel <= hi)
        out[i] = p[m].sum()
    return out


def w1(p, q):
    """1-Wasserstein on the normalised depth axis."""
    return float(np.abs(np.cumsum(p) - np.cumsum(q)).sum() / len(p))


def boot(ds, n=10000, seed=0):
    if len(ds) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    a = np.asarray(ds, float)
    m = np.sort(rng.choice(a, (n, len(a))).mean(1))
    return float(m[int(.025 * n)]), float(m[int(.975 * n)])


def sign_p(ds, tol=1e-12):
    pos = sum(1 for d in ds if d > tol); neg = sum(1 for d in ds if d < -tol)
    n = pos + neg
    if n == 0:
        return pos, neg, 1.0
    k = min(pos, neg)
    return pos, neg, min(sum(comb(n, i) for i in range(k + 1)) / 2 ** n * 2, 1.0)


def paired(label, ds, unit=""):
    lo, hi = boot(ds)
    pos, neg, p = sign_p(ds)
    print(f"  {label:<34}n={len(ds):>3}  {st.mean(ds):+.3f}{unit} [{lo:+.3f},{hi:+.3f}]"
          f"   long lower {neg}, single lower {pos}, tie {len(ds)-pos-neg}   p={p:.4f}")


def bar(p, width=44):
    m = p.max() or 1
    blocks = " .:-=+*#%@"
    return "".join(blocks[min(int(v / m * 9 + .5), 9)] for v in p)


def main():
    F = load_fields()
    cells = sorted({(m, t) for (m, t, l) in F
                    if (m, t, "long") in F and (m, t, "single") in F})
    print(f"{len(cells)} (model, task) cells | budget k = {K} | "
          f"selection = top {K:.0%} of L x H units by signed attribution\n")

    # ------------------------------------------------------------------ 1 ----
    print("=" * 116)
    print("1. PER-CELL LAYER STATISTICS")
    print("   mean depth = sum_l l*p(l);  spread = normalised entropy of p;")
    print("   top-3 mass = share of selected units in the 3 heaviest layers")
    print("=" * 116)
    print(f"  {'model':<13}{'task':<16}{'arm':<8}{'L':>4}{'units sel':>10}"
          f"{'mean depth':>12}{'rel':>7}{'spread':>8}{'layers':>8}{'top3':>7}{'peak L':>8}")
    rows = []
    for m, t in cells:
        for arm in ("long", "single"):
            s = stats(F[(m, t, arm)])
            s.update(model=m, task=t, arm=arm,
                     units_sel=int(K * F[(m, t, arm)].size))
            rows.append(s)
            print(f"  {SHORT.get(m,m):<13}{t:<16}{arm:<8}{s['n_layers_total']:>4}"
                  f"{s['units_sel']:>10}{s['mean_depth']:>12.1f}{s['rel_depth']:>7.3f}"
                  f"{s['entropy']:>8.3f}{s['layers_used']:>8}{s['top3_mass']:>7.3f}"
                  f"{s['peak_layer']:>8}")
        print()
    idx = {(r["model"], r["task"], r["arm"]): r for r in rows}

    # ------------------------------------------------------------------ 2 ----
    print("=" * 116)
    print("2. PAIRED long - single")
    print("=" * 116)
    for key, unit in (("mean_depth", " layers"), ("rel_depth", " of the stack"),
                      ("entropy", ""), ("layers_used", " layers"), ("top3_mass", "")):
        paired(key, [idx[(m, t, "long")][key] - idx[(m, t, "single")][key]
                     for m, t in cells], unit)

    # ------------------------------------------------------------------ 3 ----
    print("\n" + "=" * 116)
    print("3. POOLED DEPTH PROFILE  (p re-binned onto relative depth, averaged over cells)")
    print("=" * 116)
    PL = np.mean([profile(F[(m, t, "long")]) for m, t in cells], axis=0)
    PS = np.mean([profile(F[(m, t, "single")]) for m, t in cells], axis=0)
    print(f"  input {'-'*44} output")
    print(f"  long   [{bar(PL)}]")
    print(f"  single [{bar(PS)}]")
    print(f"\n  {'depth bin':<14}{'long p':>9}{'single p':>10}{'long - single':>15}"
          f"{'95% CI':>22}")
    edges = np.linspace(0, 1, NBIN + 1)
    for i in range(NBIN):
        d = [profile(F[(m, t, "long")])[i] - profile(F[(m, t, "single")])[i]
             for m, t in cells]
        lo, hi = boot(d)
        star = "  <--" if lo * hi > 0 else ""
        print(f"  {f'{edges[i]:.2f}-{edges[i+1]:.2f}':<14}{PL[i]:>9.3f}{PS[i]:>10.3f}"
              f"{st.mean(d):>+15.3f}   [{lo:+.3f},{hi:+.3f}]{star}")
    print("  <-- marks bins whose 95% CI excludes zero")

    # ------------------------------------------------------------------ 4 ----
    print("\n" + "=" * 116)
    print("4. SHAPE, NOT JUST LOCATION: 1-Wasserstein between layer profiles")
    print("   Decomposed properly. A null of long(task A) vs single(task B) is NOT a null:")
    print("   it still contains the long/single contrast, so it cannot isolate it. The")
    print("   comparison that does is method-effect against pure task-effect:")
    print("     METHOD  W1(long_A, single_A)  same task, different method")
    print("     TASK    W1(long_A, long_B) and W1(single_A, single_B)  same method, diff task")
    print("=" * 116)
    method, task_eff = [], []
    for m in sorted({c[0] for c in cells}):
        tasks_m = [t for (mm, t) in cells if mm == m]
        for t in tasks_m:
            method.append(w1(profile(F[(m, t, "long")]), profile(F[(m, t, "single")])))
        for arm in ("long", "single"):
            for i, a in enumerate(tasks_m):
                for b in tasks_m[i + 1:]:
                    task_eff.append(w1(profile(F[(m, a, arm)]), profile(F[(m, b, arm)])))
    lo_m, hi_m = boot(method); lo_t, hi_t = boot(task_eff)
    print(f"  METHOD effect  W1(long, single), same task     n={len(method):>3}  "
          f"mean {st.mean(method):.4f} [{lo_m:.4f},{hi_m:.4f}]")
    print(f"  TASK effect    W1 between tasks, same method   n={len(task_eff):>3}  "
          f"mean {st.mean(task_eff):.4f} [{lo_t:.4f},{hi_t:.4f}]")
    print(f"\n  ratio method/task = {st.mean(method)/st.mean(task_eff):.2f}")
    print("  <1 means swapping the localization method moves the layer profile LESS")
    print("  than swapping which concept is being localized.")

    # per-model, since L differs
    print(f"\n  {'model':<14}{'method W1':>12}{'task W1':>10}{'ratio':>8}")
    for m in sorted({c[0] for c in cells}):
        tasks_m = [t for (mm, t) in cells if mm == m]
        me = [w1(profile(F[(m, t, "long")]), profile(F[(m, t, "single")])) for t in tasks_m]
        ta = [w1(profile(F[(m, a, arm)]), profile(F[(m, b, arm)]))
              for arm in ("long", "single")
              for i, a in enumerate(tasks_m) for b in tasks_m[i + 1:]]
        if me and ta:
            print(f"  {SHORT.get(m,m):<14}{st.mean(me):>12.4f}{st.mean(ta):>10.4f}"
                  f"{st.mean(me)/st.mean(ta):>8.2f}")

    with open(HERE / "layerwise_mechanics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print("\nwrote layerwise_mechanics.csv")


if __name__ == "__main__":
    main()
