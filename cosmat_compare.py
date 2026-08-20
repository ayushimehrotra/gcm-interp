#!/usr/bin/env python3
"""Side-by-side table of the two extraction points, per shaping and n.

`svcca_cosmat.py --extract prompt` and `--extract response` write to separate
outdirs. This reads both `cosmat_stats.csv` files and prints the comparison the
CLAUDE.md 8.9 argument actually needs: prompt activations are purely inputs to
the generation, response activations are partly a CONSEQUENCE of it, so an
alignment that only appears in the response arm is not evidence that the two
localizations select equivalent *inputs*.

Columns:
    |diag|   mean |cos| on the diagonal -- how index-aligned the correspondence
             is. Low |diag| with high rho5 means a ROTATION: same subspace,
             different axes, which SVCCA cannot distinguish.
    rowmax   mean over rows of the best cosine in that row (greedy match).
    rho5     mean of the top-5 canonical correlations = SVCCA at rank n.
    floor    the same statistic with one side's rows permuted.
    excess   rho5 - floor. The only column that is evidence.

USAGE
    python cosmat_compare.py
    python cosmat_compare.py --a cosmat_figs_prompt --b cosmat_figs
"""
import argparse
import csv
from pathlib import Path

REPO = Path(__file__).resolve().parent
TASKS = ["verse", "summarization", "persona", "bias", "factual recall"]
SHAPINGS = ["bs_h", "s_h", "b_last_h", "b_sh"]
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


def main():
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
    for t in TASKS:
        for sh in SHAPINGS:
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
    for sh in SHAPINGS:
        va, vb = [], []
        for t in TASKS:
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


if __name__ == "__main__":
    main()
