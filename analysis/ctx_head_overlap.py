#!/usr/bin/env python
"""Head-set overlap between --localization_ctx cells.

Answers the structural half of the question: does changing WHICH sequences ATP
differences actually move the heads it selects, or is the 2x2 a no-op?

Jaccard is reported against the chance baseline for two independent draws of the
same size: for a selected fraction f, E[J] = f / (2 - f). Layer-level Jaccard is
reported alongside because CLAUDE.md's existing finding is that localizations
agree far more on layers (0.469) than on heads (0.058) -- if the ctx cells
differ only in head identity and not in depth, that is the same pattern.

Usage: python analysis/ctx_head_overlap.py [--model Qwen1.5-14B-Chat]
"""
import argparse
import glob
import itertools
import os

import pandas as pd

GRID = os.environ.get("CTX_GRID", "fixed")
PREFIX = {"legacy": "atp-", "fixed": "atp-o_proj_in-respfix-"}[GRID]
CTX_DIRS = {
    'br-sq': PREFIX + 'brsq',
    'br-sr': PREFIX + 'srcresp',
    'bq-sq': PREFIX + 'baseq',
    'bq-sr': PREFIX + 'baseq-srcresp',
}


def head_set(root, ctx_dir, loc, topk):
    pat = (f"{root}/from_{loc}_to_prose/{ctx_dir}/{loc.split('-')[0]}-long_eval/"
           f"*_steer/eval/numerator_1_targeted_{topk}.csv")
    hits = sorted(glob.glob(pat))
    if not hits:
        return None
    df = pd.read_csv(hits[0])
    return set(zip(df['layer'].tolist(), df['neuron'].tolist()))


def jaccard(a, b):
    if not a and not b:
        return float('nan')
    return len(a & b) / len(a | b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='Qwen1.5-14B-Chat')
    ap.add_argument('--results', default='./results')
    ap.add_argument('--topks', default='0.01,0.03,0.05,0.07,0.09,0.1')
    args = ap.parse_args()

    root = f"{args.results}/{args.model}"
    topks = [t.strip() for t in args.topks.split(',')]
    rows = []
    for loc in ('verse-long', 'verse-single'):
        for topk in topks:
            sets = {}
            for ctx, d in CTX_DIRS.items():
                s = head_set(root, d, loc, topk)
                if s:
                    sets[ctx] = s
            f = float(topk)
            chance = f / (2 - f)
            for a, b in itertools.combinations(sorted(sets), 2):
                heads_j = jaccard(sets[a], sets[b])
                layers_j = jaccard({l for l, _ in sets[a]}, {l for l, _ in sets[b]})
                rows.append(dict(loc=loc, topk=f, pair=f"{a} vs {b}",
                                 n=len(sets[a]), heads_jaccard=round(heads_j, 4),
                                 chance=round(chance, 4),
                                 layers_jaccard=round(layers_j, 4)))
    if not rows:
        print("No head-set CSVs found yet -- the sweep has not written any.")
        return
    df = pd.DataFrame(rows)
    out = f"{root}/ctx_head_overlap_{GRID}.csv"
    df.to_csv(out, index=False)
    for loc, g in df.groupby('loc'):
        print(f"\n=== {loc}")
        piv = g.pivot_table(index='pair', columns='topk', values='heads_jaccard')
        print("head Jaccard (chance in [] per k):")
        print(piv.to_string())
        print("chance:", {k: round(k / (2 - k), 4) for k in sorted(g['topk'].unique())})
        print("\nlayer Jaccard:")
        print(g.pivot_table(index='pair', columns='topk', values='layers_jaccard').to_string())
    print(f"\nwrote {out}")


if __name__ == '__main__':
    main()
