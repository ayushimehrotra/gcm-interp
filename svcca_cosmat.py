#!/usr/bin/env python3
"""Cosine-similarity matrices between the two arms' left singular vectors.

WHY THIS EXISTS
---------------
`svcca_sweep.py` reduces the cross-arm geometry to ONE NUMBER per (shaping,
measure, rank) -- the mean of the top-5 canonical correlations, or CKA, or
PWCCA. That number is a summary of an object it never shows you:

    C = U_L[:, :n].T @ U_S[:, :n]

the n x n matrix of cosines between the long-arm and single-arm left singular
vectors, both taken in SAMPLE space. `svcca_sweep.canonical_correlations` says
so in its own docstring -- the canonical correlations ARE the singular values
of C. So SVCCA at rank n is `mean(svd(C)[:5])`, and everything else about C is
discarded: whether the alignment is diagonal (component i of one arm maps to
component i of the other) or rotational (the two arms span the same subspace but
carve it up along different axes) produces the SAME SVCCA score.

This script draws C instead of summarising it, for three shapings of the
prompt-side activations and for a sweep of n.

THE THREE SHAPINGS
------------------
Every prompt token is collected -- prompts are variable length, so the
collection is RAGGED (a list of [T_i, H]) rather than a rectangular [B, S, H].
That is the point: nothing is dropped to make a reshape work.

    bs_h      (sum T_i, H)   every prompt token of every prompt is a row.
                             ~6k-11k rows here. Most rows, least independent:
                             40 tokens from one prompt are near-duplicates.
    s_h       (S, H)         averaged over prompts within polarity. Prompts
                             differ in length, so they are aligned from the
                             RIGHT -- position -1 is the last prompt token in
                             every prompt -- and truncated to the shortest
                             prompt. Rows = 2 * S_min, which is small (54-94).
    b_last_h  (B, H)         T[:, -1, :]: the activation at the LAST prompt
                             token, one row per prompt. The pre-generation
                             state, the moment eval/activations.py builds
                             steering vectors from, and the row that s_h's last
                             position is the batch-mean of.

Right-alignment is what makes the three coherent: s_h position -1 is the mean
of what b_last_h takes per prompt.

WHAT TO LOOK FOR
----------------
    bright diagonal        the arms agree component-by-component; the SVD
                           orderings match, so "same subspace" is also
                           "same axes"
    bright off-diagonal    same subspace, different basis -- a rotation. SVCCA
                           reports this identically to the diagonal case
    bright block near the  the leading (high-variance) components align and the
    top-left corner        tail is noise; the honest r is where the block ends
    bright everywhere      degeneracy, not agreement -- see below

SIGN IS ARBITRARY
-----------------
Left singular vectors are defined up to sign: flipping u_i and v_i both flips
the sign of C[i, j] for a whole row/column without changing any angle. So the
signed panel is what was literally asked for, and `--abs` (|C|, the principal
angles themselves) is the panel that is actually interpretable. Both are
written. Read the signed one for block STRUCTURE and the absolute one for
magnitude.

THE DEGENERACY TRAP
-------------------
Two n-dimensional subspaces of an (n_samples - 1)-dimensional space are FORCED
to intersect once 2n >= n_samples - 1, and inside that region the leading
cosines are pinned at 1 on ANY data -- real or shuffled. This bites the `s_h`
shaping hard: it has only 2 * S_min rows (54-94 here), so n=50 is always
inside it, and b_last_h needs --n_items 100 (200 rows) for n=50 to be legal. Those panels are drawn with a red border and labelled
DEGENERATE; the brightness in them is an artefact of dimension counting, not a
finding. `--shuffle_control` draws the same grid with one side's rows permuted,
which makes the point visually: wherever the control looks like the observation,
the observation is measuring nothing.

USAGE
-----
    python svcca_cosmat.py --dry_run
    python svcca_cosmat.py --models Qwen1.5-14B-Chat --tasks verse
    python svcca_cosmat.py --models Qwen1.5-14B-Chat --tasks verse --shuffle_control
    python svcca_cosmat.py --ns 10,20,50 --n_items 100
    python svcca_cosmat.py --replot --outdir cosmat_figs_all   # no GPU
"""
import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm, Normalize

from svcca_sweep import (REPO, MODELS, STEMS, _basis, _centre,
                         canonical_correlations, degenerate, prep,
                         runnable_cells, top_units)

# Ordering for the figure rows. NOT svcca_sweep.SHAPINGS: the third shaping here
# is b_last_h = T[:, -1, :], the last prompt token, not b_sh = (B, S*H).
SHAPINGS = ("bs_h", "s_h", "b_last_h")

# ------------------------------------------------------------------ palette
# From the data-viz reference palette. Diverging = blue <-> red with a NEUTRAL
# GRAY midpoint (never a hue at the midpoint); sequential = the single blue
# ramp, light -> dark. Both are listed light-first; the dark variants are the
# same hues re-stepped for the dark surface, not an inversion.
INK = {"light": dict(surface="#fcfcfb", plane="#f9f9f7", primary="#0b0b0b",
                     secondary="#52514e", muted="#898781", grid="#e1e0d9",
                     axis="#c3c2b7", mid="#f0efec"),
       "dark": dict(surface="#1a1a19", plane="#0d0d0d", primary="#ffffff",
                    secondary="#c3c2b7", muted="#898781", grid="#2c2c2a",
                    axis="#383835", mid="#383835")}

SEQ_BLUE = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
            "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281",
            "#0d366b"]
DIV_RED = ["#7a1f1f", "#a02c2c", "#c03636", "#d03b3b", "#e34948", "#e87070",
           "#f0a0a0", "#f6cccc"]


def make_cmaps(mode):
    """(diverging, sequential) colormaps. Midpoint is gray, never a hue.

    Equal step count per arm, blue for negative and red for positive, so the
    two poles read as opposites and zero recedes into the surface.
    """
    mid = INK[mode]["mid"]
    blue_arm = [SEQ_BLUE[-1], SEQ_BLUE[-4], SEQ_BLUE[-7], SEQ_BLUE[2]]
    red_arm = [DIV_RED[-1], DIV_RED[-4], DIV_RED[3], DIV_RED[0]]
    div = LinearSegmentedColormap.from_list(
        "blue_gray_red", blue_arm + [mid] + red_arm)
    seq = LinearSegmentedColormap.from_list("blue_seq", SEQ_BLUE)
    return div, seq


# ------------------------------------------------------------ collection
def collect_prompt_tokens(mh, toks, starts, blocks, D, bs, torch, limit=None):
    """EVERY prompt token of every prompt, ragged: a list of [T_i, H] arrays.

    `svcca_sweep.collect_structured` returns a rectangular [B, S, H] and to do
    that it must drop prompts that cannot supply S positions. Prompts are
    naturally 27-66 tokens here, so a fixed S either throws prompts away or
    throws tokens away. Ragged keeps both; the shaping functions decide how to
    rectangularise, and only s_h ever needs to.

    Tokens are the ones BEFORE the response starts. Padding is left-side
    (model_handler sets padding_side='left'), so the attention mask -- not index
    0 -- marks where each prompt actually begins.

    `limit` caps tokens per prompt, taken from the END (nearest the response) if
    set; None means all of them.
    """
    # Flat column index into the (layer, hidden) plane, in `blocks` order. This
    # reproduces torch.cat([v[l, D*u:D*(u+1)] for l, u in blocks]) exactly --
    # flattening [n_layers, hidden] puts A[l, ..., c] at l*hidden + c -- but as
    # one fancy-index instead of a per-token Python loop over ~155 blocks.
    seqs = []
    flat = None
    with torch.no_grad():
        for i in range(0, toks["input_ids"].shape[0], bs):
            sl = slice(i, i + bs)
            batch = {k: v[sl].to(mh.device) for k, v in toks.items()}
            with mh.model.trace(batch):
                saved = [l.self_attn.o_proj.output.detach().cpu().save()
                         for l in mh.model.model.layers]
            A = torch.stack([x.to(torch.float32) for x in saved])  # [L,B,T,hid]
            L, Bb, Tt, hid = A.shape
            if flat is None:
                flat = torch.cat([torch.arange(l * hid + D * u,
                                               l * hid + D * (u + 1))
                                  for l, u in blocks])
            A2 = A.permute(1, 2, 0, 3).reshape(Bb, Tt, L * hid)
            am = toks["attention_mask"][sl]
            for j, s_ in enumerate(starts[sl]):
                take = [t for t in range(0, int(s_)) if am[j, t] == 1]
                if not take:
                    seqs.append(None)
                    continue
                if limit is not None and len(take) > limit:
                    take = take[-limit:]
                seqs.append(A2[j][take][:, flat].numpy())     # [T_i, H_union]
            del A, A2, saved
    return seqs


def build_arms_all(seqs, n_items, shaping, cL, cS):
    """(XL, XS, note) for one shaping, from the ragged per-prompt activations.

    `seqs` is [desired..., undesired...] with n_items desired prompts first.
    `note` records what the shaping had to discard, so a panel never silently
    hides a truncation.
    """
    B = len(seqs)
    if shaping == "bs_h":
        M = np.concatenate(seqs, 0)                       # (sum T_i, H)
        note = f"{B} prompts, {M.shape[0]} tokens"
    elif shaping == "b_last_h":
        M = np.stack([q[-1] for q in seqs])               # (B, H)
        note = f"{B} prompts, last prompt token"
    elif shaping == "s_h":
        # Right-aligned: position -1 is the last prompt token in EVERY prompt,
        # so the average is over comparable positions rather than over whatever
        # happened to land at index k. Truncated to the shortest prompt.
        S = min(len(q) for q in seqs)
        A = np.stack([q[-S:] for q in seqs])              # (B, S, H)
        M = np.concatenate([A[:n_items].mean(0), A[n_items:].mean(0)])
        note = (f"S_min={S} of "
                f"{min(len(q) for q in seqs)}-{max(len(q) for q in seqs)}")
    else:
        raise ValueError(shaping)
    return M[:, cL], M[:, cS], note


def bases(X, max_n):
    """Left singular vectors once, sliced for every n -- U does not depend on n.

    _basis(X, n) is U[:, :n] of the SAME decomposition for every n, so computing
    it per n would repeat a (6000 x 10240) SVD three times over.
    """
    return _basis(X, max_n)


def cosmat_from_bases(QL, QS, n, perm=None):
    """n x n cosines from precomputed bases; `perm` gives the shuffle control.

    Permuting the rows of a matrix permutes the rows of its left-singular basis
    exactly -- centring is permutation-invariant and P*U is still orthonormal --
    so the shuffled control is QS[perm] and needs no second SVD. Verified
    against a fresh SVD of the permuted matrix: principal angles agree to 1e-15.
    """
    k = min(QL.shape[1], QS.shape[1], n)
    if k == 0:
        return None, 0
    Y = QS[perm] if perm is not None else QS
    return QL[:, :k].T @ Y[:, :k], k


# ------------------------------------------------------------------ geometry
def cosine_matrix(X, Y, n):
    """n x n cosines between the two arms' top-n left singular vectors.

    Columns of each basis are orthonormal, so the dot product IS the cosine --
    no renormalisation. Returns the matrix and the actual n kept (a rank-
    deficient arm can supply fewer than n directions).
    """
    Qx, Qy = _basis(X, n), _basis(Y, n)
    k = min(Qx.shape[1], Qy.shape[1])
    if k == 0:
        return None, 0
    return Qx[:, :k].T @ Qy[:, :k], k


def panel_stats(C):
    """Numbers that back up what the picture shows.

    diag_mean       how component-by-component the alignment is
    rowmax_mean     greedy best partner per long-arm component. CLAUDE.md 8.10
                    measures 0.477 here against a diagonal of 0.248 and
                    singular values of 0.986 -- the gap between rowmax_mean and
                    svcca5 is exactly how ROTATIONAL the correspondence is, and
                    it is the reason the entrywise picture looks like noise
                    while the scalar is decisive
    offdiag_max     the largest cross-component cosine; if this beats the
                    diagonal the alignment is rotational, not index-aligned
    rho             the singular values SVCCA would have averaged
    """
    A = np.abs(C)
    k = C.shape[0]
    off = A.copy()
    np.fill_diagonal(off, 0.0)
    rho = np.clip(np.linalg.svd(C, compute_uv=False), 0, 1)
    return dict(
        k=k,
        diag_mean=float(np.mean(np.abs(np.diag(C)))),
        diag_max=float(np.max(np.abs(np.diag(C)))),
        offdiag_mean=float(off.sum() / max(k * k - k, 1)),
        offdiag_max=float(off.max()) if k > 1 else 0.0,
        rowmax_mean=float(np.mean(A.max(1))),
        frob_frac=float((A ** 2).sum() / k),       # 1.0 == subspaces identical
        svcca5=float(np.mean(rho[:min(5, k)])),
        rho=rho)


# ------------------------------------------------------------------ plotting
def draw_grid(mats, title, subtitle, path, mode, absolute):
    """One figure: rows = shapings, cols = n. Shared scale, one legend.

    `mats` is {(shaping, n): (C, meta)}; meta carries n_samples, degeneracy and
    the stats dict. Panels that could not be computed are drawn as an explicit
    "not available" cell rather than silently dropped, so a missing panel is
    never mistaken for a blank one.
    """
    ink = INK[mode]
    div, seq = make_cmaps(mode)
    cmap = seq if absolute else div
    norm = Normalize(0, 1) if absolute else TwoSlopeNorm(vcenter=0, vmin=-1, vmax=1)

    shapings = sorted({k[0] for k in mats}, key=SHAPINGS.index)
    ns = sorted({k[1] for k in mats})
    nr, nc = len(shapings), len(ns)

    two_line = any(e[1]["degenerate"] for e in mats.values() if e[0] is not None)
    fig, axes = plt.subplots(
        nr, nc,
        figsize=(3.05 * nc + 1.45,
                 3.05 * nr + 1.4 + (0.14 if two_line else 0)),
        squeeze=False)
    fig.patch.set_facecolor(ink["plane"])

    im = None
    for i, sh in enumerate(shapings):
        for j, n in enumerate(ns):
            ax = axes[i][j]
            ax.set_facecolor(ink["surface"])
            entry = mats.get((sh, n))
            if entry is None or entry[0] is None:
                ax.text(0.5, 0.5, "not available", ha="center", va="center",
                        color=ink["muted"], fontsize=9, transform=ax.transAxes)
                ax.set_title(f"{sh}   n={n}", fontsize=9.5,
                             color=ink["primary"], pad=20, loc="left")
                ax.set_xticks([]); ax.set_yticks([])
                for s in ax.spines.values():
                    s.set_color(ink["axis"])
                continue
            C, meta = entry
            M = np.abs(C) if absolute else C
            im = ax.imshow(M, cmap=cmap, norm=norm, interpolation="nearest",
                           aspect="equal")

            k = C.shape[0]
            step = 1 if k <= 12 else 4
            ticks = list(range(0, k, step))
            ax.set_xticks(ticks); ax.set_yticks(ticks)
            ax.set_xticklabels([str(t + 1) for t in ticks], fontsize=7,
                               color=ink["muted"])
            ax.set_yticklabels([str(t + 1) for t in ticks], fontsize=7,
                               color=ink["muted"])
            ax.tick_params(length=0)
            # hairline cell separators: 2px of surface between fills
            ax.set_xticks(np.arange(-.5, k, 1), minor=True)
            ax.set_yticks(np.arange(-.5, k, 1), minor=True)
            ax.grid(which="minor", color=ink["surface"], linewidth=0.6)
            ax.tick_params(which="minor", length=0)

            deg = meta["degenerate"]
            for s in ax.spines.values():
                s.set_color("#d03b3b" if deg else ink["axis"])
                s.set_linewidth(1.8 if deg else 0.8)
            st = meta["stats"]
            head = f"{sh}   n={k}"
            if deg:
                head += "   ⚠ DEGENERATE"
            ax.set_title(head, fontsize=9.5, color=ink["primary"], pad=20,
                         loc="left")
            ax.text(0.0, 1.012, f"rows={meta['n_samples']}   "
                                f"|diag|={st['diag_mean']:.2f}   "
                                f"row-max={st['rowmax_mean']:.2f}   "
                                f"ρ̄₅={st['svcca5']:.2f}",
                    transform=ax.transAxes, fontsize=7.3, color=ink["muted"],
                    va="bottom")
            if i == nr - 1:
                ax.set_xlabel("single-arm component", fontsize=8.5,
                              color=ink["secondary"])
            if j == 0:
                ax.set_ylabel("long-arm component", fontsize=8.5,
                              color=ink["secondary"])

    fig.suptitle(title, fontsize=13, color=ink["primary"], x=0.012, ha="left",
                 y=0.985)
    fig.text(0.012, 0.947, subtitle, fontsize=9, color=ink["secondary"],
             ha="left", va="top")
    # positioned in INCHES, not axes fractions: the same fractions that fit a
    # 3-column grid clip the colorbar label on a 2-column one
    fig_w = fig.get_size_inches()[0]
    fig.tight_layout(rect=[0, 0.035, (fig_w - 1.45) / fig_w, 0.925])

    if im is not None:
        cax = fig.add_axes([(fig_w - 1.30) / fig_w, 0.10, 0.13 / fig_w, 0.62])
        cb = fig.colorbar(im, cax=cax)
        cb.set_label("|cos| between singular vectors" if absolute
                     else "cos between singular vectors",
                     fontsize=8.5, color=ink["secondary"])
        cb.ax.tick_params(labelsize=7.5, colors=ink["muted"], length=0)
        cb.outline.set_edgecolor(ink["axis"])
        cb.outline.set_linewidth(0.6)

    # the degeneracy sentence only earns its place when a panel is degenerate
    note = ("Singular vectors are sign-arbitrary, so a negative cell means the "
            "same angle as its positive twin; read |cos| for magnitude.")
    if two_line:
        note += ("\nRed border = 2n ≥ rows−1, where the leading cosines "
                 "are pinned at 1 on any data, real or shuffled.")
    fig.text(0.012, 0.012, note, fontsize=7.2, color=ink["muted"],
             ha="left", va="bottom", linespacing=1.5)
    fig.savefig(path, dpi=170, facecolor=ink["plane"])
    plt.close(fig)
    return path


def draw_spectrum(mats, title, path, mode):
    """The singular values of every C, i.e. the canonical-correlation spectra.

    This is the quantitative companion to the heatmaps: the heatmap shows WHERE
    the alignment lives, this shows HOW MUCH there is and where it falls off.
    One line per (shaping, n); shaping picks the hue, n picks the marker, so
    identity never rests on colour alone.
    """
    ink = INK[mode]
    hues = {"bs_h": "#2a78d6", "s_h": "#eb6834", "b_last_h": "#1baf7a"}
    all_ns = sorted({k[1] for k in mats})
    marks = {n: m for n, m in zip(all_ns, ["o", "s", "^", "D", "v", "P"])}
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    fig.patch.set_facecolor(ink["plane"])
    ax.set_facecolor(ink["surface"])
    seen = set()
    for (sh, n), (C, meta) in sorted(mats.items(),
                                     key=lambda kv: (SHAPINGS.index(kv[0][0]),
                                                     kv[0][1])):
        if C is None:
            continue
        rho = meta["stats"]["rho"]
        style = "--" if meta["degenerate"] else "-"
        ax.plot(np.arange(1, len(rho) + 1), rho, style,
                color=hues.get(sh, "#4a3aa7"),
                marker=marks.get(n, "o"), markersize=4.5, linewidth=1.8,
                markeredgecolor=ink["surface"], markeredgewidth=0.7,
                label=f"{sh}  n={n}" + (" (degenerate)" if meta["degenerate"]
                                        else ""))
        seen.add(sh)
    ax.set_xlabel("canonical direction", fontsize=9.5, color=ink["secondary"])
    ax.set_ylabel("canonical correlation ρ", fontsize=9.5,
                  color=ink["secondary"])
    ax.set_ylim(0, 1.02)
    ax.grid(axis="y", color=ink["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(ink["axis"]); ax.spines[s].set_linewidth(0.8)
    ax.tick_params(labelsize=8, colors=ink["muted"], length=0)
    ax.set_title(title, fontsize=12, color=ink["primary"], loc="left", pad=10)
    leg = ax.legend(fontsize=8, frameon=False, labelcolor=ink["secondary"],
                    loc="lower left", ncol=2)
    for t in leg.get_texts():
        t.set_color(ink["secondary"])
    fig.tight_layout()
    fig.savefig(path, dpi=170, facecolor=ink["plane"])
    plt.close(fig)
    return path


# ---------------------------------------------------------------------- main
def replot(outdir, mode, ns_filter=None):
    """Redraw every figure from the saved .npz + cosmat_stats.csv -- no GPU.

    Collecting activations is the expensive half and the matrices are already on
    disk, so a change to the FIGURE should never mean re-running the model. The
    npz carries C itself; the CSV carries the row count and degeneracy flag that
    the panels annotate. Everything else is recomputed from C.
    """
    outdir = Path(outdir)
    csv_path = outdir / "cosmat_stats.csv"
    if not csv_path.exists():
        print(f"no {csv_path} -- nothing to replot")
        return
    meta_by = {}
    for r in csv.DictReader(open(csv_path)):
        meta_by[(r["model"], r["task"], r["arm"], r["shaping"], int(r["n"]))] = r

    for npz_path in sorted(outdir.glob("cosmat_*.npz")):
        stem = npz_path.stem[len("cosmat_"):]
        model, rest = stem.split("_", 1)
        task_tag, extract = rest.rsplit("_", 1)
        task = task_tag.replace("-", " ")
        z = np.load(npz_path)
        mats, ctrl = {}, {}
        for key in z.files:
            shuffled = key.startswith("shuf_")
            sh, ntag = (key[5:] if shuffled else key).rsplit("_n", 1)
            n = int(ntag)
            if ns_filter and n not in ns_filter:
                continue
            row = meta_by.get((model, task, "shuffled" if shuffled else
                               "observed", sh, n))
            if row is None:
                print(f"  {npz_path.name}: no stats row for {key}, skipped")
                continue
            C = z[key]
            entry = (C, dict(n_samples=int(row["rows"]),
                             degenerate=bool(int(row["degenerate"])),
                             stats=panel_stats(C)))
            (ctrl if shuffled else mats)[(sh, n)] = entry
        if not mats:
            continue
        # Rebuild the real caption from the stats rows rather than naming the
        # npz: bs_h's row count IS the token total and b_last_h's IS the prompt
        # count, so the run parameters are recoverable without re-running.
        def _rows(sh):
            e = next((v for (s_, _), v in mats.items() if s_ == sh), None)
            return e[1]["n_samples"] if e else None
        toks, prompts = _rows("bs_h"), _rows("b_last_h")
        bits = ["long vs single localisation", "k=0.05"]
        if prompts:
            bits.append(f"{prompts // 2} prompts/polarity")
        bits.append(f"every prompt token ({toks} total)" if toks
                    else "prompt tokens")
        bits.append("rows differ by shaping")
        sub = " \u00b7 ".join(bits)
        tag = f"{model}_{task_tag}_{extract}"
        for absolute, suffix in ((False, "signed"), (True, "abs")):
            print("  wrote", draw_grid(
                mats, f"Cosine similarity between left singular vectors "
                      f"— {model} / {task}", sub,
                outdir / f"cosmat_{tag}_{suffix}.png", mode, absolute))
        print("  wrote", draw_spectrum(
            mats, f"Canonical correlation spectra — {model} / {task}",
            outdir / f"spectrum_{tag}.png", mode))
        if ctrl:
            for absolute, suffix in ((False, "signed"), (True, "abs")):
                print("  wrote", draw_grid(
                    ctrl, f"SHUFFLE CONTROL — rows permuted — {model} / {task}",
                    "one side's rows permuted: whatever survives is what "
                    "dimension alone manufactures",
                    outdir / f"cosmat_{tag}_shuffled_{suffix}.png",
                    mode, absolute))


def write_stats(outdir, rows):
    """Flush the per-panel numbers. Called after every cell, not just at the end."""
    if not rows:
        return None
    out = outdir / "cosmat_stats.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return out


def main():
    ap = argparse.ArgumentParser(
        description="Cosine-similarity matrices between the arms' left "
                    "singular vectors, per shaping and per n")
    ap.add_argument("--models", default="Qwen1.5-14B-Chat")
    ap.add_argument("--tasks", default="verse")
    ap.add_argument("--ns", default="10,20,50",
                    help="how many leading components to compare")
    ap.add_argument("--shapings", default=",".join(SHAPINGS))
    ap.add_argument("--k", type=float, default=0.05)
    ap.add_argument("--n_items", type=int, default=100,
                    help="prompts per polarity; 100 is the whole file, and "
                         "b_last_h needs it for n=50 to clear degeneracy")
    ap.add_argument("--save_bases", type=int, default=0,
                    help="also store the first N left singular vectors per arm "
                         "in the .npz, so any n can be swept later without "
                         "re-running the model; 0 = off")
    ap.add_argument("--max_tokens", type=int, default=0,
                    help="cap prompt tokens per prompt, taken from the end; "
                         "0 = every prompt token (the default)")
    ap.add_argument("--shuffle_control", action="store_true",
                    help="also draw the grid with one side's rows permuted")
    ap.add_argument("--mode", default="light", choices=("light", "dark"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--full_precision", action="store_true",
                    help="bf16 instead of the default nf4 quantisation")
    ap.add_argument("--extra_repo", default=None)
    ap.add_argument("--outdir", default="cosmat_figs")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--replot", action="store_true",
                    help="redraw from the saved .npz + stats csv; no model load")
    a = ap.parse_args()

    roots = [REPO] + ([Path(a.extra_repo).resolve()] if a.extra_repo else [])
    models = [m for m in a.models.split(",") if m in MODELS]
    tasks = [t.strip() for t in a.tasks.split(",")]
    ns = sorted(int(x) for x in a.ns.split(","))
    shapings = [s.strip() for s in a.shapings.split(",") if s.strip() in SHAPINGS]

    outdir = REPO / a.outdir
    if a.replot:
        replot(outdir, a.mode, set(ns))
        return
    cells = runnable_cells(models, tasks, roots)
    print(f"shapings : {shapings}")
    print(f"ns       : {ns}")
    print(f"outdir   : {outdir}")
    print(f"\n{len(cells)} runnable cells:")
    for model, task, _, _, _ in cells:
        print(f"    {model:<24}{task}")
    if a.dry_run:
        return
    if not cells:
        print("\nnothing to run")
        return
    outdir.mkdir(parents=True, exist_ok=True)

    import torch
    sys.path.insert(0, str(REPO))
    from model_handler import ModelHandler

    stat_rows = []
    by_model = defaultdict(list)
    for c in cells:
        by_model[c[0]].append(c)

    for model, cs in by_model.items():
        # One load per model, not one per task. `source` only reaches the marker
        # choice, and for every task in this grid it lands on the same branch --
        # svcca_sweep.main takes the first cell's stem for the same reason.
        cfg = SimpleNamespace(args=SimpleNamespace(
            model_id=MODELS[model], device=a.device, pyreft=False,
            full_precision=a.full_precision, source=STEMS[cs[0][1]]))
        mh = ModelHandler(cfg)
        D = mh.dim
        print(f"\n{'='*100}\n{model}   block width D={D}   k={a.k}   "
              f"{len(cs)} tasks\n{'='*100}", flush=True)

        for _, task, F, fd, fu in cs:
            print(f"\n{'-'*100}\n{model} / {task}\n{'-'*100}", flush=True)
            L, S_ = F[(task, "long")], F[(task, "single")]
            uL, uS = top_units(L, a.k), top_units(S_, a.k)
            union = sorted(set(uL) | set(uS))
            pos = {b_: i for i, b_ in enumerate(union)}

            cap = a.max_tokens or None
            td, sd = prep(mh, fd, a.n_items, torch)
            tu, su = prep(mh, fu, a.n_items, torch)
            sq_d = collect_prompt_tokens(mh, td, sd, union, D, a.batch_size,
                                         torch, cap)
            sq_u = collect_prompt_tokens(mh, tu, su, union, D, a.batch_size,
                                         torch, cap)
            # A prompt with no unmasked prompt token cannot contribute to any
            # shaping; drop it from BOTH polarities' counts explicitly rather
            # than letting a None reach the stack.
            sq_d = [q for q in sq_d if q is not None]
            sq_u = [q for q in sq_u if q is not None]
            nb = min(len(sq_d), len(sq_u))
            if nb == 0:
                print(f"  {task}: no usable prompt tokens, skipped")
                continue
            seqs = sq_d[:nb] + sq_u[:nb]
            lens = [len(q) for q in seqs]
            print(f"  {nb} prompts/polarity, prompt tokens per prompt "
                  f"min={min(lens)} med={int(np.median(lens))} max={max(lens)} "
                  f"total={sum(lens)}", flush=True)

            def cols(units):
                return np.concatenate([np.arange(pos[b_] * D, (pos[b_] + 1) * D)
                                       for b_ in units])
            cL, cS = cols(uL), cols(uS)

            mats, ctrl, store = {}, {}, {}
            rng = np.random.default_rng(0)
            for sh in shapings:
                XL, XS, note = build_arms_all(seqs, nb, sh, cL, cS)
                n_samples = len(XL)
                print(f"    {sh:<9} rows={n_samples:<6} d_L={XL.shape[1]:<7}"
                      f" d_S={XS.shape[1]:<7} ({note})", flush=True)
                # one SVD per arm, sliced for every n
                # Widen the decomposition when bases are being stored: U does
                # not depend on n, so one SVD at the widest rank serves every
                # later sweep.
                nmax = max(max(ns), a.save_bases)
                QL, QS = bases(XL, nmax), bases(XS, nmax)
                if a.save_bases:
                    store[f"{sh}_QL"] = QL[:, :a.save_bases].astype(np.float32)
                    store[f"{sh}_QS"] = QS[:, :a.save_bases].astype(np.float32)
                perm = rng.permutation(n_samples)
                for n in ns:
                    C, k = cosmat_from_bases(QL, QS, n)
                    meta = dict(n_samples=n_samples,
                                degenerate=degenerate(n, n_samples),
                                stats=panel_stats(C) if C is not None else None)
                    mats[(sh, n)] = (C, meta)
                    if C is None:
                        continue
                    st = meta["stats"]
                    flag = " DEGENERATE" if meta["degenerate"] else ""
                    print(f"      n={n:<3} k={k:<3} |diag|={st['diag_mean']:.3f}"
                          f" rowmax={st['rowmax_mean']:.3f}"
                          f" svcca5={st['svcca5']:.3f}{flag}", flush=True)
                    stat_rows.append(dict(
                        model=model, task=task, shaping=sh, n=n, k=k,
                        rows=n_samples, d_L=XL.shape[1], d_S=XS.shape[1],
                        arm="observed", degenerate=int(meta["degenerate"]),
                        **{q: round(st[q], 6) for q in
                           ("diag_mean", "diag_max", "offdiag_mean",
                            "offdiag_max", "rowmax_mean", "frob_frac",
                            "svcca5")},
                        rho=";".join(f"{v:.6f}" for v in st["rho"])))
                    if a.shuffle_control:
                        Cc, kc = cosmat_from_bases(QL, QS, n, perm)
                        mc = dict(n_samples=n_samples,
                                  degenerate=meta["degenerate"],
                                  stats=panel_stats(Cc))
                        ctrl[(sh, n)] = (Cc, mc)
                        stc = mc["stats"]
                        stat_rows.append(dict(
                            model=model, task=task, shaping=sh, n=n, k=kc,
                            rows=n_samples, d_L=XL.shape[1], d_S=XS.shape[1],
                            arm="shuffled", degenerate=int(mc["degenerate"]),
                            **{q: round(stc[q], 6) for q in
                               ("diag_mean", "diag_max", "offdiag_mean",
                                "offdiag_max", "rowmax_mean", "frob_frac",
                                "svcca5")},
                            rho=";".join(f"{v:.6f}" for v in stc["rho"])))
                del XL, XS, QL, QS

            tag = f"{model}_{task.replace(' ', '-')}_prompt"
            sub = (f"long vs single localisation · k={a.k} · {nb} "
                   f"prompts/polarity · every prompt token "
                   f"({sum(lens)} total) · rows differ by shaping")
            for absolute, suffix in ((False, "signed"), (True, "abs")):
                print("  wrote", draw_grid(
                    mats, f"Cosine similarity between left singular vectors "
                          f"— {model} / {task}", sub,
                    outdir / f"cosmat_{tag}_{suffix}.png", a.mode, absolute))
            print("  wrote", draw_spectrum(
                mats, f"Canonical correlation spectra — {model} / {task}",
                outdir / f"spectrum_{tag}.png", a.mode))
            if a.shuffle_control:
                for absolute, suffix in ((False, "signed"), (True, "abs")):
                    print("  wrote", draw_grid(
                        ctrl, f"SHUFFLE CONTROL — rows permuted — "
                              f"{model} / {task}",
                        "one side's rows permuted: whatever survives is what "
                        "dimension alone manufactures",
                        outdir / f"cosmat_{tag}_shuffled_{suffix}.png",
                        a.mode, absolute))
            np.savez_compressed(
                outdir / f"cosmat_{tag}.npz",
                **{f"{sh}_n{n}": C for (sh, n), (C, _) in mats.items()
                   if C is not None},
                **{f"shuf_{sh}_n{n}": C for (sh, n), (C, _) in ctrl.items()
                   if C is not None},
                **store)
            write_stats(outdir, stat_rows)
            del seqs, sq_d, sq_u, mats, ctrl

        del mh
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    out = write_stats(outdir, stat_rows)
    if out:
        print(f"\nwrote {out}  ({len(stat_rows)} rows)")


if __name__ == "__main__":
    main()
