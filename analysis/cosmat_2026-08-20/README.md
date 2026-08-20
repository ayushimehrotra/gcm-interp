# Cosine-similarity matrices between the arms' left singular vectors

Qwen1.5-14B-Chat, all 5 tasks, k=0.05, **100 prompts/polarity, every prompt
token**. Written by `svcca_cosmat.py`; compared by `cosmat_compare.py`.
Figures in `figs_prompt_alltokens/`.

## The procedure

1. Collect **every prompt token** of every prompt — no response tokens, and no
   cap on tokens per prompt. Prompts are 27–66 tokens, so the collection is
   ragged (a list of `[T_i, H]`), not a rectangular `[B, S, H]`. Nothing is
   dropped to make a reshape work.
2. Build the activation matrix, three ways:

   | shaping | matrix | rows | what one row is |
   |---|---|---|---|
   | `bs_h` | `(ΣT_i, H)` | 5 956 – 10 842 | one (prompt, token) |
   | `s_h` | `(S, H)` | 54 – 94 | one position, averaged over prompts within polarity |
   | `b_last_h` | `(B, H)` | 200 | `T[:, -1, :]` — the **last prompt token**, one per prompt |

3. Centre it (`svcca_sweep._centre`).
4. SVD each localization's matrix (`svcca_sweep._basis`).
5. `M = U_L[:, :n].T @ U_S[:, :n]` — cosines in shared **sample** space, for
   n ∈ {10, 20, 50}.

`s_h` has to rectangularise, and prompts differ in length, so it aligns them
from the **right**: position −1 is the last prompt token in every prompt,
truncated to the shortest. That is what makes the three coherent — `s_h`'s last
position is the batch-mean of exactly what `b_last_h` takes per prompt.

Verified by decoding the selected indices: for a verse prompt with `start=42`,
the prompt window is 3..41 (system + user turn), the response window 42..168.
Disjoint, and left-padding is respected via the attention mask.

## 1. `b_last_h` is the shaping that answers the question, and it is positive everywhere

The last prompt token is the pre-generation state — purely an input, never a
consequence of the output — and the moment
`eval/activations.py:steering_reps_cache` builds steering vectors from. At 200
rows it clears degeneracy at every n up to 50.

```
task              n=10                 n=20                 n=50
                  rho5  floor  exc     rho5  floor  exc     rho5  floor  exc
verse            0.983  0.300 +0.683  0.995  0.511 +0.484  1.000  0.822 +0.178
summarization    0.914  0.314 +0.600  0.970  0.500 +0.470  1.000  0.824 +0.176
persona          0.996  0.273 +0.723  0.999  0.462 +0.537  1.000  0.816 +0.184
bias             0.990  0.286 +0.705  0.998  0.503 +0.494  1.000  0.831 +0.169
factual recall   0.988  0.313 +0.675  0.994  0.503 +0.491  1.000  0.815 +0.185
```

Positive in all 15 points, and remarkably tight across tasks (+0.60 to +0.72 at
n=10). This is the causal claim's own shaping, on input-only activations, and it
holds.

## 2. Read `excess`, not `rho5` — at n=50 rho5 is 1.000 everywhere

Every shaping, every task, reaches `rho5 = 1.000` at n=50. Taken at face value
that says the two localizations select *identical* subspaces. The floor says
otherwise: 0.82 for `b_last_h`, and exactly 1.000 for `s_h`. The whole n=50
signal is the floor moving, not the observation.

This is the same trap CLAUDE.md 8.10 flags for `rho1`, one level up: as n grows
the statistic saturates and stops discriminating, while the null it must be read
against grows faster. The excess column is the only one that is evidence.

## 3. `s_h` collapses — 54–94 rows cannot support n=50

`2n ≥ rows−1` makes the leading cosines 1 on any data. `s_h` is inside that
region at n=50 for every task:

```
verse, s_h        n=10   rho5 1.000   floor 0.511   excess +0.489
                  n=20   rho5 1.000   floor 0.817   excess +0.183
                  n=50   rho5 1.000   floor 1.000   excess +0.000   DEGENERATE
```

The shuffled control scores **exactly 1.000** while its diagonal is 0.18 — bright
from dimension counting, not agreement. Those panels carry a red border.
**Do not quote `s_h` at n=50.** Averaging over prompts throws away the row count
that every alignment statistic needs, and no choice of n fixes that.

`bs_h` has the largest excess (+0.84 to +0.95) but its ~8 000 rows are ~40
near-duplicate tokens from each of 200 prompts, so nominal n badly overstates
the evidence. `b_last_h`'s 200 rows are genuinely independent.

## 4. The entrywise picture is task-dependent — the scalar hides it

CLAUDE.md 8.10 records that for gemma verse the matrix "looks like noise even
when the subspaces coincide" — diagonal 0.248 against singular values 0.986, and
concludes the correspondence is a rotation. **That is not general.** On
`b_last_h` at n=10, where rho5 is 0.91–1.00 for all five tasks:

```
task              |diag|   row-max    rho5
persona            0.757    0.808    0.996
verse              0.659    0.707    0.983
bias               0.614    0.698    0.990
summarization      0.352    0.536    0.914
factual recall     0.262    0.600    0.988
```

|diag| spans **0.26 to 0.76** at essentially constant rho5. Persona, verse and
bias are genuinely index-aligned — component i of one arm maps to component i of
the other, a clean diagonal you can read straight off the heatmap. Factual
recall is much closer to a pure rotation: same subspace, different axes. SVCCA
scores both the same.

So "the arms are near-linearly-equivalent" is doing less work than it sounds:
for three of five tasks the arms agree on the *axes*, which is a much stronger
statement, and for factual recall they agree only on the *span*, which is
weaker. The scalar cannot tell you which you have.

|diag| also falls monotonically with n (persona `b_last_h`: 0.76 → 0.57 → 0.43),
which is the expected signature — the leading components align pairwise, the
tail is noise that only matches as a subspace.

## 5. Reading the figures

Each figure is a 3×3 grid: rows = shaping, cols = n ∈ {10, 20, 50}.

| file | what |
|---|---|
| `cosmat_<model>_<task>_prompt_abs.png` | \|cos\|, sequential ramp. **Read this for magnitude** |
| `..._signed.png` | signed cos, diverging. Read for block structure |
| `..._shuffled_abs.png` / `_signed.png` | same grid, one side's rows permuted |
| `spectrum_<model>_<task>_prompt.png` | singular values of every M = the canonical correlations |
| `cosmat_<model>_<task>_prompt.npz` | the raw M matrices |
| `cosmat_stats.csv` | \|diag\|, row-max, off-diagonal, rho5, full spectrum, per panel |

Singular vectors are sign-arbitrary, so a negative cell is the same angle as its
positive twin — the signed panel shows structure, the absolute panel magnitude.
Always read a panel against its shuffled twin.

## 6. Reproducing

```bash
python svcca_cosmat.py --models Qwen1.5-14B-Chat \
  --tasks "verse,summarization,persona,bias,factual recall" \
  --ns 10,20,50 --n_items 100 --shuffle_control --outdir figs_prompt_alltokens
python svcca_cosmat.py --replot --outdir figs_prompt_alltokens   # redraw, no GPU
```

~9 minutes for all 5 cells. Two implementation notes that matter if this is
extended:

- **One SVD per arm per shaping**, sliced for every n. `U` does not depend on n,
  so computing `_basis` per n would repeat a 10 842 × 10 240 decomposition three
  times.
- **The shuffle control needs no second SVD.** Permuting a matrix's rows permutes
  its left-singular basis exactly (centring is permutation-invariant, and `P·U`
  is still orthonormal), so the control is `QS[perm]`. Checked against a fresh
  SVD of the permuted matrix: principal angles agree to 1e-15.
- CPU `numpy.linalg.svd` at 9 559 × 10 240 takes 39 s; the GPU `torch` version is
  3.1× faster but float32 costs ~1e-3 relative accuracy in the singular values,
  so CPU is used.

## Earlier runs, superseded

`cosmat_figs/` (response tokens) and `figs_prompt_12pos/` (12 prompt positions)
used a fixed S=12 window and defined the third shaping as `b_sh = (B, S*H)`.
They are kept because they carry the one comparison this run cannot make —
prompt vs response at matched settings:

```
mean excess over non-degenerate points     prompt      response     diff
  bs_h                                     +0.837      +0.836      +0.001
  s_h                                      +0.166      +0.157      +0.009
  b_sh   (B, S*H)                          +0.468      +0.417      +0.051
```

The alignment does not depend on the response — slightly **stronger** on
input-only activations. That closes the causal-direction caveat in CLAUDE.md 8.9
independently of the `b_last_h` result above.

## Environment note

CLAUDE.md 8.11 describes a `.venv/` with the pinned `requirements.txt` and all
five models cached. **Neither exists on this pod.** What was built here:
`.venv-svcca/` (`python -m venv --system-site-packages`), then nnsight 0.4.11,
transformers 4.57.6, accelerate, bitsandbytes 0.48.2, and — not obvious —
`pillow>=10`, `numpy>=1.26`, and `jinja2>=3.1`, since the system versions of all
three are too old for transformers' import chain and `apply_chat_template`.

**gemma-3-12b-it could not be run**: the repo metadata is public but the weights
are gated, and no `HF_TOKEN` is set here. Every gemma number quoted above is
from CLAUDE.md, not re-measured. Qwen1.5-14B-Chat is public and was used
throughout.

---

## What is in this directory

Graphs only. The `.npz` matrices, per-panel `cosmat_stats.csv`, and run logs are
deliberately not committed — re-running the scripts below regenerates them.

| directory | what |
|---|---|
| `figs_prompt_alltokens/` | **main set.** Every prompt token, n = 10 / 20 / 50 |
| `figs_prompt_alltokens_n10-20/` | same, n = 10 / 20 only (nothing degenerate) |
| `figs_prompt_sweep/` | `ladder_*` = heatmap with n as columns up to full rank; `sweep_*` = mean/mean-squared/min singular value vs n; `fulln_*` = matrix at full rank |
| `figs_prompt_12pos/` | earlier run, 12 sampled prompt positions |
| `figs_response_12pos/` | earlier run, 12 sampled response positions |

## Scripts (repo root)

| file | what it does |
|---|---|
| `svcca_sweep.py` | pre-existing. **Modified**: added `--extract prompt`, which takes n_pos positions from the input prompt instead of the response, keeping S > 1 so the shapings stay distinct |
| `svcca_cosmat.py` | collects activations, builds the three shapings, SVDs each arm, writes the cosine-matrix heatmaps. `--save_bases N` caches the singular vectors so any n can be re-plotted without a GPU; `--replot` redraws from cache |
| `cosmat_sweep_n.py` | reads those cached singular vectors and draws the n-ladder heatmaps and the vs-n curves. No GPU |
| `cosmat_compare.py` | side-by-side table of two runs (e.g. prompt vs response) |

## Rebuilding from scratch

```bash
python svcca_cosmat.py --models Qwen1.5-14B-Chat \
  --tasks "verse,summarization,persona,bias,factual recall" \
  --ns 10,20,50 --n_items 100 --save_bases 400 --shuffle_control \
  --outdir analysis/cosmat_2026-08-20/figs_prompt_alltokens
python cosmat_sweep_n.py --indir  analysis/cosmat_2026-08-20/figs_prompt_alltokens \
                         --outdir analysis/cosmat_2026-08-20/figs_prompt_sweep --ladder
```

~9 minutes on one GPU for the five cells; the sweep step needs no GPU.
