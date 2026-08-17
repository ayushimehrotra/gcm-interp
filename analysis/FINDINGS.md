# Long-form vs single-token localization: findings

Comparison of **long-form (LF)** and **single-token (ST)** ATP localization across
5 models and 5 tasks. Every number below is arm-against-arm on the same grid.

**Grid: 22 of 25 cells.** Three cells have no localization in either repo and
cannot be recovered by analysis: Falcon3-10B bias and factual recall, and
OLMo-2-13B factual recall.

| | verse | summarization | bias | factual recall | persona |
|---|---|---|---|---|---|
| gemma-3-12b-it | ✓ | ✓ | ✓ | ✓ | ✓ |
| Qwen1.5-14B-Chat | ✓ | ✓ | ✓ | ✓ | ✓ |
| Qwen1.5-32B-Chat | ✓ | ✓ | ✓ | ✓ | ✓ |
| OLMo-2-1124-13B-DPO | ✓ | ✓ | ✓ | — | ✓ |
| Falcon3-10B-Instruct | ✓ | ✓ | — | — | ✓ |

Falcon persona was localized and generated all along; it was simply never scored,
and a grid panel that tests for *accuracy* reported it as "not localized". Scoring
it (2026-08-17) took the grid from 21 to 22 cells.

Two cell counts appear below and are not interchangeable. Structural measures
(placement, concentration, containment, similarity) run on all **22** localized
cells. Figures and the standardized effect summary run on **19**, excluding
Falcon, because they average across tasks and Falcon has only 3 of 5 — averaging
it over a different task set would make the means incomparable. Geometry is
**21**: SVCCA additionally needs stored activations, which Falcon persona lacks.

---

## The short version

The two localizations produce **statistically unrelated attribution fields**
(Spearman −0.012 [−0.029, +0.003], 6.4% unit overlap) that nonetheless **select
activations carrying the same latent factors** (SVCCA excess 0.656 [0.631, 0.676]
over the shuffle floor, 21/21 cells). They differ in *placement*, not in *what the
model represents*.

Long-form localizes **earlier and tighter**: 3.56 layers shallower (21/22,
p = 1.1e-05, d = −1.59) and more concentrated by layer (22/22, p = 4.8e-07).
These are the largest and most reliable effects in the study.

The **behavioural** advantage is real but narrow, and it **reverses with
evaluation format** (§1) — which is a finding about format matching, not a clean
win for either arm.

Against random controls (§8), localization beats **uniform** random decisively
(37–8, p < 0.0001) and **depth-matched** random only modestly (22–10, p = 0.050).
An earlier claim that depth-matched random *tied* localization was a bug in my
own tally, not a result (§10.1).

**The mechanism (§8.2.3).** Each arm's depth-matched control keeps its per-layer
counts and randomizes the units inside them, so `depth-LF vs depth-ST` is the
method contrast with head identity destroyed. It reproduces the real result:
LF vs ST is +0.62 grid steps (17–5, p = 0.017) and depth-LF vs depth-ST is +0.90
(15–3, p = 0.008), with a null residual. **Free-form's advantage is a property of
the depth profile it selects, not of the units it identifies** — which is the
same conclusion the placement statistics (§2) reach from the field alone.

---

## 0. Definitions and math

Everything below is defined once here and referenced by name afterwards.

### 0.1 The unit of analysis

Localization scores **residual-stream coordinate blocks**, not attention heads.
With hidden size `H` and `n_heads` attention heads,

```
D = H // n_heads                    block width
block (l, u) = coordinates [uD, (u+1)D)  of layer l's o_proj output
N = n_layers x n_blocks             total blocks,  n_blocks = H / D
```

`model_handler.py:49` sets `self.dim = hidden_size // self.num_heads`, and
`eval/generation.py:54` slices `o_proj.output[..., D*u : D*(u+1)]`. For gemma-3
this is doubly wrong as a "head": `D = 3840/16 = 240` while the true attention
`head_dim` is 256, so blocks do not even align to head boundaries.

### 0.2 ATP (attribution patching)

Per example, let `a_l ∈ R^{S x H}` be layer `l`'s `o_proj` output on the **base**
input, and `a_l^src` the **source** activations aligned to base at the assistant
marker (`data_handler.py:276`). Let `r` be the response-start index.

Summed response log-likelihood (`patching_utils.py:19`):

```
LL(x) = sum over t = r .. S-2  of  log P(x_{t+1} | x_{<=t})
```

The objective and the attribution (`patching.py:48-69`):

```
L          = LL(undesired) - LL(desired)
effect_l   = (dL / da_l)  o  (a_l^src - a_l)          elementwise, R^{S x H}
net_l      = sum over sequence positions s of effect_l[s, :]      -> R^H
```

Reduce coordinates to blocks and average over examples
(`eval/logits_handler.py:67`, `get_top_k_layer_and_head`):

```
A[l, u] = mean over examples of  sum over d in block u of  net_l[d]
```

`A ∈ R^{n_layers x n_blocks}` is **the attribution field**. Both arms are the
same computation; they differ only in which dataset plays source/base and hence
in how many response tokens `LL` sums over.

### 0.3 Selection

The pipeline keeps the largest **signed** values (`eval/logits_handler.py:92`
calls `flat.topk`), *not* the largest magnitudes:

```
S_k(A) = the first  floor(k*N)  entries of  argsort_desc( vec(A) )
```

About 48% of every field is negative, so ranking by `|A|` yields a different set —
they agree at only Jaccard 0.431. Every selection in this document uses the
signed rule.

### 0.4 Concentration statistics

On the **positive part** `a = max(A, 0)` flattened, since negative blocks are
never selected at any budget. Let `a_(1) >= a_(2) >= ...` be `a` sorted
descending and `T = sum(a)`.

```
capture(k)  = ( sum_{i <= floor(k*N)} a_(i) ) / T        share of attributed
                                                          effect a budget buys
PR          = ( sum a )^2 / sum a^2                       participation ratio,
PR / N        reported as a fraction of all blocks        low = concentrated
entropy     = -( sum_c p_c log p_c ) / log N,  p = a / T  normalised to [0,1]
gini        = ( 2 * sum_i i * a_(i)^asc ) / ( N * T ) - (N+1)/N
top1_share  = a_(1) / T
```

All are scale-free, so arms whose raw attribution magnitudes differ by an order
of magnitude remain comparable.

### 0.5 Set and rank agreement

```
Jaccard(A, B)   = |A ∩ B| / |A ∪ B|
Spearman(x, y)  = Pearson correlation of rank(x), rank(y), ties averaged
```

**Percentile-in.** With `r_B(b)` the 0-indexed rank of block `b` in arm B's
descending order:

```
pct(S_A in B) = mean over b in S_A of  ( 1 - r_B(b) / (N - 1) )
```

`0.5` is chance.

**AUC (Mann-Whitney).** Probability a randomly chosen selected block outranks a
randomly chosen unselected one under the other arm's scores. With `n1` selected,
`n2` unselected, and `R1` the rank-sum of the selected group:

```
AUC = ( R1 - n1*(n1-1)/2 ) / ( n1 * n2 )
```

`0.5` is chance, by construction — no empirical control arm is involved.

**Containment asymmetry**, the directional quantity in §5:

```
asymmetry = AUC(S_LF under ST's scores) - AUC(S_ST under LF's scores)
```

### 0.6 Layer profile statistics

Let `p(l)` be the share of an arm's selected blocks lying in layer `l`.

```
mean depth   = sum_l  l * p(l)
rel. depth   = mean depth / (n_layers - 1)
entropy      = -( sum_l p(l) log p(l) ) / log n_layers
layers used  = #{ l : p(l) > 0 }
top-3 mass   = sum of the three largest p(l)
```

**1-Wasserstein** between two layer profiles, on the shared depth axis:

```
W1(p, q) = sum_l | CDF_p(l) - CDF_q(l) | / (n_layers - 1)
```

decomposed as METHOD `W1(LF_task, ST_task)` vs TASK `W1(LF_A, LF_B)` and
`W1(ST_A, ST_B)`. The ratio is the §2 headline: a null of `LF_A` vs `ST_B` would
still contain the method contrast and so cannot isolate it.

### 0.7 SVCCA (the manifold test)

Activations are `o_proj.output` at ATP-matched response positions. A sample is
one (prompt, position) pair; the same samples index both arms, so the matrices
are **row-paired**:

```
X_L ∈ R^{n x d_L}   columns = LF's selected blocks' coordinates
X_S ∈ R^{n x d_S}   columns = ST's selected blocks' coordinates
```

Center each, take the top-`r` left singular vectors as an orthonormal basis, and
read canonical correlations off the cross-product:

```
X~ = X - mean(X)
Q_X = first r columns of U, where X~ = U diag(s) V^T
rho = singular values of ( Q_X^T Q_S ),   rho_i ∈ [0, 1]
```

`rho_i ≈ 1` means some direction in LF's coordinates is a near-deterministic
linear function of ST's — the same latent factor in a different basis.
**This is invariant to which coordinates each arm owns**, which is why it works
where subspace overlap measures cannot (§10.3).

**Permutation floor.** Canonical correlation inflates when `r` is large relative
to `n`. Destroying the row correspondence measures that inflation directly:

```
rho_perm = mean over permutations pi of  mean( top-5 rho( X_L, X_S[pi] ) )
excess   = mean(top-5 rho) - rho_perm
```

Only `excess` is evidence of shared information. (At n=128 the floor was 0.678;
at n=640, 0.313.)

**Concept-bearingness.** With `y` the desired/undesired label and `C_L` the
canonical variates `Q_L U`:

```
lab_L = max over i <= 5 of  | Pearson( C_L[:, i], y ) |
```

### 0.8 Concept direction and density

Per-coordinate **effect size**, over the full residual write:

```
sigma_pooled(c) = sqrt( ( var_des(c) + var_und(c) ) / 2 )
d_c             = ( mean_des(c) - mean_und(c) ) / sigma_pooled(c)
```

Standardizing per coordinate is what makes this a *concept* measure rather than a
*magnitude* measure — an unstandardized mean difference is dominated by a few
outlier dimensions and returned density ≈ 0.0004 for both arms.

```
energy(U)   = ( sum over (l,u) in U of || d_{l,u} ||^2 ) / || d ||^2
s(U)        = |U| / N                                    coordinate share
density(U)  = energy(U) / s(U)                           1.0 = its fair share
```

`density > 1` means the arm's coordinates are concept-richer than their size
warrants. `s` is an exact structural constant, not an empirical control.

Within-arm spread of concept energy, `e_b = ||d_b||^2` over the arm's own blocks:

```
PR_blocks = ( sum_b e_b )^2 / ( sum_b e_b^2 ) / |U|
```

### 0.9 Behaviour margin and ablation

Each long-form prompt carries two committed continuations. The margin is the
**per-token mean** log-likelihood difference:

```
Lmargin(p) = [ log P(desired | p) - log P(undesired | p) ] / n_tokens
```

Per-token, not summed: the summed form is length-confounded, with
corr(length difference, margin) = +0.535 at slope +1.01 nats/token — enough on
its own to flip summarization's baseline negative (§10.5).

Ablating a block set `U` zeroes those coordinates during the forward pass:

```
a_l[ uD : (u+1)D ] <- 0     for every (l, u) in U
damage(U)  = Lmargin_ablated - Lmargin_baseline      (negative = behaviour lost)
```

### 0.10 The behavioural threshold

With `acc_arm(k)` the judged accuracy at budget `k` on the long-form eval:

```
kstar(arm, B) = min { k in grid : acc_arm(k) >= B }        (None if never reached)
```

The grid is `{0.01, 0.03, 0.05, 0.07, 0.09, 0.1, 0.5, 1.0}`, so `kstar` is
reported as a **grid rank** (index into that list), not a raw fraction — the
spacing is not uniform. **Complete-case** keeps cells where both arms reach `B`;
**censored-worst** assigns rank 8 (one past the grid) to an arm that never does.

### 0.11 The truncated objective (§6)

The only modification made to the pipeline. `T` counts response tokens:

```
L_T = sum over t = r .. min(r+T, S-1) - 1  of  [ log P_und(x_{t+1}) - log P_des(x_{t+1}) ]
```

`T = None` delegates to the original function, so the untruncated path is the
repo's own code unchanged. Everything downstream — alignment, response-start
detection, padding, head reduction, averaging — is the pipeline's.

### 0.12 Statistics

All effects are **paired within cell** (a cell is one model x task), so model and
task difficulty cancel.

```
bootstrap CI : 10,000 resamples of the cell-level differences, 2.5 / 97.5 pct
sign test    : two-sided binomial on #(d > 0) vs #(d < 0), ties excluded
               p = 2 * sum_{i <= min(pos,neg)} C(n, i) / 2^n,  capped at 1
Pearson p    : t = |r| sqrt((n-2)/(1-r^2)), two-sided, normal approximation
monotonicity : a cell counts if the statistic is ordered across all of
               T in {1, 4, 16, all} -- not merely different at the endpoints
```

Reported as `effect [CI]  direction hits/n  p`.

### 0.13 The k=1.0 provenance gate

At `k = 1.0` every arm selects **every** block, so any two arms apply an
identical intervention and must return identical accuracy. The observed spread
there is therefore not an effect — it is a **measurement floor**, and no
arm-vs-arm difference smaller than it is interpretable.

```
floor(cell) = max_arms acc(arm, k=1.0) - min_arms acc(arm, k=1.0)
```

The floor is non-zero whenever two arms were generated under different numerical
conditions: decoding is greedy (`eval/generation.py:25-28`), so a float-level
difference flips a token at a near-tie and the continuation diverges. A cell
whose arms all sit at exactly 0 or 1 passes the gate **trivially** and is weak
evidence of comparability; those are reported separately (§8.1).

---

## 1. The behavioural claim: the advantage is narrow and format-dependent

**On free-form evaluation** (19 cells, judged, steering matched to eval):

| | k=0.03 | k=0.05 | k=0.07 | k=0.09 | k=0.1 | k=0.5 |
|---|---|---|---|---|---|---|
| LF − ST | resolved | resolved | resolved | resolved | **+0.128** [+0.016, +0.252] | +0.011 |

The paired difference is positive at every budget through `k = 0.1`, peaks at
**+0.128 [+0.016, +0.252]** at `k = 0.1`, and its bootstrap CI excludes zero at
`k ∈ {0.03, 0.05, 0.07, 0.09, 0.1}` — and nowhere else. In levels, both arms peak
near `k = 0.1` (LF 0.560, ST 0.432) and decline as steering degrades fluency.

**On single-token evaluation** the result reverses. The LF−ST difference peaks at
only **+0.031 [−0.073, +0.158]** at `k = 0.03` and is resolved in LF's favour at
**no budget at all**; it is resolved in *ST's* favour at `k ∈ {0.05, 0.09, 0.1}`.
Levels peak at `k = 0.07` with **ST ahead** (0.658 vs 0.609).

The honest statement is therefore **format matching**: each localization wins
under the evaluation format whose supervision it was built from. A claim that
"long-form localization is better" is only true on free-form evaluation, and the
paper should say so in those words.

---

## 2. Placement: LF localizes earlier and tighter — the strongest result

All 22 cells, LF − ST:

| statistic | effect [95% CI] | d | direction | p |
|---|---|---|---|---|
| mean depth (layers) | **−3.564** [−4.486, −2.650] | −1.59 | LF earlier 21/22 | 1.1e-05 |
| relative depth | −0.084 [−0.108, −0.061] | −1.47 | LF earlier 21/22 | 1.1e-05 |
| layer entropy | −0.073 [−0.091, −0.056] | −1.67 | LF tighter 22/22 | 4.8e-07 |
| top-3 layer mass | +0.091 [+0.070, +0.113] | +1.73 | LF tighter 22/22 | 4.8e-07 |
| layers used | −4.273 [−5.864, −2.773] | −1.13 | LF fewer 18/22 | 4.0e-04 |

Two statistics are unanimous across all 22 cells. These are the effects to lead
with: they are large, consistent, and structural — they do not depend on the
judge, the steering sweep, or the accuracy metric.

---

## 3. Concentration: mostly a null

Within-field concentration on the positive part, LF − ST, 22 cells:

| statistic | effect [95% CI] | d | direction | p |
|---|---|---|---|---|
| capture at k=0.01 | +0.0131 [+0.0031, +0.0247] | +0.49 | LF higher 16/22 | 0.052 |
| PR / N | −0.0152 [−0.0333, +0.0019] | −0.35 | LF lower 14/22 | 0.286 |
| gini | +0.0018 [−0.0119, +0.0168] | +0.05 | — 10/22 | 0.832 |
| entropy | −0.0021 [−0.0110, +0.0061] | −0.10 | — 11/22 | 1.000 |

Only the small-budget capture term is even marginal, and it does not survive as a
claim. **LF's field is not globally sharper.** What differs is *where* the mass
sits (§2), not how concentrated it is — an important distinction, because the
intuitive story "LF needs fewer heads because its field is peakier" is not what
the data shows.

This section reversed direction once the signed-vs-magnitude selection bug was
fixed (§10.2).

---

## 4. Same concept manifold — the central result

21 cells with stored activations, rank r = 24, n = 640 samples:

```
mean top-5 canonical correlation   0.969 [0.944, 0.989]
top-1 canonical correlation        0.989 [0.979, 0.996]
row-shuffle permutation floor      0.313 [0.311, 0.315]
excess over floor                  0.656 [0.631, 0.676]      21/21 cells
```

At a unit Jaccard of **0.064** — 6.4% overlap, near chance — the activations the
two arms select are **near-linearly-equivalent**. Different coordinates, same
latent factors.

This is the paper's central mechanistic claim, and it is the reason the two arms
can steer comparably while agreeing on almost no units.

### Concept density

```
LF density   1.386        ST density   1.228
difference   +0.158 [+0.080, +0.235]     18/21 cells      p = 0.0015
```

Both arms select coordinates carrying **more** concept energy than their
coordinate share warrants (density > 1), and LF's are denser. This is a genuine
difference in *what gets selected*, not merely where.

---

## 5. Containment: LF's field subsumes ST's, not the reverse

22 cells:

```
AUC(S_ST under LF's scores)   0.535        ST's picks rank well under LF
AUC(S_LF under ST's scores)   0.486        LF's picks rank at chance under ST
asymmetry                     +0.049 [+0.023, +0.074]   16/22   p = 0.052
```

LF's ranking partially *contains* ST's: blocks ST selects are above chance in LF's
ordering, while blocks LF selects are at chance (0.486 ≈ 0.5) in ST's. The
asymmetry is real but modest, and at p = 0.052 it is borderline — report it as
directional evidence, not a resolved effect.

---

## 6. Why: supervision count

ATP's objective integrates over ~50 response tokens for LF and 1 for ST (§0.11),
so supervision count is the candidate causal variable. Truncating LF's objective
to `T` response tokens (§0.11) and watching the field move tests it directly.

**19 validated cells**, 5 models, 5 tasks. Two cells that the previous run
included are now excluded by the validation gate, which requires the
reconstructed `T=all` field to match the pipeline's own on-disk field at
Jaccard ≥ 0.5: gemma bias (0.490) and Qwen1.5-14B persona (0.429). Falcon persona
fails setup (`KeyError: 'factor'`). Surviving cells validate at 0.55–0.95,
Spearman +0.82 to +0.99.

| statistic, T=all vs T=1 | effect [95% CI] | direction | p |
|---|---|---|---|
| rank correlation with LF's field | **+0.811** [+0.756, +0.865] | T=all closer to LF **19/19** | <1e-4 |
| mean layer | +4.403 [+2.570, +6.288] | T=1 later 17/19 | 0.0007 |
| capture at k=0.01 | +0.050 [+0.028, +0.075] | T=all more concentrated 16/19 | 0.0044 |

Truncating supervision walks the field away from LF and **deeper**, in almost
every cell. The dose-response is ordered across `T ∈ {1, 4, 16, all}` in 18/19
cells for rank correlation and 16/19 for set overlap — but only 3–4/19 for
capture and mean layer, so the *ranking* moves monotonically with supervision
while the depth and concentration statistics do not.

### But ST is not "LF with one token"

The truncation **overshoots**. Regressing the depth shift it induces against the
real LF↔ST depth gap, over the 19 cells:

```
mean real gap    +2.57 layers
mean shift       +4.40 layers        171% of the gap
Pearson r=+0.474   Spearman rho=+0.263   slope +0.70
```

The shift points the right way and is larger where the arms genuinely differ more
in depth (positive slope), but it is **1.7x too big** on average. One response
token is not what makes ST what it is; supervision count explains the *direction*
of the difference, not its magnitude.

---

## 7. Ablation: LF's units are more load-bearing

Standardized effect **+0.30 [+0.17, +0.42]** over n = 95 (cell × budget)
observations: zeroing LF's selected blocks costs more per-token behaviour margin
than zeroing ST's. Smaller than the placement effects but reliably signed.
(Pre-pull value was +0.33 — the rerun moved it slightly and changed nothing.)

Note on file naming: no current script writes `dose_*.csv`. The analysis lives in
`necessity_dose.py`, which writes `necessity_<margin>.csv`; the `dose_*.csv`
files are orphaned output from an earlier naming (as are `atpmatch_*.csv`, which
no script produces either). `figures_analysis.py` globbed the orphaned pattern,
so this panel kept showing pre-pull numbers however often the analysis was rerun
— repointed 2026-08-17. Two lessons worth keeping: rerunning an analysis does not
refresh a figure whose consumer reads a stale filename, and `necessity_dose.py`
defaults to a 3-cell pilot (`--models gemma-3-12b-it`), so the grid must be
passed explicitly.

---

## 8. Random control arms

Two controls per cell (CLAUDE.md §1): **uniform** random blocks, and
**depth-matched** random, which reproduces the real arm's per-layer count
histogram exactly and so isolates whether head-level choice earns anything beyond
depth. Seed 0 only.

### 8.1 Provenance gate first (§0.13)

| data | comparable rows | floor at k=1.0 | note |
|---|---|---|---|
| persona, single-token eval | 9 | **0.000** | all 9 saturated at 0/1 — passes trivially |
| persona, free-form eval | 8 | **0.080** | 4 of 8 saturated |
| verse + summarization | 40 | median 0.000, mean 0.014, max 0.120 | 2/40 above 0.10, both Falcon |

Free-form persona therefore has a **0.080 resolution floor**: differences smaller
than that are not interpretable there. Single-token persona has clean shared
provenance (both arms generated on this machine) but saturates, so its clean gate
is weaker evidence than it appears.

### 8.2 Does localization beat random?

min-k to 80% of the cell's **common** ceiling (the best any arm reaches there —
scoring each arm against its own ceiling flatters a flat, low control, which
clears 80% of its own low asymptote at the first budget). Cells split by whether
the real arm's accuracy moves at all across the sweep (range ≥ 0.15 = "live"):

| comparison | live cells | all cells |
|---|---|---|
| vs **uniform** random | **37–8**, 5 tied, p < 0.0001 | 41–9, p < 0.0001 |
| vs **depth-matched** random | **22–10**, 21 tied, p = 0.050 | 24–13, p = 0.099 |

**Localization is clearly better than uniform random** — this is robust across
every variant tried (task subset, gating, pipeline).

### 8.2.1 Decomposition: how much is layers, how much is heads?

The depth-matched arm keeps the real per-layer histogram and randomizes *which
heads within those layers*. So the three comparisons separate the two
contributions directly (live cells):

| comparison | what it isolates | result | p |
|---|---|---|---|
| depth-matched vs uniform | the **layer** contribution alone | 25–8, 11 tied | **0.0046** |
| localization vs uniform | the **total** effect | 37–8, 5 tied | **<0.0001** |
| localization vs depth-matched | the **head** increment on top of layers | 22–10, 21 tied | 0.050 |

**Layer choice carries most of the effect.** Simply reproducing the real layer
histogram — with heads chosen at random inside it — recovers a large and
decisively significant share of localization's advantage over uniform random.

**A residual head-level contribution is present but not established.** It is
positive in point estimate everywhere and borderline pooled (p = 0.050), but the
pooled test combines the two localizations, which are measured on the same cells
and are not independent. Split by arm, neither reaches significance:

```
free-form    vs depth-matched-to-free-form     13 finer,  8 tied,  6 coarser   p = 0.167   (n=27)
single-token vs depth-matched-to-single-token   9 finer, 13 tied,  4 coarser   p = 0.267   (n=26)
```

The honest reading is therefore **"mostly a layer picker, but probably not only
one"** — the direction is consistent (localization finer in 13 of 19 decided
free-form comparisons) while the sample and the coarse grid leave it unresolved.
§8.3 is a live counter-example where the head increment is decisive.

### 8.2.2 Do LF and ST differ in *how much* heads matter?

No. Pairing each arm's head increment within the same cell (increment = how many
grid steps finer the real arm is than its own depth-matched control), 24 live
cells carrying both arms:

```
free-form    head increment   mean +0.54 grid steps, median 0
single-token head increment   mean +1.08 grid steps, median 0
paired difference (LF - ST)   mean -0.54 [-2.04, +0.71]
                              LF larger in 9, ST larger in 8, equal in 7   p = 1.00
```

The CI spans zero and the sign test is exactly null; the difference in means is
carried by a few outliers, both medians being 0. **Neither localization is more
of a layer picker than the other.**

This sharpens what the LF/ST contrast actually is. The two arms differ strongly
in *which* layers they select (§2: 3.56 layers shallower, unanimous on layer
entropy and top-3 mass) and hardly at all in *which units within a layer* (unit
Jaccard 0.064 vs layer Jaccard 0.505). They do not differ in how much the
within-layer choice is worth. Both methods are predominantly **layer-level**
phenomena that point at different depths — which is also why their two
depth-matched controls are not interchangeable and must be plotted separately.

### 8.2.3 The LF advantage over ST is reproduced by the layer profiles alone

Each localization has its own depth-matched control, so `depth-LF vs depth-ST` is
the LF/ST contrast **with head identity randomized away** — same per-layer
counts, random heads inside them. If that comparison reproduces the real one, the
method difference is a depth effect and nothing else. 29 live cells carrying all
four arms, min-k in grid steps (positive = LF finer):

| contrast | mean | LF finer | tied | ST finer | p |
|---|---|---|---|---|---|
| **LF vs ST** (real localizations) | +0.62 | 17 | 7 | 5 | **0.0169** |
| **depth-LF vs depth-ST** (layer profile only) | +0.90 | 15 | 11 | 3 | **0.0075** |
| residual (real − layer-profile) | −0.28 | 13 | 7 | 9 | 0.524 |

**Discarding head identity entirely preserves the free-form advantage.** Keeping
only "how many units per layer" and drawing the units at random reproduces
LF > ST at the same significance — if anything slightly larger — and the residual
is a clean null.

This is the mechanistic answer to the paper's motivating question. Free-form
localization reaches the behaviour at a smaller budget **because of the depth
profile it selects, not because it identifies better units**. It composes with
§8.2.1 and §8.2.2 without contradiction: within-layer choice does buy a little
(borderline), but it buys the *same* little for both arms, so it cancels from the
LF−ST difference and leaves depth carrying all of it.

### 8.3 The one cell with real dynamic range

Falcon persona, free-form eval, matched steering — the cleanest separation in the
dataset, and the only cell where the metric is neither saturated nor at floor:

```
k        LF atp   ST atp   uniform   depth-matched
0.05      0.120    0.000     0.000        0.000
0.09      0.260    0.140     0.000        0.020
0.10      0.280    0.120     0.000        0.000
1.00      0.000    0.000     0.000        0.000     <- gate passes exactly
```

Both localizations beat both controls decisively, depth-matched **fails
completely**, and LF beats ST 2:1. Where there is range to measure, localization
matters and head choice matters. This is a single cell and should be reported as
such, but it is the cell with the most power.

---

## 9. Established nulls

- **Shared subspace does not predict accuracy.** 21 cells: concept-bearingness vs
  peak accuracy r = +0.238, p = 0.300; within-cell (which arm has the stronger
  manifold vs which wins) r = +0.006, sign agreement 8/16. "Same manifold" is a
  statement about representation, **not** an explanation of steering success.
- **Concentration does not predict accuracy** (§3).
- **Linear decodability of the concept** — standardized effect −0.33
  [−0.65, +0.13], n = 19, includes zero. (Pre-pull −0.27; still null.)
- **Shared subspace carries the concept** — +0.04 [−0.35, +0.58], n = 19,
  includes zero.

Two content measures sit at zero while every placement measure is large. The
arms differ in *where they look*, not in *what is there to find*.

---

## 10. Methodological corrections (things that were wrong)

### 10.1 NaN-as-tie inflated the control comparison

Tallying min-k comparisons from a pandas DataFrame, unreached cells became `NaN`.
`a is None` is `False` for `NaN`, so every such comparison fell through to the
"tied" branch: ties were inflated from 12 to 29 on verse+summarization and the
sign test returned **p = 1.00**, supporting a headline that depth-matched random
*ties* localization and therefore that "head-level selection earns nothing".

The dict-based implementation handles unreached cells explicitly and gives
**22–10, p = 0.050**. The curves themselves were byte-identical between the two
pipelines — the data never disagreed, only my tally. The "layer choice carries
the localization" reading was an artefact and is withdrawn.

### 10.2 Selection ranks signed values, not magnitudes

The pipeline uses `flat.topk` on signed attributions (§0.3), while my first
analyses ranked by `|A|`. The two agree at only Jaccard 0.431. Fixing it inverted
two headlines: concentration weakened to a null (§3) and concept density
strengthened (§4).

### 10.3 Eigenvector cosine and eigenvalue spectra are structurally uninformative

Cosine similarity between arms' eigenvectors is arithmetically ~0 whenever the
supports are disjoint, regardless of shared structure; eigenvalue spectra match at
0.998 even for unrelated signals. Both were demonstrated on synthetic data and
replaced with SVCCA (§0.7), which is invariant to which coordinates each arm owns.

### 10.4 Accuracy roots must be priority-ordered, never merged

`manifold_vs_accuracy.py` held a private copy of the accuracy harvest that never
received the `results_pipeline_with_answers` root, silently dropping 6 of 21 cells
— every persona cell and two bias cells. On the truncated data the headline
correlation read r = +0.462, p = 0.084; on the full grid it is r = +0.238,
p = 0.300. The marginal effect was an artefact of missing cells.

The same class of bug appeared again in `dynamic_range_gate.py`, which took a max
*across* roots. Since atp appears in several roots and the control arms only in
ayushi's, merging handed the real arm a systematic advantage unrelated to
localization. Both now take the highest-priority root that has the cell.

### 10.5 Behaviour margin must be per-token

The summed form is length-confounded (corr +0.535, slope +1.01 nats/token),
enough on its own to flip summarization's baseline negative.

### 10.6 Control arms must use max over steering factors

Control arms were aggregated with `mean` while real arms used `max`, which broke
convergence at k=1.0 (0.514 vs the true 0.800). All arms now use max over N.
Separately, both depth-matched controls were collapsed under one key though they
differ by up to 0.62; they are now keyed per arm.

### 10.7 Judge pipeline bugs (2026-08-17)

Four bugs made a 20-cell persona judge run report `rc=0` on every cell while
producing nothing:

1. `extraversion` was missing from `SOURCE_TO_TEMPLATE`, so every free-form cell
   died with `unknown SOURCE`. The template itself already existed.
2. `--no_judge_prefill` was not passed. The default `"("` prefill shifts ratings
   down one step and `w_rf` counts only rating == 5, so accuracies would have
   collapsed toward zero — a *silent* failure, unlike (1).
3. Single-token cells resolved their test set from the localization name rather
   than the eval mode, so cross combinations looked for a file that cannot exist.
4. `compute_single_accuracies` scores an unconditional yes-rate where umang's
   pipeline scores a flip rate conditioned on the unsteered answer not being
   "yes". **Numerically inert here** — the unsteered baseline never answers "yes"
   in single-token eval (measured 0.000 across all 5 models and both trees) — so
   no number changed. Recorded because it would bite on any task whose baseline is
   not at floor.

`run_judge.py` exits 0 when it prepares nothing, so exit codes cannot be trusted
as evidence of work; check output counts.

---

## 11. Data-integrity issues in the source repos

- **CLAUDE.md §2's blocking confound is largely resolved.** It documents a +0.346
  gap at k=1.0 favouring the random arm on gemma verse. Measured now across all 40
  verse/summarization cells under max-over-N: median 0.000, mean 0.014, max 0.120,
  with only 2/40 above 0.10 (both Falcon, excluded from figures). gemma verse is
  now 0.000. The per-condition divergence §2 identified is real; the summary
  statistic the paper reports is insensitive to it, and fuller judging closed the
  gap. **§2 should be updated.**
- **Metric tags differ by repo and eval mode**, not by repo alone: umang writes
  `comb` for free-form and `w_rf` for single-token. Keying on repo alone makes
  every single-token real arm invisible.
- **`comb` and `w_rf` are the same quantity**: `judge >= 5 AND flu >= 2 AND
  rel >= 2` versus `judge == 5 AND flu == 2 AND rel == 2`, on 1–5 and 0–2 scales.
- **Only draw seed 0 exists.** CLAUDE.md §4 wants an across-seed spread before
  calling any arm "within noise"; every control number here is a point estimate.

---

## 12. Open questions

1. **Seeds.** One draw seed makes §8 a point estimate. Seeds 1–2 would let the
   "within noise" question actually be answered.
2. **Budget grid resolution.** 21 of 53 live comparisons in §8.2 are ties, most of
   them the same grid point. A finer grid between 0.01 and 0.1 would sharpen the
   depth-matched comparison, which is where the remaining ambiguity sits.
3. **Does §8.3 generalize?** The one cell with real dynamic range shows
   depth-matched random failing completely. Whether that holds wherever range
   exists is the highest-value next experiment.
4. **Format matching (§1).** The reversal by eval format deserves its own
   treatment rather than a caveat.
5. **Supervision count** pending the §6 rerun.

---

## Scripts

| script | produces |
|---|---|
| `figures.py` | budget figures: gap, levels, per-cell grid, both eval modes |
| `figures_analysis.py` | depth profile, shared information, dose, containment, effect summary |
| `similarity_numbers.py` | all overlap / similarity numbers |
| `layerwise_mechanics.py` | `layerwise_mechanics.csv` (§2) |
| `localization_concentration.py` | `localization_concentration.csv` (§3) |
| `activation_geometry.py` | `activation_geometry.csv` (§4) — SVCCA + density |
| `containment_asymmetry.py` | `containment_asymmetry.csv` (§5) |
| `manifold_vs_accuracy.py` | `manifold_vs_accuracy.csv` (§9) |
| `persona_provenance_assay.py` | persona k=1.0 gate + min-k (§8.1) |
| `dynamic_range_gate.py` | control-arm comparison split by dynamic range (§8.2) |
| `position_count_mechanism.py` | `position_count*.csv` (§6) — drives the repo's own `Patching` |
| `necessity_dose.py` | `dose_*.csv` (§6) |
| `probe_units.py` | `probe_*.csv` (§9) |
| `contrastive_covariance.py` | `contrastive_covariance.csv` |
