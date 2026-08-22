# TASK: random-head control experiments

You are running the **random-head baseline** for the localization paper. This is
the blocking experiment — every other result in the paper is uninterpretable
without it. Read this whole file before running anything.

**Read section 2 first.** There is an open confound that makes the current
arm-vs-arm numbers untrustworthy. Resolving it outranks collecting more data.

## 1. Why this experiment exists

The paper compares two ways of localizing attention heads for steering:
**long-form** localization (`<task>-long`) and **single-token** localization
(`<task>-single`). Established so far:

- The two methods select head sets that are **~94% disjoint** (mean Jaccard 0.063
  vs 0.030 expected by chance, at k ≤ 0.1).
- They nevertheless steer **equally well** (+0.007 [−0.049, +0.065], n = 34).
- Long-form reaches 80% of its own ceiling with **fewer heads** (−0.099
  [−0.192, −0.019]).

Those results have two incompatible explanations, and the current data cannot
tell them apart:

- **(A)** Localization finds real causal structure that is redundant, so many
  distinct valid head sets exist.
- **(B)** Head choice is irrelevant — steering *any* k% of heads with a task
  vector works, and top-k is just a dosage knob. (Note `topk` already explains
  η² = 0.179 of the variance, the largest single factor, which is exactly the
  signature (B) predicts.)

This experiment discriminates them. **Two control arms are needed, not one:**

1. **Uniform random** — heads sampled uniformly from all (layer, head) pairs.
   Answers: *does localization do anything at all?*
2. **Layer-matched random** — random heads sampled so the **per-layer count
   histogram exactly matches** the real localized set. Answers: *does
   localization do anything beyond picking layers?*

Arm 2 matters because the two localizations agree far more on **layers**
(Jaccard 0.469) than on **heads** (0.058). If layer-matched random matches the
real localization, then head-level selection is not earning its cost — a strong
result in its own right.

## 2. RESOLVED (2026-08-17) — the confound survives per-condition but not in the
## summary statistic

**Status: no longer blocking.** Keep reading — the underlying divergence is real
and the gate below is still mandatory before any arm-vs-arm claim. What changed
is that the statistic the paper actually reports turns out to be insensitive to
it.

Measured across **all 40** verse/summarization cells under the max-over-N
summary (not the single cell this section was originally written from):

```
gap at k=1.0    median 0.000    mean 0.014    max 0.120
cells above 0.10:  2 / 40   (both Falcon3-10B paragraph, excluded from figures)
```

gemma verse — the cell whose +0.346 is tabulated below — now measures **0.000**.
Two things closed the gap: the per-N table below compares individual steering
factors, whereas every reported number takes the **max over N**, and far more
long-eval cells have since been judged (the original table was computed when 56
of 1,936 files were scored).

Per-task floors, for reference when reading any arm-vs-arm number:

| data | comparable rows | floor at k=1.0 | note |
|---|---|---|---|
| persona, single-token eval | 9 | 0.000 | all saturated at 0/1 — passes trivially |
| persona, free-form eval | 8 | 0.080 | 4 of 8 saturated |
| verse + summarization | 40 | 0.000 median | 2 cells at 0.120 |

**Any arm-vs-arm difference smaller than the relevant floor is not
interpretable.** Free-form persona in particular has a 0.080 floor, which is
larger than most effects measured there. `analysis/persona_provenance_assay.py`
computes this gate; run it after any regeneration.

Note the asymmetry that explains the floors: single-token persona gates at 0.000
because *both* arms were generated on this machine. Free-form persona crosses
provenance (umang's atp vs local controls) and does not.

The original analysis follows, unchanged, because the mechanism it identifies is
still real at the per-condition level.

---

`topk=1.0` is a built-in null: at k=1.0 both arms select **every** head, so the
two arms apply an identical intervention and must produce identical output.
Verified on gemma-3-12b-it verse-long: the two head-set CSVs are set-equal
across all 768 heads (48 layers × 16), and `generate_with_patches`
(`eval/generation.py:20,51`) reads only `layer` and `neuron` — the atp CSV's
`value` column is never used, so the head set *is* the whole intervention.

They do not produce identical output:

| N | atp `w_rf` | random-s0 `w_rf` | delta |
|---|---|---|---|
| 1 | 0.420 | 0.780 | +0.360 |
| 2 | 0.200 | 0.800 | +0.600 |
| 4 | 0.280 | 0.760 | +0.480 |
| 5 | 0.220 | 0.760 | +0.540 |
| 6 | 0.060 | 0.380 | +0.320 |
| 8 | 0.060 | 0.120 | +0.060 |
| 10 | 0.020 | 0.080 | +0.060 |

**Mean +0.346 at an identical intervention, always favouring the random arm.**

The divergence is not caused by steering. The *unsteered* baseline (`old_<base>`
in the gen JSON) also differs between the trees, and generation is greedy
(`do_sample=False`, `top_p`/`top_k`/`temperature` all `None` —
`eval/generation.py:25-28`), so this is not sampling noise. Two checks localize
it:

- **Within** a tree the baseline is perfectly stable: 0 differences across all
  56 condition files. Generation is deterministic on a given machine.
- **Between** the atp and random-s0 trees the baselines diverge for every model
  and both task families, while the queries align 50/50:

  | model | verse | summarization |
  |---|---|---|
  | Falcon3-10B-Instruct | 33/50 | 31/50 |
  | Qwen1.5-14B-Chat | 13/50 | 32/50 |
  | OLMo-2-1124-13B-DPO | 10/50 | 35/50 |
  | gemma-3-12b-it | 21/50 | 33/50 |

Every generation flag recorded in `config.yml` is identical across the trees
(`batch_size 1`, `kv_caching true`, `max_new_tokens 256`, `seed 42`,
`steering_type last_token`). So the two trees were generated under different
**numerical** conditions — different machine, GPU, or library versions. Under
greedy decoding a tiny float difference flips a token at a near-tie and the
whole continuation diverges.

**Why this looked disqualifying.** A +0.35 artefact biased toward the random arm
would swamp the effects the paper reports and push in exactly the direction that
manufactures the section 6.3 headline. That reasoning was right; what it was
missing is that the tabulated gaps are per-N, and no reported quantity is.

**Why it is no longer blocking.** Under max-over-N the same comparison is
median 0.000 across 40 cells (see the top of this section). The artefact
reshuffles *which* steering factor wins without moving the max. Regenerating
both arms on one machine would still be the cleanest fix and is worth doing if
free-form persona matters, since that is where the 0.080 floor sits.

Use `topk=1.0` as a permanent assay: after any regeneration, the arms must agree
there. **A saturated row (every arm exactly 0 or 1) agrees trivially and is weak
evidence** — count those separately, as the assay script does.

Scope of what has been measured: the baseline divergence is confirmed for all
four models above at the per-condition level. The k=1.0 summary gap is now
measured on all 40 verse/summarization cells and all 17 persona rows.

## 3. What already exists in this repo — DO NOT rebuild it

Both control arms are implemented and work end to end. The arm and the draw seed
are encoded in `--patch_algo` (`random-s0`, `randomlayer-s2`, …), and
`set_output_prefix()` interpolates it into the results path, so every arm and
seed gets its own tree with no filename collisions.

- `eval/logits_handler.py:24,28,32` — `is_random`, `is_layer_matched`,
  `draw_seed`. The draw seed is deliberately independent of `--seed`, which
  feeds `set_seed()` and also controls generation.
- `eval/logits_handler.py` — `retrieve_random_k` (uniform) and
  `retrieve_layer_matched_k` (layer-matched; asserts the drawn per-layer
  histogram reproduces the reference atp histogram, and fails loudly if the
  reference CSV is missing rather than falling back to uniform).
- `eval/eval_runner.py:102,108,110` — arm dispatch in `save_top_k()`.
- `random_control_mirror_uniform.py` — the uniform draw ignores `--source`, so
  for a given eval mode the `-long` and `-single` localizations are
  bit-identical. The `-long` arm is generated and mirrored into `-single`
  rather than paying for it twice. Layer-matched is **never** mirrored: its draw
  follows that localization's real per-layer histogram.
- `judge-evals/config.py` — `GEN_RE` already accepts `random|targeted`, so the
  judge pipeline needs no change.
- `random_control_analysis.py` — reporting (section 6). Already collapses each
  model over its own steering-factor sweep via `MODEL_SFS`.
- `run_random_control_seed0.sh` — driver. `MODELS=… SEED=… bash …`.

## 4. What to run

Steering vector is always **matched to the eval mode** — long-steer with
long-eval, single-steer with single-eval. Do not cross them; that axis is
deliberately fixed in this paper.

Sweeps must match the existing atp runs exactly so the arms are comparable:

- `topk`: `0.01,0.03,0.05,0.07,0.09,0.1,0.5,1.0`
- `steering_factors` (N): `1,2,4,5,6,8,10` — except Falcon3-10B-Instruct, whose
  atp sweep also covers `15,20`. Collapsing two arms over different N sets would
  let the wider sweep win on max alone; `MODEL_SFS` in
  `random_control_analysis.py` handles this.
- 50 test items per condition
- Models: `tiiuae/Falcon3-10B-Instruct`, `Qwen/Qwen1.5-14B-Chat`,
  `google/gemma-3-12b-it`, `allenai/OLMo-2-1124-13B-DPO`, `Qwen/Qwen1.5-32B-Chat`

There is **no `--patch_model` step** for random arms — no attribution is
computed, so only the eval step runs.

### Current state (generation) — updated 2026-08-17

| model | verse + summarization, both arms, seed 0 | extraversion (persona) |
|---|---|---|
| Falcon3-10B-Instruct | complete (72/cell — includes N=15,20) | complete (448/448) |
| Qwen1.5-14B-Chat | complete (56/cell) | complete (448/448) |
| gemma-3-12b-it | complete (56/cell) | complete (448/448) |
| OLMo-2-1124-13B-DPO | complete (56/cell) | complete (448/448) |
| Qwen1.5-32B-Chat | complete (56/cell) | complete (448/448) |

**Persona is fully generated and scored**: 2240 generations across 5 models x 2
localizations x 2 arms x 2 eval modes, then 2240 accuracy files. Driven by
`scripts/persona_dispatch.sh` (two concurrent model-jobs, 5-min stagger,
Qwen1.5-32B solo) and `scripts/persona_random_one.sh`.

Depth-matched random needs the ATP reference from the **same** localization tree
(`eval/logits_handler.py:112`). For persona that lives in the umang checkout and
is symlinked in as `results/<model>/from_extraversion-*/atp`. Without the
symlink the arm fails silently — 3 minutes, 0 generations, exit 0.

**Only draw seed 0 exists.** `results/_dropped_partial_seed1/` holds a killed
Falcon seed-1 run. This makes the section 6.3 test uncomputable — "is the random
arm within noise of the real localization" needs an across-seed spread, and one
draw has none. `random_control_analysis.py` reports those conditions as "not yet
checkable" rather than as a clean bill of health. Breadth across models at seed 0
was chosen over depth; seeds 1–2 remain to be run.

Resume is filename-based, so re-running is safe and finished conditions are
skipped. **If you change sampling logic mid-run, delete the affected outputs** or
stale draws are silently reused.

## 5. Scoring

Score with the **same judge pipeline and the same metric as the atp runs**, or
the comparison is void. The metric is chosen by eval mode:

| eval mode | metric | how |
|---|---|---|
| **long-form** (`*-long_eval`) | judge **+ fluency + relevance** | `judge-evals/run_judge.py` → `w_rf` |
| **single-token** (`*-single_eval`) | token/letter matching | `judge-evals/compute_single_accuracies.py` |

Judge model `unsloth/Meta-Llama-3.1-70B-Instruct-bnb-4bit`, same prompts.

Two settings are **not** free choices — every committed atp long-eval script
uses them, and changing either invalidates the comparison:

- `--no_judge_prefill`. The default `"("` prefill shifts ratings down one step
  (5 → 4). `run_judge.py` detects cached prompts built under the other setting
  and rebuilds them automatically.
- `--batch_size 16` (`BATCH_SIZE=16` in every committed judge script).

Run it with `judge-evals/scripts/random_control_judge.sh`. That script chunks one
invocation per (model, task, localization) rather than one `--all` pass, because
`_evaluate_all_workdirs_batched()` buffers an entire mode across every workdir in
memory and writes the JSONL only after that mode finishes — a single crash in a
monolithic run discards hours of inference. Resume granularity is therefore
**(chunk, mode)**.

Measured on an H100 (gemma verse-long, 56 files): ~14 prompts/s, ~2.5 min of
model-load per chunk. The full 1,936-file long-eval grid is roughly 6.5 hours.

### State (scoring) — updated 2026-08-17

- Every `*-single_eval` condition is scored (token matching, no GPU).
- Long-eval scoring is far more complete than when this section was written:
  verse/summarization control arms are judged for all five models, and all 10
  persona free-form control cells (1120 accuracy files) plus both Falcon persona
  atp trees are done.
- Accuracy JSONs land under
  `judge-evals/accuracy/{model}/from_X_to_Y/random-s*/…` — confirm they never
  merge into the `atp` tree.

**`run_judge.py` exits 0 when it prepares nothing.** A 20-cell pass once reported
`rc=0` on every cell while producing zero accuracy files. Never treat the exit
code as evidence of work; check `Phase 1 done: N files prepared, 0 errors` and
count the output files. Four bugs caused it, all now fixed — see section 7.

## 6. Reporting back

For each arm produce, per (model, task, eval, k): the accuracy **mean and spread
across seeds** — the control is a distribution, not a point estimate. Then:

1. **k-curves**: real long-form, real single-token, uniform random, layer-matched
   random, on one plot per (model, task, eval), with the random arms as shaded
   bands.
2. **min-k to 80% of ceiling** for all four arms — the paper's precision metric.
3. Flag immediately if **either** random arm is within noise of the real
   localizations at k ≤ 0.1. That is the headline result and changes the paper —
   which is exactly why the section 2 gate must be checked first.

### Findings so far (2026-08-17, seed 0 only)

Full detail in `analysis/FINDINGS.md`. Headline, on min-k to 80% of each cell's
**common** ceiling, split by whether the real arm's accuracy moves at all across
the sweep ("live" cells):

- **vs uniform random**: localization wins 37–8, p < 0.0001. Decisive.
- **vs depth-matched random**: 22–10, p = 0.050. Modest, and non-significant
  when split by localization (free-form alone: 13–6, p = 0.167).
- **Layers carry most of it**: depth-matched beats uniform 25–8, p = 0.0046.
- **The LF-over-ST advantage is entirely a depth effect.** `depth-LF vs depth-ST`
  — the method contrast with head identity randomized away — reproduces the real
  result (15–3, p = 0.008 vs 17–5, p = 0.017) with a null residual.

An earlier claim that depth-matched random *tied* localization was a tally bug
(NaN counted as a tie), not a result. Do not repeat it.

Two things would resolve what remains: **seeds 1–2**, and a **finer budget grid
between 0.01 and 0.1** — 21 of 53 live comparisons are ties at the same grid
point, so the metric is resolution-limited.

Do not compute or report cross-eval-mode differences: long-form evals are judged
and single-token evals are token-matched, so those numbers are not comparable.
All comparisons must be **within** an eval mode.

## 7. Known traps

- **Provenance**: see section 2. Check the `topk=1.0` agreement before trusting
  any arm-vs-arm number. Saturated rows pass trivially.
- **The judge did not know about `extraversion`.** `SOURCE_TO_TEMPLATE` in
  `judge-evals/config.py` lacked the mapping although `PROMPT_TEMPLATES` had the
  template, so every free-form persona cell died with `unknown SOURCE` while the
  process still exited 0. Fixed; the task is also in `PAIRED_TEMPLATES`, since
  umang builds it as Response (1) = steered, (2) = unsteered.
- **`--no_judge_prefill` is mandatory and fails silently without.** The default
  `"("` prefill shifts ratings down one step and `w_rf` counts only rating == 5,
  so omitting it produces plausible near-zero accuracies rather than an error.
- **Cross combinations resolve the test set from the EVAL mode, not the
  localization.** A long-form localization scored under single-token eval looked
  for `extraversion-single/introversion-long-test.jsonl`, which cannot exist.
  Fixed by `resolve_test_base()` in `compute_single_accuracies.py`.
- **`comb` and `w_rf` are the same metric, and the tag depends on eval mode.**
  umang writes `comb` for free-form and `w_rf` for single-token. Keying the tag
  on repo alone makes every single-token real arm invisible.
- **Rerunning an analysis does not refresh a figure whose consumer reads a stale
  filename.** `dose_*.csv` and `atpmatch_*.csv` are orphaned outputs that no
  current script writes — `necessity_dose.py` writes `necessity_<margin>.csv` and
  `probe_units.py` writes one combined `probe_units.csv`. Both consumers in
  `figures_analysis.py` globbed the old patterns and so kept showing pre-pull
  numbers. Check what a script actually writes before trusting a rerun.
- **Script defaults are not the grid.** `necessity_dose.py` defaults to
  `--models gemma-3-12b-it --tasks verse,summarization,bias` — a 3-cell pilot
  that looks like a successful full run. Pass models and tasks explicitly.
- **Accuracy roots are priority-ordered, never merged.** Taking a max across
  roots inflates whichever arm appears in more of them; atp lives in several and
  the control arms only in `judge-evals/accuracy`. This bug has appeared twice.
- **Path prefix**: several committed scripts hardcode `/workspace/gcm-interp`,
  which has been wrong on every machine so far. Verify before launching.
  `judge-evals/scripts/sycophancy_single_eval_judge.sh` still has it.
- **`ninja` is missing from `requirements.txt`.** vLLM shells out to it when it
  JIT-compiles during CUDA graph capture; without it the engine dies at startup
  with `FileNotFoundError: 'ninja'`. Installing it is not enough — the venv's
  `bin` must be on `PATH`, so invoking `.venv/bin/python` directly still fails.
  `source .venv/bin/activate`, or export `PATH` as
  `judge-evals/scripts/random_control_judge.sh` does.
- **Seeds must not collide with `--seed`**: the head-draw seed comes from
  `--patch_algo`, never from `--seed`, which also controls generation.
- **Do not overwrite the `atp/` tree.** Everything here writes to `random-s*/`
  and `randomlayer-s*/`.
- **`num_attention_heads` is the query-head count** for GQA models (Falcon3
  12/layer, Gemma 16/layer, Qwen32B 40/layer). That is the correct dimension —
  the atp files are indexed by query head. Do not substitute
  `num_key_value_heads`.
- **Killing the judge mid-mode** can truncate the JSONL being written. On resume
  `out_path.exists()` is true, so a short file is skipped and silently yields a
  wrong accuracy. After any interruption:
  ```bash
  find judge-evals/workdirs -path '*random*' -name '*_ratings.jsonl' \
    -exec sh -c 'n=$(wc -l < "$1"); [ "$n" -ne 50 ] && echo "SHORT $n $1"' _ {} \;
  ```
  Delete anything it prints before resuming.
- **`phi-4` has been dropped from the paper.** Do not run it. Its results under
  `results/phi-4/` and `judge-evals/accuracy/phi-4/` are retained but unused.

## 8. Activation-geometry sweep (2026-08-20) — `svcca_sweep.py` rewritten

Separate task from the random-head control above. Recorded here because the
script was rewritten and several results change what `analysis/FINDINGS.md`
says.

### 8.1 What ran, and what the files are

| file | what | state |
|---|---|---|
| `svcca_sweep.csv` | 1320 points, 22 cells, `n_items=32` | **complete** |
| `svcca_sweep_n50.csv` | same grid at `n_items=50` | **PARTIAL — 16/22 cells** (pod stopped) |
| `analysis/geometry_2026-08-20/cosine_matrix.png` | the real 24x24 `Q_L^T Q_S`, shuffled control, spectrum | gemma verse |
| `analysis/geometry_2026-08-20/svcca_sweep_pre_rewrite.py` | the script as it was | also at git `515e8ae9b` |

Both CSVs were written by the **pre-rewrite** script. The rewrite (below) has a
different schema — do not concatenate them.

### 8.2 The robustness question is settled: the manifold result is not an (r, n) artefact

- **22/22 cells** positive at the published setting (r=24, n=640).
- **22/22 cells** have excess *rising* monotonically with n. The docstring's
  "the one that would actually threaten the paper" failure mode is absent; the
  opposite happens, because `rho_top5` is flat in n while the floor falls.
- **0 of 1122** non-degenerate grid points have excess <= 0.
- Excess does **not** rise monotonically with r either — it peaks at **r=8**
  (0.787) and decays to 0.220 by r=128, so r=24 is conservative, not lucky.
- Reproduces the published numbers on different hardware: 21 comparable cells,
  mean diff **+0.0013**, sd **0.0070**, no directional bias (Falcon3/verse
  +0.0158 and Falcon3/summarization -0.0099 are the extremes). SVCCA is immune
  to the section-2 provenance problem because it is one teacher-forced forward
  pass — no sampling, no autoregressive cascade.

### 8.3 The geometry grid is 22 cells, not 21

`FINDINGS.md` says geometry runs on 21 "cells with stored activations, which
Falcon persona lacks". `svcca_sweep.py` collects activations itself, so the cell
runs: **Falcon3-10B-Instruct / persona, excess +0.6758** (and +0.6794 on the
n=50 rerun — two independent measurements agreeing to 0.004). Among the
strongest cells in the grid.

**`analysis/activation_geometry.csv` is stale by one row**, and the "21/21
cells" headline should read **22/22**. That also aligns geometry with the
22-cell count the structural measures already use.

### 8.4 Prompt coverage was NOT inflating the result

The n-axis of the sweep varies *token positions* drawn from a fixed 32 prompts
per polarity — it never adds a prompt, so it could not rule out a prompt-set
artefact. Rerun at `n_items=50` (100 prompts, files hold 100):

```
16 of 22 cells   mean shift -0.0008   sd 0.0042   range -0.0133 .. +0.0037
```

A quarter of the per-cell measurement noise. **The caveat is closed for those 16
cells**; the remaining 6 (OLMo persona/bias, all five Qwen1.5-32B minus verse)
were not reached. Resume with
`python svcca_sweep.py --n_items 50 --out svcca_sweep_n50.csv` — it is NOT
resume-safe, it recomputes from scratch.

### 8.5 The degeneracy rule — use it, it is exact here

Two r-dim subspaces of an (n-1)-dim space are **forced** to intersect once
`2r >= n-1`, pinning the leading cosines at 1 on any data.

```
198 grid points satisfy 2r >= n-1   -> ALL 132 excess<=0 points are inside
1122 points do not                  -> ZERO have excess<=0, min is +0.0336
```

This is now enforced in the rewritten script. The old script's only guard was
for literal rank deficiency, which does not catch it.

### 8.6 Structure of the subspaces themselves (gemma verse, measured)

```
participation ratio  PR ~ 28   of 9120 coordinates   (PR/d = 0.003)
dims for 50% of variance: 18        for 90%: 279
variance on the desired-vs-undesired axis:  LF 6.93%   ST 3.69%
max |corr(PC_i, label)| over top 5:         LF 0.594   ST 0.397
```

Three consequences:

1. **PR ~ 28 is independent justification for r ~ 24.** The published rank sits
   at the measured effective dimensionality. It also explains why excess peaks
   at r=8 and decays past ~30.
2. **The concept is a minority component.** >90% of the variance in these
   subspaces is not the verse/prose contrast. "The arms carry the same latent
   factors" is true, but the factors are mostly not the concept.
3. **LF is concept-denser than ST** on both measures — independently
   reproducing FINDINGS 4's density result (1.386 vs 1.228) from a different
   calculation.

Worse for the strong reading: ranked by alignment, the **most-shared** canonical
direction (rho 0.9941, 11.1% of variance) has label correlation **0.042**. The
concept lives in canonical directions 2-3. The arms agree most strongly about
something unrelated to the behaviour being steered.

### 8.7 The shared structure is in the activations, NOT the weights

Weight-space test on the o_proj **row** slices the localization actually scores,
rank-truncated, floored against random blocks in the same layer:

```
weights      (gemma verse, r=24)   excess +0.065 +/- 0.060   ~1 sigma, NULL
activations  (same cell)           excess +0.673             decisive
```

The arms' read-out maps are about as related as random blocks. They converge on
what they *carry*, not how they are *wired* — which is redundancy hypothesis (A)
supported from the weight side. Note the test was biased *toward* agreement: it
can only run on the 11 layers both arms occupy, discarding the placement axis on
which they most differ.

### 8.8 The result does not depend on truncating at all

```
linear CKA (NO rank parameter)     observed 0.7699   floor 0.0367   excess +0.7333
SVCCA top-5 at r=24                observed 0.9858   floor 0.2850   excess +0.7008
PWCCA r=8..128                     excess +0.746 -> +0.507   (SVCCA: +0.83 -> 0.00)
```

CKA needs no rank choice and agrees. Prefer it when the choice of r cannot be
justified; prefer PWCCA when a rank is needed but stability matters.

### 8.9 The rewritten script

Same CLI shape, new capabilities. **Different output schema** from the old one.

- `--shapings bs_h,s_h,b_sh` — the [B, S, H] tensor flattened three ways:
  `(B*S, H)` one row per (prompt, token), `(S, H)` averaged over prompts within
  polarity, `(B, S*H)` one row per prompt. These change **n** by an order of
  magnitude, which is what sets every floor: 768 / 24 / 64 at the old defaults.
- `--extract last_prompt` — one activation per prompt at the token before the
  response starts. The pre-generation state, and the same moment
  `eval/activations.py:steering_reps_cache` builds the steering vectors from.
  This is the shaping to use for "how do input activations differ, causing
  different outputs", because response-position activations are partly a
  *consequence* of the output. Sets S=1, so all three shapings coincide.
- `--measures cka,svcca,pwcca` — `cka` takes no rank at all.
- `--struct_out` — per-arm PR, dims50/90, label correlation, class-variance
  fraction (section 8.6 as a first-class output).
- `--dump_spectra` — the full canonical-correlation spectrum per point.
- Degeneracy guard enforced; rank threshold made relative (`s > 1e-8*s[0]`)
  instead of scale-dependent absolute; prompts that cannot supply `n_pos`
  positions are **dropped, not padded**, so the [B, S, H] reshape is exact.

### 8.10 Traps specific to this analysis

- **Untruncated CCA is identically 1.** Measured on the real data: at r=767 both
  real and shuffled give exactly 1.000000. 9120 columns in a 768-dim sample
  space span all of it, so both arms' column spaces are the whole space.
  Truncation is not denoising — it is what makes the question exist.
- **`rho1` saturates and cannot discriminate.** 18 of 22 cells exceed 0.99;
  range 0.073 vs `rho_top5`'s 0.181. It also has the *highest* floor (0.358 vs
  0.312) because a max over a large search space is the most inflated statistic.
- **The spectrum carries information the scalar destroys.** `rho1 - rho_top5` is
  0.001-0.017 in 19 cells but **0.10-0.11** in gemma summarization/bias/factual
  recall — the three anomalous cells. They do not merely share *less*, they
  share **fewer dimensions**. Plot the spectrum for at least one cell per model.
- **The heatmap of `M` looks like noise even when the subspaces coincide.** For
  gemma verse the diagonal mean is 0.248 and greedy row-max 0.477, while the
  singular values are 0.986. The correspondence is a rotation mixing all 24
  directions, invisible entrywise. Do not read the matrix as the answer — but do
  plot it, because it shows *why* the scalar is necessary.
- **Weight-space analyses on the selected heads degenerate above
  `hidden/head_dim` heads.** Stacking column slices of `o_proj` for m heads
  gives m*head_dim vectors in a hidden-dim space; past m = hidden/head_dim
  (15 for gemma, 40 for Qwen/OLMo, 12 for Falcon3) the span is the ENTIRE
  residual stream and any two head sets agree at 1.0000 by dimension counting.
  At k=0.05 gemma selects 38 blocks, 31 disjoint — twice the threshold. Work
  per-layer, or cut the budget, and always add a random-head floor.
- **The localization indexes o_proj's OUTPUT axis, not attention heads.** Column
  slices (`o_proj.weight[:, h*head_dim:...]`) are a different object on a
  different axis; for gemma they are not even the same width (240 vs 256). See
  FINDINGS 0.1. An OV-circuit analysis answers a different question than the one
  the localization poses.
- **`--out` defaults to `svcca_sweep.csv` and will overwrite a completed run.**
  Always pass an explicit `--out` for reruns.

### 8.11 Environment (this pod)

`.venv/` at the repo root, python3.10, the full pinned `requirements.txt`
installs cleanly on aarch64 (GH200) — torch 2.11.0+cu130, transformers 5.14.1,
nnsight 0.4.11, bitsandbytes 0.48.2. All five models cached under
`~/.cache/huggingface` (181 GB). Sweep runtime ~3.5 h for 22 cells; the cost is
CPU-side SVDs, not the GPU, so `svcca()` recomputing the basis per rank is the
thing to optimise if it ever matters.

## 9. Intervention site and response span (2026-08-22) — two new flags

Both default to the existing behaviour, and both give any non-default variant its
own results tree, so nothing here changes a published number. Full write-up and
data: `analysis/o_proj_input_site/README.md`.

### 9.1 `--patch_site {o_proj_out,o_proj_in}`

`o_proj.output` is `W_O @ concat(z_h)`, so every coordinate is a sum over ALL
heads: a block there is residual-stream coordinates, not a head (FINDINGS 0.1).
`o_proj.input` is `concat(z_h)`, where block `u` IS head `u`, `head_dim` wide,
GQA included. `eval/patch_site.py`; threaded through attribution, ACP, the
steering cache, generation (both kv branches), `eval_extant`'s hook (which
becomes a forward **pre**-hook), and the random arms. `model_handler.dim` carries
the site-dependent width — 240 vs 256 on gemma.

Measured, verse, long-steer/long-eval, max over N — the input site reaches its
ceiling at a far smaller budget:

| k | Qwen in/out | gemma in/out |
|---|---|---|
| 0.01 | 0.92 / 0.04 | 0.86 / 0.12 |
| 0.03 | 0.96 / 0.62 | 0.86 / 0.28 |

Single-token localization is flat at both sites. Cross-site head overlap is at
chance (it must be — a unit index names a different object at each site); the
**layer** profiles agree strongly (Jaccard 0.48–0.77, ρ 0.61–0.87).

### 9.2 `--response_span {legacy,full}` — the `-single` metric is near-vacuous

`get_response_logits` starts one position late and **never scores the first
response token**. `-long` loses 1 of ~127. `-single` has a one-letter assistant
turn, so the answer is the dropped token and the summed span is just
`<|im_end|>`/`<end_of_turn>` + `\n`. ATP is differentiating *how readily the model
closes the turn after each letter* — the letter is conditioned on, never scored.

Verified on the shipped code: 2 summed terms vs 126; gradient mass exactly 0 at
the answer-predicting position; LL bit-identical under ±100 on the answer logit
while the `<|im_end|>` logit moves it; holds padded/unpadded and on both models.

`full` fixes it and gets its own tree (`-respfix`). Default stays `legacy`
because resume is filename-based and a mid-stream flip would mix two metrics in
one tree. Switching replaces most of the `-single` head set (Jaccard vs legacy
0.10–0.17 at o_proj_out, **0.00–0.09** at o_proj_in) and barely moves `-long`
(0.78–0.89). **Both structural findings survive**: LF vs ST stays near-chance
disjoint, and LF still localizes earlier than ST (the gap widens).

### 9.3 New traps

- **The judge validates METHOD against a whitelist.** `merge_outputs.py` rejected
  `atp-o_proj_in` on every file and `run_judge.py` still exited 0 having produced
  nothing. Fixed via `split_site_suffix` in `judge-evals/config.py`, which strips
  variant suffixes before validating but keeps METHOD as the full directory name
  so each variant keeps its own accuracy tree. Any new suffix must be added to
  `VARIANT_DIR_SUFFIXES`.
- **Generation is NOT deterministic across processes on this machine.** Two runs
  from the same script, same commit, 90 min apart, `no_deterministic: false`,
  differ on **25/50** unsteered baselines — identical queries, ~973 shared
  characters, then a token flips. Same rate and signature as section 2's
  cross-tree table, so **section 2's "different machine/GPU/library" diagnosis is
  probably wrong**; it is run-to-run nondeterminism. `determinism.py` leaves the
  SDP backend unpinned and cannot cover bitsandbytes NF4 matmuls.
- **Dose is not comparable across sites.** The steering vector is normalized in
  the space it is applied to. At k=1.0 the output site wins on gemma (0.80 vs
  0.36) purely from this. Compare k-curves within a site.
- **`nohup` does not survive a tool timeout** — it blocks SIGHUP, not SIGTERM.
  Long runs need `setsid`. A judge killed mid-mode truncates its JSONL; run the
  section 7 short-file check before resuming.
