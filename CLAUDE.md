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

## 2. OPEN CONFOUND — the atp and random trees are not numerically comparable

**Do not report any random-vs-atp difference until this is resolved.**

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

**Why this is disqualifying.** A +0.35 artefact biased toward the random arm
swamps the effects the paper reports (+0.007 [−0.049, +0.065]) and pushes in
exactly the direction that would manufacture the section 6.3 headline ("random
matches the real localization → the paper's framing changes"). Any such finding
could be this artefact rather than a result.

**How to resolve it.** Regenerate one atp cell on the current machine and diff
it against the committed atp tree:

- If it reproduces byte-for-byte, that machine matches the atp provenance and
  the random arms can be regenerated there for a valid comparison.
- If it does not, **both** arms must be regenerated on one machine before any
  comparison is made. Only the `w_rf`/`wo_rf` numbers computed from a single
  provenance are usable.

Use `topk=1.0` as a permanent assay: after any regeneration, the two arms must
agree there. A non-zero gap at k=1.0 means the trees are still incomparable.

Scope of what has been measured: the baseline divergence is confirmed for all
four models above. The +0.346 accuracy gap is gemma verse at k=1.0 only, because
the other cells are not judged yet.

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

### Current state (generation)

| model | verse + summarization, both arms, seed 0 |
|---|---|
| Falcon3-10B-Instruct | complete (72/cell — includes N=15,20) |
| Qwen1.5-14B-Chat | complete (56/cell) |
| gemma-3-12b-it | complete (56/cell) |
| OLMo-2-1124-13B-DPO | complete except `verse-long / randomlayer-s0 / verse-single_eval` at 49/56 |
| Qwen1.5-32B-Chat | **16 of 448** — `random-s0`, verse-long only, N=1,2 only |

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

### State (scoring)

- Every `*-single_eval` condition is scored (token matching, no GPU).
- Of 1,936 long-eval files, **56 are scored**: gemma-3-12b-it,
  `from_verse-long_to_prose`, `random-s0`, `verse-long_eval`.
- Accuracy JSONs land under
  `judge-evals/accuracy/{model}/from_X_to_Y/random-s*/…` — confirm they never
  merge into the `atp` tree.

## 6. Reporting back

For each arm produce, per (model, task, eval, k): the accuracy **mean and spread
across seeds** — the control is a distribution, not a point estimate. Then:

1. **k-curves**: real long-form, real single-token, uniform random, layer-matched
   random, on one plot per (model, task, eval), with the random arms as shaded
   bands.
2. **min-k to 80% of ceiling** for all four arms — the paper's precision metric.
3. Flag immediately if **either** random arm is within noise of the real
   localizations at k ≤ 0.1. That is the headline result and changes the paper —
   which is exactly why section 2 must be resolved first.

Do not compute or report cross-eval-mode differences: long-form evals are judged
and single-token evals are token-matched, so those numbers are not comparable.
All comparisons must be **within** an eval mode.

## 7. Known traps

- **Provenance**: see section 2. Check the `topk=1.0` agreement before trusting
  any arm-vs-arm number.
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
