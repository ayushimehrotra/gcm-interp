# TASK: random-head control experiments

You are running the **random-head baseline** for the localization paper. This is
the blocking experiment — every other result in the paper is uninterpretable
without it. Read this whole file before running anything.

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

## 2. What already exists in this repo — DO NOT rebuild it

`--patch_algo random` is already implemented and works end to end.

- `eval/logits_handler.py:74` — `retrieve_random_k(num_layers, num_heads, k, seed=42)`
  samples `int(k * num_layers * num_heads)` head pairs uniformly.
- `eval/eval_runner.py:91` — `save_top_k()` calls it when `reps_type == 'random'`.
- `eval/eval_runner.py:111` — skips loading ATP logits when `patch_algo == 'random'`.
- `eval/eval_runner.py:120,130` — sets `reps_types = ['random']`,
  `logit_metric = 'random'`.
- `config.py:124` — `set_output_prefix()` puts `patch_algo` in the path, so
  random runs land in a **separate tree** from `atp/`.
- `judge-evals/config.py` — the `GEN_RE` filename regex already accepts
  `(?P<REPS>random|targeted)`, so the judge pipeline needs no change.

**Two gaps to close.** These are the only code changes required.

### Gap 1 — the random draw is a single fixed sample

`seed=42` is a hardcoded default and is never overridden, so every run produces
the *same* random head set. A single draw is not a baseline; we need a
distribution over draws.

**Do NOT reuse the existing `--seed` flag for this.** `--seed` feeds
`set_seed(config.args.seed)` in `run_eval`, which also controls generation.
Changing it would vary the generations as well as the head draw and confound the
comparison. The head-draw seed must be independent.

### Gap 2 — layer-matched random does not exist

Nothing in the repo implements it.

## 3. Required code changes

Keep the diff minimal and keep every output in its own directory tree.

### 3.1 Encode the arm and seed in `patch_algo`

Use `patch_algo` values of the form:

- `random-s0`, `random-s1`, ... — uniform random, draw seed 0, 1, ...
- `randomlayer-s0`, `randomlayer-s1`, ... — layer-matched random

This is deliberate: `set_output_prefix()` already interpolates `patch_algo` into
the results path, so each arm and seed automatically gets its own tree
(`results/{model}/from_X_to_Y/random-s0/...`) with **zero filename collisions**
between seeds. Do not try to encode the seed in the filename instead — the
existing gen/CSV filenames have no seed field and different seeds would silently
overwrite each other.

Add two helpers and replace the existing exact-match checks:

```python
def is_random(algo):        # covers both arms
    return algo.startswith('random')

def is_layer_matched(algo):
    return algo.startswith('randomlayer')

def draw_seed(algo):        # 'random-s3' -> 3
    return int(algo.split('-s')[-1])
```

Replace `config.args.patch_algo == 'random'` at `eval/eval_runner.py:111` and
`:120` and `:130` with `is_random(config.args.patch_algo)`. Keep
`logit_metric = 'random'` for both arms so gen filenames stay
`{N}_random_steer_{topk}_{test}_gen.json` and the judge regex keeps matching.

### 3.2 Uniform random with a real seed

Thread `draw_seed(patch_algo)` into `retrieve_random_k`. No other change.

### 3.3 Layer-matched random

New function in `eval/logits_handler.py`:

```python
def retrieve_layer_matched_k(config, topk, num_layers, num_heads, seed):
    """Random heads whose per-layer counts exactly match the real ATP selection
    for this (model, source, base, topk)."""
```

Specification:

1. Read the reference ATP selection for the **same `--source` and `--base`**:
   `results/{model}/from_{source}_to_{base}/atp/*/*/eval/numerator_1_targeted_{topk}.csv`
   Any eval/steer subdirectory is fine — the head ranking is identical across
   them within a localization (verified; see `mrr_localization_analysis.py`).
   **Fail loudly if the file is missing.** Never silently fall back to uniform.
2. Count selected heads per layer: `hist[layer] = n_selected_in_that_layer`.
3. For each layer, sample `hist[layer]` heads uniformly **without replacement**
   from that layer's `num_heads` heads, using a seeded RNG.
4. Return a DataFrame with columns `layer,neuron`, sorted by `layer,neuron` —
   identical schema to `retrieve_random_k`.

Do **not** exclude the genuinely-selected heads from the draw. Sampling from all
heads in the layer is the correct null (it asks whether this particular set is
special among sets with the same layer profile). At large k some overlap is
forced by construction; that is expected and fine.

Because `--source` already carries the localization
(`verse-long` vs `verse-single`), the layer profile is matched to the right
localization automatically. No extra CLI flag is needed.

### 3.4 Validate before launching the full grid

Run these and confirm before burning GPU hours:

- Two different seeds produce **different** head sets for the same (model, task, k).
- Layer-matched output has a per-layer histogram **identical** to the reference
  ATP CSV (assert this in code).
- Uniform and layer-matched sets have the **same total size** as the real ATP set
  at each k.
- Outputs land in separate `random-s*/` and `randomlayer-s*/` trees and do not
  touch anything under `atp/`.

## 4. What to run

Steering vector is always **matched to the eval mode** — long-steer with
long-eval, single-steer with single-eval. Do not cross them; that axis is
deliberately fixed in this paper.

Sweeps must match the existing ATP runs exactly so the arms are comparable:

- `topk`: `0.01,0.03,0.05,0.07,0.09,0.1,0.5,1.0`
- `steering_factors` (N): `1,2,4,5,6,8,10`
- 50 test items per condition (unchanged)

### Tier 1 — do this first (validates the pipeline and may settle the question)

Two models, one per attention architecture, chosen because they sit at opposite
ends of the measured effect:

| model | arch | why |
|---|---|---|
| `tiiuae/Falcon3-10B-Instruct` | GQA | largest long-form advantage (+0.222) |
| `Qwen/Qwen1.5-14B-Chat` | MHA | reliably prefers single-token (−0.096) |

Tasks: `verse-long_prose`, `verse-single_prose`, `paragraph-long_sentence`,
`paragraph-single_sentence`. Both eval modes. **3 seeds.** Both arms.

= 2 models × 4 source/base combos × 2 evals × 8 topk × 7 N × 3 seeds × 2 arms.

**Stop and report after Tier 1.** If random already matches the real
localization, the paper's framing changes and there is no point running Tier 2.

### Tier 2 — only after Tier 1 is reviewed

Remaining three models: `google/gemma-3-12b-it`,
`allenai/OLMo-2-1124-13B-DPO`, `Qwen/Qwen1.5-32B-Chat`. Same grid, 3 seeds,
both arms. Add seeds 3–4 to Tier 1 models if the variance across draws looks
large.

**Do not run `phi-4`** — it has been dropped from the paper.

### Command shape

Copy an existing script (e.g. `scripts/falcon3_vp-long.sh`) and change only
`--patch_algo`. Note there is **no `--patch_model` step** for random arms —
there is no attribution to compute, so run the eval step only:

```bash
python run.py --model_id "$model_id" \
              --batch_size 1 \
              --patch_algo "random-s0" \
              --source verse-long \
              --base prose \
              --device cuda:0 \
              --eval_model \
              --kv_caching \
              --eval_test "$REPO/data/${model_name}/verse-long/prose-test.jsonl" \
              --steering \
              --ablation steer \
              --steering_add_path "$REPO/data/${model_name}/verse-long/verse-long-desired-all.jsonl" \
              --steering_sub_path "$REPO/data/${model_name}/verse-long/prose-desired-all.jsonl" \
              --topk_vals 0.01,0.03,0.05,0.07,0.09,0.1,0.5,1.0 \
              --steering_factors 1,2,4,5,6,8,10
```

`$REPO` is this repo's absolute path. **Check it** — several committed scripts
hardcode `/workspace/gcm-interp`, which has been wrong on every machine so far.

## 5. Scoring

Score with the **same judge pipeline and the same metric as the ATP runs**, or
the comparison is void. The metric is chosen by eval mode:

| eval mode | metric | how |
|---|---|---|
| **long-form** (`*-long_eval`) | judge **+ fluency + relevance** | `judge-evals/run_judge.py` → `w_rf` |
| **single-token** (`*-single_eval`) | token/letter matching | `judge-evals/compute_single_accuracies.py` |

Use the same judge model (`unsloth/Meta-Llama-3.1-70B-Instruct-bnb-4bit`) and the
same prompts. The judge pipeline already handles `random` in its filename regex,
so it should run unmodified — but confirm the accuracy JSONs land under
`judge-evals/accuracy/{model}/.../random-s*/...` and not merged into the `atp`
tree.

## 6. Reporting back

For each arm produce, per (model, task, eval, k): the accuracy **mean and spread
across seeds** — the control is a distribution, not a point estimate. Then:

1. **k-curves**: real long-form, real single-token, uniform random, layer-matched
   random, on one plot per (model, task, eval), with the random arms as shaded
   bands.
2. **min-k to 80% of ceiling** for all four arms — the paper's precision metric.
3. Flag immediately if **either** random arm is within noise of the real
   localizations at k ≤ 0.1. That is the headline result and changes the paper.

Do not compute or report cross-eval-mode differences: long-form evals are judged
and single-token evals are token-matched, so those numbers are not comparable.
All comparisons must be **within** an eval mode.

## 7. Known traps

- **Path prefix**: committed scripts may hardcode `/workspace/gcm-interp`. Verify
  before launching.
- **Seeds must not collide with `--seed`**: keep the head-draw seed separate from
  the generation seed (§2, Gap 1).
- **Do not overwrite the `atp/` tree.** Everything here writes to `random-s*/`
  and `randomlayer-s*/`.
- **`num_attention_heads` is the query-head count** for GQA models (Falcon3
  12/layer, Gemma 16/layer, Qwen32B 40/layer). That is the correct dimension —
  the ATP files are indexed by query head. Do not substitute `num_key_value_heads`.
- **Resume is filename-based**: `eval_runner` skips a condition if the gen files
  already exist. If you change the sampling logic mid-run, delete the affected
  outputs or the stale draws will be silently reused.
