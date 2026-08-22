# Localizing and steering on `o_proj.input` (2026-08-22)

Two changes to the pipeline, both opt-in and both path-isolated, plus the verse
experiments that motivated them. Nothing here alters any published number: the
default flags reproduce the existing `results/<model>/from_*/atp/` layout
byte-for-byte.

## 1. `--patch_site {o_proj_out,o_proj_in}`

`o_proj.output` is `W_O @ concat(z_1..z_H)`, so **every** coordinate of it is a
sum over all heads. A coordinate block there is not a head; it is a block of
residual-stream coordinates that no head owns, `hidden_size // num_heads` wide.
For gemma-3-12b that is 3840/16 = 240 while the true `head_dim` is 256, so the
blocks do not even have head width (FINDINGS 0.1).

`o_proj.input` is `concat(z_1..z_H)`: block `u` **is** head `u`, `head_dim` wide,
including for GQA models (the axis is num query heads x head_dim). A write there
reaches the residual stream through the model's own `W_O`.

Implementation: `eval/patch_site.py`. The site is threaded through attribution
(`patching_utils.get_activations`), ACP (`patch_heads`), the steering-vector
cache (`eval/activations.py`), generation (`eval/generation.py`, both the
kv-caching and `model.all()` branches), the lm-eval hook path
(`eval/eval_extant.py`, which switches to a forward **pre**-hook), and the random
control arms. `model_handler.dim` carries the site-dependent block width, so every
call site follows from one line.

Verified on a tiny random Llama with `head_dim != hidden//n_heads`: in-place
writes to `o_proj.input` propagate exactly (`o_proj.output == linear(edited_input)`
to float equality), `retain_grad` yields real gradients, and a one-head edit moves
**32/32** residual coordinates -- which is why output blocks cannot be heads.

## 2. `--response_span {legacy,full}`

`get_response_logits` starts its sum one position late and therefore **never
scores the first response token**. For `-long` data that drops 1 token of ~127.
For `-single` data the assistant turn is a single letter, so the dropped token is
the whole answer: the summed span is just `<|im_end|>`/`<end_of_turn>` and `\n`.

Verified four ways on the shipped code: 2 summed terms for verse-single vs 126
for verse-long; gradient mass exactly 0 at the answer-predicting position; the
returned log-likelihood bit-identical under +/-100 on the answer token's logit
(-23.358849 both ways) while the `<|im_end|>` logit moves it; unchanged with and
without padding; reproduced on gemma with a different tokenizer and template.
See `eval/response_span.py` for the full write-up.

Default stays `legacy` because flipping it changes every `-single` localization
and resume here is filename-based, so a mid-stream change would silently mix two
metrics in one tree. `full` gets its own results tree (`-respfix`).

## 3. What was run

Qwen1.5-14B-Chat and gemma-3-12b-it, verse, **long-form steering vector with
long-form eval** (matched, per CLAUDE.md section 4), both localizations, at
`--patch_site o_proj_in`. Sweeps match each model's existing atp runs exactly
(topk 0.01-1.0 x 8, N = 1,2,4,5,6,8,10, 50 items/cell). Judged with the same
pipeline and settings as the atp runs (`--no_judge_prefill`, `--batch_size 16`).

Drivers: `scripts/{qwen,gemma}_vp-o_proj_in.sh`,
`judge-evals/scripts/verse_o_proj_in_judge.sh`.

## 4. Results

`k_curves_max_over_N.csv` (max over N, as the paper reports it) and
`w_rf_by_site.csv` (the full 448-cell grid). Figures:
`site_compare_<model>.png/pdf`, built by `fig_site_compare.py`.

**Long-form localization gets dramatically more precise at small budgets:**

| k | Qwen in / out | gemma in / out |
|---|---|---|
| 0.01 | **0.92** / 0.04 | **0.86** / 0.12 |
| 0.03 | **0.96** / 0.62 | **0.86** / 0.28 |
| 0.05 | **0.90** / 0.20 | 0.82 / 0.72 |

On gemma the two sites converge from k=0.05 up; the input site simply reaches its
ceiling ~5x sooner. On Qwen the input site wins at k<=0.05 then falls off.

**Single-token localization is flat at both sites** (max 0.06-0.12 on Qwen at
k<=0.1; gemma similar at low k, with the *output* site ahead above k=0.09).
Consistent with the metric bug: a localization that scored turn-closing does not
steer wherever you inject it.

**Head-set overlap.** Cross-site unit overlap is at or near chance (0.00 at
k=0.01, ~2x chance at k=0.1) -- as it must be, since a unit index names a
different object at each site. The meaningful comparison is layers, which *are*
the same objects: layer-set Jaccard 0.48-0.77, per-layer count rho 0.61-0.87,
mean depth within 0.04. The attribution finds the same depth regardless of which
tensor it is scored on.

**The paper's structural findings survive both changes.** Long-form vs
single-token stays near-chance disjoint at every site and under the corrected
metric (0.00-0.18 vs 0.005-0.053 chance), and long-form still localizes earlier
than single-token (-0.029 shipped, -0.032 at the input site, and the gap *widens*
to -0.03..-0.09 under `--response_span full`).

## 5. Corrected-metric attribution

`attribution/` holds, per cell, the mean-over-items attribution matrix and the
top-k head sets, for both `legacy` and `respfix`. These are CSVs because `*.pt`
is gitignored, so the raw tensors do not survive. The `respfix` cells were
produced by `respfix_attribution_driver.py`, which monkeypatches the metric and
redirects the output prefix -- it predates the `--response_span` flag and is kept
only for provenance; use the flag instead.

Switching to `full` replaces most of the single-token head set and barely touches
long-form (Jaccard vs legacy, k=0.01..0.1):

| arm | o_proj_out | o_proj_in |
|---|---|---|
| long-form (control) | 0.88 .. 0.87 | 0.78 .. 0.89 |
| single-token | 0.10 .. 0.17 | **0.00 .. 0.09** |

## 6. Caveats

- **Dose is not comparable across sites.** The steering vector is normalized in
  the space it is applied to, so a given N is a different perturbation at each
  site. At k=1.0 the *output* site wins on gemma (0.80 vs 0.36), which is this
  effect, not a head-selection effect. Compare k-curves within a site.
- **Section 2's provenance gate does not pass, and its diagnosis looks wrong.**
  The unsteered baseline differs on **25/50** items between two runs launched from
  the same script on the same machine 90 minutes apart, with `no_deterministic:
  false` in both configs. Queries are identical 50/50; continuations share a ~973
  character prefix then a token flips. That is the same rate and signature section
  2 attributes to "different machine, GPU, or library versions" -- so it is
  plain run-to-run nondeterminism across processes, not provenance.
  `determinism.py` pins cuBLAS and `use_deterministic_algorithms(warn_only=True)`
  but leaves the SDP backend unpinned and cannot cover bitsandbytes NF4 matmuls.
  The effects above (0.7-0.9) are far larger than section 2's measured max
  artefact (0.120), but the gate is not clean.
- **One cell, one task.** verse only, two models. Falcon3 (`3072/12 = head_dim
  256`, aligned) is the clean third point for separating "unmixing W_O" from
  "block misalignment": gemma's low-k gain (+0.74) is *not* larger than Qwen's
  (+0.88), which is mild evidence against misalignment being the driver.
- **No steering evidence for the corrected metric.** Only attribution was re-run
  under `--response_span full`; whether those head sets steer better is unmeasured.
