# --localization_ctx 2x2 — Qwen1.5-14B-Chat / verse

Two full grids were run (16 cells, 1792 generations, 50 items each, one machine):

| grid | site | span | trees |
|---|---|---|---|
| legacy | `o_proj_out` | `legacy` | `atp-<ctx>` |
| **corrected** | `o_proj_in` | `full` (respfix) | `atp-o_proj_in-respfix-<ctx>` |

**The corrected grid supersedes the legacy one.** Under `legacy`, the scored
span starts one position late and never scores the first response token — which
for `-single` data is the entire answer (`eval/response_span.py`: gradient mass
exactly 0 at the answer-predicting position). Under `o_proj_out`, a selected
"head" is the 128 residual coordinates a head writes into AFTER W_O has mixed
them; only at `o_proj_in` is block u actually head u.

```
net_effect = grad(A_base_full) * ( A_src - A_base_patch )
             \___ fixed ____/     \____ varied by ctx ___/
```

| ctx | A_src | A_base_patch |
|---|---|---|
| `br-sq` | source prompt only | base prompt+response  (= current method) |
| `br-sr` | source prompt+**response** | base prompt+response |
| `bq-sq` | source prompt only | base **prompt only** |
| `bq-sr` | source prompt+**response** | base **prompt only** |

## 1. HEADLINE (corrected grid): ctx makes essentially no difference

Judged `w_rf`, max over N / mean over N:

| localization | `br-sq` (current) | `br-sr` | `bq-sq` | `bq-sr` |
|---|---|---|---|---|
| verse-long   | 0.980 / 0.449 | 1.000 / 0.557 | 0.980 / 0.523 | 0.980 / 0.609 |
| verse-single | 1.000 / 0.640 | 1.000 / 0.609 | 1.000 / 0.637 | 1.000 / 0.609 |

min-k to 80% of ceiling is **0.01** for all four in verse-long and **0.03** for
all four in verse-single. In verse-single the four k-curves are nearly
superimposed (0.060 / 0.980 / 1.000 / ... identical across cells).

**Conclusion: once the site and span are correct, which sequences supply the
differenced activations does not measurably change steering performance.**

## 2. RETRACTED — the legacy-grid headline was an artefact

An earlier version of this document reported, from the legacy grid, that "the
source's response is what matters" and that `bq-sq` "isn't learning anything".
**Both claims fail under the corrected grid.**

Judged `w_rf`, max over N:

| localization | ctx | legacy | corrected |
|---|---|---|---|
| verse-long | `br-sq` | 0.800 | 0.980 |
| verse-long | `br-sr` | 0.940 | 1.000 |
| verse-long | `bq-sq` | **0.200** | **0.980** |
| verse-long | `bq-sr` | 0.940 | 0.980 |
| verse-single | `br-sq` | **0.080** | **1.000** |
| verse-single | `br-sr` | 0.940 | 1.000 |
| verse-single | `bq-sq` | 0.440 | 1.000 |
| verse-single | `bq-sr` | 0.940 | 1.000 |

`bq-sq` 0.200 -> 0.980 and `br-sq`/verse-single 0.080 -> 1.000 are the two
largest moves. The second is exactly what `eval/response_span.py` predicts: under
`legacy` the single-token localization never scores its answer token.

It was also claimed the effect held independently of the broken metric because it
appeared in the long-form localization too. That was wrong — the long-form
localization is where `bq-sq` moved 0.200 -> 0.980.

## 3. What actually matters: the site and the span

The real effect in this work is not ctx but the site/span correction, which
improves EVERY cell:

- min-k to 80% of ceiling: 0.05-0.10 (legacy) -> **0.01-0.03** (corrected).
- Generation quality at the winning cells changes qualitatively. Legacy best
  cells produced degenerate output (`99999990 a．09.99991999...`); corrected
  cells produce fluent verse ("In ancient days, art was born, / With Greeks and
  Romans, forms adorned, ...").

## 4. Mechanism (measured, still true)

`analysis/ctx_attribution_mass.py`, verse-long, one batch:

```
src_qs  : PAD at 102/102 response positions
src_full: PAD at   0/102 response positions
base_qs : PAD at 102/102 response positions

config            |attr| @prompt  |attr| @response
br-sq (current)          5.4%           94.6%
br-sr                    5.3%           94.7%
bq-sq                   28.6%           71.4%
bq-sr                    5.3%           94.7%
```

A prompt-only source ends at the generation prompt, so after `align_toks`
everything past the marker is padding, and ~95% of the attribution magnitude sits
at those positions. This is a real property of the pipeline and describes what
the delta represents — `br-sq` differences real response activations against
pad-position activations (which do still carry the verse prompt's context).

**But note what section 1 shows: this does NOT translate into a performance
difference.** The padding asymmetry is a fact about the intervention's semantics,
not a defect — and it is not costing anything measurable.

## 5. Caveats

- **Saturation.** Corrected ceilings are 0.98-1.00, which compresses any real
  difference between cells. A harder eval would be needed to separate them.
- **n=50 per cell, SE ~0.07**; max-over-N inflates the winner. mean-over-N is
  reported alongside for that reason.
- **One model, one task family, one draw.** No seed spread.
- **k=1.0 gate passes only trivially** (all cells saturated at 0.000), which
  CLAUDE.md counts as weak provenance evidence, not a pass.
- **`bq-*` are contrasts, not patching estimates** — ATP expands around the
  response-bearing base, and those cells subtract a different point.
- Head-set Jaccard is NOT comparable across the two grids: `o_proj_in` and
  `o_proj_out` index different objects.

## 6. Open question this does not answer

No cell here separates "the verse instruction changed the representation" from
"different response text is present" — `br-sr` changes both at once. The
controlled version (source prompt + the BASE's own response, so response tokens
are held identical and only the instruction differs) has not been run;
`get_templated_prompts(_base_completion=...)` in `data_handler.py:247` already
supports it.

## Reproduce

```bash
CTX_GRID=fixed  python analysis/ctx_sweep_report.py     # tables
CTX_GRID=fixed  python analysis/ctx_plots.py            # k-curve figures
CTX_GRID=fixed  python analysis/ctx_head_overlap.py     # head/layer Jaccard
PATCH_SITE=o_proj_in RESPONSE_SPAN=full bash scripts/qwen_verse_ctx_sweep.sh
ALGOS="atp-o_proj_in-respfix-brsq ..." bash judge-evals/scripts/ctx_sweep_judge.sh
```
