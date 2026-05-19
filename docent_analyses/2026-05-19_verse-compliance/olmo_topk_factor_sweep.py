#!/usr/bin/env python3
"""
OLMo-2-1124-13B-DPO verse compliance sweep: topk × steering_factor parameter space.
Filters to runs where steering changed the output (steering_changed_output=true).
Classifies each steered output as verse / degraded_prose / collapsed.
Uses same topk/factor subset as the Qwen sweep for direct comparison.
"""
import sys, os
_cwd = os.getcwd()
if os.path.exists(os.path.join(_cwd, "docent.py")):
    sys.path = [p for p in sys.path if os.path.abspath(p) != os.path.abspath(_cwd) and p not in ("", ".")]

from docent.sdk.client import Docent
from docent.data_models.context_config import AgentRunContextConfig
from docent.data_models.metadata_util import GlobFilter

COLLECTION_ID = "ac3e3937-a10d-4570-ac48-5ff5257698c3"
TOPK_VALUES   = [1, 4, 8]
FACTOR_VALUES = [0.01, 0.05, 0.1, 0.5, 1.0]
SAMPLES_PER_CELL = 10  # 3 × 5 × 10 = 150 total

topk_in   = ", ".join(f"'{v}'" for v in TOPK_VALUES)
factor_in = ", ".join(f"'{v}'" for v in FACTOR_VALUES)

client = Docent()
client.plan_name = "olmo_verse_topk_factor_sweep"

# ---------------------------------------------------------------------------
# Phase 1: Sample and classify
# ---------------------------------------------------------------------------
with client.step_group("Phase 1 — Classify per run"):
    sample_query = client.query(
        COLLECTION_ID,
        f"""
        SELECT sub.run_id AS run
        FROM (
            SELECT
                ar.id AS run_id,
                ar.metadata_json->>'topk'            AS topk,
                ar.metadata_json->>'steering_factor' AS factor,
                ROW_NUMBER() OVER (
                    PARTITION BY ar.metadata_json->>'topk',
                                 ar.metadata_json->>'steering_factor'
                    ORDER BY ar.id
                ) AS rn
            FROM agent_runs ar
            WHERE ar.metadata_json->>'topk'                  IN ({topk_in})
              AND ar.metadata_json->>'steering_factor'        IN ({factor_in})
              AND ar.metadata_json->>'steering_changed_output' = 'true'
        ) AS sub
        WHERE sub.rn <= {SAMPLES_PER_CELL}
        ORDER BY CAST(sub.topk AS INTEGER), CAST(sub.factor AS DOUBLE PRECISION), sub.run_id
        """,
        name=f"Sample {SAMPLES_PER_CELL} changed-output runs per (topk, factor) — {SAMPLES_PER_CELL * len(TOPK_VALUES) * len(FACTOR_VALUES)} total",
    )

    characterize = client.read(
        prompt_template=[
            sample_query.run.as_type("agent_run"),
            """This agent run contains two transcripts:
- **steered**: output after activation patching toward verse style (confirmed to differ from baseline)
- **baseline**: original unsteered output

Classify the **steered** assistant response into exactly one of three categories:

**verse** — Clearly uses line breaks, stanzas, or poetic formatting. May or may not be metrically perfect, but structurally distinct from flowing prose.

**degraded_prose** — Recognizable prose, but shortened, less fluent, slightly restructured, or with isolated verse-like hints (e.g., one or two line breaks) without committing to verse format. Also includes outputs that are plausible prose but noticeably different in style or length from a normal response.

**collapsed** — Generation failure: empty output, highly repetitive tokens, non-English / mixed-language text, numeric strings, garbled or incoherent content.

Cite the key steered text that determines your classification. Be brief.
""",
        ],
        context_configs={
            "run": AgentRunContextConfig(
                agent_run_metadata=GlobFilter(include=("topk", "steering_factor", "condition")),
                transcript_metadata=GlobFilter(include=("response_type",)),
            )
        },
        model="openai/gpt-5.5",
        output_schema={
            "type": "object",
            "properties": {
                "output_type": {
                    "type": "string",
                    "enum": ["verse", "degraded_prose", "collapsed"],
                },
                "reasoning": {"type": "string", "citations": True},
            },
            "required": ["output_type", "reasoning"],
        },
        name=f"Classify steered outputs: verse / degraded_prose / collapsed ({SAMPLES_PER_CELL * len(TOPK_VALUES) * len(FACTOR_VALUES)} runs)",
    )

# ---------------------------------------------------------------------------
# Phase 2: Per-cell aggregation + summaries
# ---------------------------------------------------------------------------
with client.step_group("Phase 2 — Per-cell summaries"):
    per_cell_query = client.query(
        COLLECTION_ID,
        f"""
        SELECT
            topk,
            factor,
            COUNT(topk)                                                       AS n,
            SUM(CASE WHEN output_type = 'verse'          THEN 1 ELSE 0 END)  AS verse_n,
            SUM(CASE WHEN output_type = 'degraded_prose' THEN 1 ELSE 0 END)  AS prose_n,
            SUM(CASE WHEN output_type = 'collapsed'      THEN 1 ELSE 0 END)  AS collapse_n,
            array_agg(result_id)                                              AS per_run_results
        FROM (
            SELECT
                ar.metadata_json->>'topk'            AS topk,
                ar.metadata_json->>'steering_factor' AS factor,
                rr.output->>'output_type'            AS output_type,
                rr.id                                AS result_id
            FROM reading_results rr
            JOIN reading_result_links rrl
                ON rrl.result_id = rr.id
            JOIN agent_runs ar
                ON CAST(ar.id AS text) = rr.arguments_dict->'run'->>'id'
            WHERE rrl.reading_id = '{characterize}'
        ) AS subq
        GROUP BY topk, factor
        ORDER BY CAST(topk AS INTEGER), CAST(factor AS DOUBLE PRECISION)
        """,
        name="Per-cell counts: verse / degraded_prose / collapsed (topk × steering_factor)",
    )

    cell_summaries = client.read(
        prompt_template=[
            "topk=", per_cell_query.topk.as_type("text"),
            "  steering_factor=", per_cell_query.factor.as_type("text"),
            "  (n=", per_cell_query.n.as_type("text"),
            "  verse=", per_cell_query.verse_n.as_type("text"),
            "  prose=", per_cell_query.prose_n.as_type("text"),
            "  collapsed=", per_cell_query.collapse_n.as_type("text"), ")\n\n",
            "Per-run characterizations:\n",
            per_cell_query.per_run_results.as_type("reading_result", is_list=True),
            """
Write 2–3 sentences characterizing this (topk, steering_factor) cell:
- What fraction are verse / degraded prose / collapsed?
- What does the verse or degraded output concretely look like when it appears?
- Any notable patterns (e.g., verse only at high alpha, collapse mostly numeric)?
Cite specific examples.
""",
        ],
        model="openai/gpt-5.5",
        name="Per-cell characterization (15 cells: topk × steering_factor)",
    )

# ---------------------------------------------------------------------------
# Phase 3: Final threshold report
# ---------------------------------------------------------------------------
with client.step_group("Phase 3 — Threshold report"):
    all_summaries_query = client.query(
        COLLECTION_ID,
        f"""
        SELECT array_agg(rr.id) AS summaries
        FROM reading_results rr
        JOIN reading_result_links rrl ON rrl.result_id = rr.id
        WHERE rrl.reading_id = '{cell_summaries}'
        """,
        name="Collect all 15 per-cell summaries",
    )

    report = client.read(
        prompt_template=[
            f"""You are analyzing when verse output first appears reliably across an activation-patching parameter sweep on OLMo-2-1124-13B-DPO.

**Setup:** activation patching steers the model from prose toward verse. All runs here confirmed that steering *changed* the output.

**Parameter grid (pooled across all 4 experimental conditions):**
- topk ∈ {TOPK_VALUES} (number of patches applied)
- steering_factor (alpha) ∈ {FACTOR_VALUES}
- n = {SAMPLES_PER_CELL} runs per cell

**Output categories:**
- **verse**: clear line breaks / stanza structure
- **degraded_prose**: prose that's shortened, less fluent, or with minor verse hints
- **collapsed**: empty / garbled / non-English / repetitive failure

Per-cell characterizations (ordered by topk then factor):
""",
            all_summaries_query.summaries.as_type("reading_result", is_list=True),
            f"""
Write a structured report in markdown:

## Parameter Grid

A table with **rows = topk** ({TOPK_VALUES}), **columns = steering_factor** ({FACTOR_VALUES}).
Cell format: `V/P/C` counts — e.g. `3V 5P 2C` (V=verse, P=prose, C=collapse).

## Where Verse First Appears

At which (topk, steering_factor) combinations does verse *first* emerge?
At which combinations does it become *reliable* (≥ 2 of {SAMPLES_PER_CELL} runs, i.e. 20%+)?

## Low vs. High Steering Factor

Compare behavior at low alpha (0.01–0.1) vs. high alpha (0.5–1.0) across the three topk values.
Does increasing alpha reliably shift outputs from prose → verse, or does it jump to collapse?

## Effect of topk

Does higher topk widen the window where verse appears before collapse,
or does it just shift the onset point?

## Best Onset Combination

Based on the evidence, which (topk, steering_factor) offers the best tradeoff —
verse appearing reliably without too much collapse?

## Comparison with Qwen1.5-14B-Chat

The Qwen sweep (same parameter grid, same task) found:
- Verse first appeared at topk=4, α=0.05 (2/10 runs)
- topk=8 caused collapse at α=0.1 (10/10 collapsed)
- Best onset: topk=4, α=0.05 (2V 8P 0C)

How does OLMo compare? Does verse emerge at lower or higher alpha? Is collapse more or less aggressive?

## Implications

What does this pattern suggest about the geometry of verse vs. prose activations in OLMo
and how the transfer sensitivity compares to Qwen1.5-14B-Chat?

Cite per-cell summaries throughout.
""",
        ],
        model="openai/gpt-5.5",
        name="OLMo verse compliance threshold report (topk × steering_factor)",
    )

client.flush(auto_approve=True)
report_text = report.results[0].output

print("\n" + "=" * 70)
print("OLMO VERSE THRESHOLD REPORT  (topk × steering_factor)")
print("=" * 70)
print(report_text)
print(f"\nFull analysis: https://docent.transluce.org/dashboard/{COLLECTION_ID}")
