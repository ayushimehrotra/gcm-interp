#!/usr/bin/env python3
"""
Qwen1.5-32B-Chat verse compliance analysis: compare steered outputs across
4 activation-patching conditions.
Collection: 51875bbe-91b2-4be3-a1e2-b0ccf0be0f65
"""
import sys, os
_cwd = os.getcwd()
if os.path.exists(os.path.join(_cwd, "docent.py")):
    sys.path = [p for p in sys.path if os.path.abspath(p) != os.path.abspath(_cwd) and p not in ("", ".")]

from docent.sdk.client import Docent
from docent.data_models.context_config import AgentRunContextConfig
from docent.data_models.metadata_util import GlobFilter

COLLECTION_ID = "51875bbe-91b2-4be3-a1e2-b0ccf0be0f65"
SAMPLES_PER_CONDITION = 25

client = Docent()
client.plan_name = "qwen32b_verse_compliance_comparison"

with client.step_group("Phase 1 — Per-run characterization"):
    sample_query = client.query(
        COLLECTION_ID,
        f"""
        SELECT sub.run_id AS run, sub.condition AS condition
        FROM (
            SELECT
                ar.id AS run_id,
                ar.metadata_json->>'condition' AS condition,
                ROW_NUMBER() OVER (
                    PARTITION BY ar.metadata_json->>'condition'
                    ORDER BY ar.id
                ) AS rn
            FROM agent_runs ar
        ) AS sub
        WHERE sub.rn <= {SAMPLES_PER_CONDITION}
        ORDER BY sub.condition, sub.run_id
        """,
        name=f"Sample {SAMPLES_PER_CONDITION} runs per condition ({SAMPLES_PER_CONDITION*4} total)",
    )

    characterize = client.read(
        prompt_template=[
            sample_query.run.as_type("agent_run"),
            """This agent run holds two transcripts:
- **steered**: the model's output after activation patching toward verse style
- **baseline**: the original unsteered model output

The model (Qwen1.5-32B-Chat) was prompted to answer a factual question in prose.
Activation patching was applied to transfer verse-style activations and steer it toward verse output.

Analyze the **steered** transcript's assistant response. Answer each question with brief reasoning and a citation:

1. **Verse structure** — Does the steered output use line breaks, stanza structure, or poetic formatting instead of flowing prose paragraphs?
2. **Metric coherence** — If verse structure is present, does it have recognizable meter, rhythm, or rhyme? Answer "yes", "no", or "n/a" if no verse structure present.
3. **Language collapse** — Does the output contain non-English tokens, highly repetitive phrases, garbled/incoherent text, or other signs of generation failure?
4. **Identical to baseline** — Is the steered output essentially the same text as the baseline output (i.e., did the activation patch have no observable effect)?
""",
        ],
        context_configs={
            "run": AgentRunContextConfig(
                agent_run_metadata=GlobFilter(include=("condition", "localization", "steering", "topk", "steering_factor")),
                transcript_metadata=GlobFilter(include=("response_type",)),
            )
        },
        model="openai/gpt-5.5",
        output_schema={
            "type": "object",
            "properties": {
                "reasoning": {"type": "string", "citations": True},
                "uses_verse_structure": {"type": "boolean"},
                "verse_metrically_coherent": {"type": "string", "enum": ["yes", "no", "n/a"]},
                "language_collapse": {"type": "boolean"},
                "identical_to_baseline": {"type": "boolean"},
            },
            "required": [
                "reasoning", "uses_verse_structure", "verse_metrically_coherent",
                "language_collapse", "identical_to_baseline",
            ],
        },
        name=f"Characterize steered vs. baseline ({SAMPLES_PER_CONDITION*4} runs)",
    )

with client.step_group("Phase 2 — Per-condition summaries"):
    per_cond_query = client.query(
        COLLECTION_ID,
        f"""
        SELECT
            ar.metadata_json->>'condition' AS condition,
            array_agg(rr.id) AS per_run_results
        FROM reading_results rr
        JOIN reading_result_links rrl ON rrl.result_id = rr.id
        JOIN agent_runs ar
            ON CAST(ar.id AS text) = rr.arguments_dict->'run'->>'id'
        WHERE rrl.reading_id = '{characterize}'
        GROUP BY ar.metadata_json->>'condition'
        ORDER BY ar.metadata_json->>'condition'
        """,
        name="Group per-run results by condition",
    )

    cond_summaries = client.read(
        prompt_template=[
            "Activation-patching condition: **",
            per_cond_query.condition.as_type("text"),
            "**\n\nPer-run characterizations (25 runs):\n",
            per_cond_query.per_run_results.as_type("reading_result", is_list=True),
            """
Summarize verse compliance for this condition in 2–3 paragraphs covering:
- What fraction of steered outputs use verse structure, and what does that verse look like?
- Is the verse metrically coherent or fragmented/garbled?
- How often does language collapse occur, and what does it look like?
- How often is the steered output identical to the baseline (steering had no effect)?
Cite specific examples from the characterizations above.
""",
        ],
        model="openai/gpt-5.5",
        name="Per-condition verse compliance summary (4 conditions)",
    )

with client.step_group("Phase 3 — Comparative report"):
    all_summaries_query = client.query(
        COLLECTION_ID,
        f"""
        SELECT array_agg(rr.id) AS summaries
        FROM reading_results rr
        JOIN reading_result_links rrl ON rrl.result_id = rr.id
        WHERE rrl.reading_id = '{cond_summaries}'
        """,
        name="Collect all four per-condition summaries",
    )

    report = client.read(
        prompt_template=[
            """You are writing a comparative analysis of verse compliance across four activation-patching conditions applied to Qwen1.5-32B-Chat.

**Experimental context:**
- Task: model was prompted to respond in prose to factual questions
- Intervention: activation patching steered it toward verse-style output
- 2×2 design:
  - **Localization** (which activations were patched): `longform` (from a long verse example) vs. `single_token` (from a single-token verse example)
  - **Steering** (direction of the activation vector): `longform_steer` vs. `single_token_steer`
- Four conditions: longform_loc__longform_steer, longform_loc__single_token_steer, single_token_loc__longform_steer, single_token_loc__single_token_steer

Per-condition summaries (each based on 25 sampled runs):
""",
            all_summaries_query.summaries.as_type("reading_result", is_list=True),
            """
Write a structured comparative report in markdown:

## Summary Table
| Condition | Verse Structure | Metric Coherence | Language Collapse | Identical to Baseline |
|-----------|----------------|-----------------|-------------------|----------------------|
(Fill in approximate rates or qualitative labels: high/medium/low)

## Condition-by-Condition Analysis
One paragraph per condition.

## Cross-Condition Patterns
- Does the **localization method** (longform vs. single_token) drive the most variance?
- Does the **steering direction** matter more or less than localization?
- Which condition most effectively transfers verse style without causing collapse?
- Which condition causes the most language collapse?

## Ranking
Rank the four conditions from most to least effective at verse transfer. Justify with citations.

## Implications
What do these results suggest about how activation patching transfers stylistic behaviors across conditions in Qwen1.5-32B-Chat?
""",
        ],
        model="openai/gpt-5.5",
        name="Comparative verse compliance report (Qwen1.5-32B-Chat)",
    )

client.flush(auto_approve=True)
report_text = report.results[0].output

print("\n" + "=" * 70)
print("QWEN1.5-32B-CHAT VERSE COMPLIANCE REPORT")
print("=" * 70)
print(report_text)
print(f"\nFull analysis: https://docent.transluce.org/dashboard/{COLLECTION_ID}")
