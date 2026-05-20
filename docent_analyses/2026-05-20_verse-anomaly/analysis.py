#!/usr/bin/env python3
"""
Verse steering qualitative analysis: anomaly detection + target similarity comparison.
Collection: f4e5c48c-8fb1-4668-84d7-6e5932134ecd

Design (all phases run with auto_approve=True):
  Phase 1  (448 calls): Classify anomaly type in sampled steered outputs
  Phase 2  (DQL only):  Aggregate anomaly counts by condition / topk / steering_factor
  Phase 3  (4 calls):   Per-condition anomaly narrative synthesis
  Phase 4  (144 calls): Pairwise steered-vs-target comparison (36 pairs × 4 conditions)
  Phase 5  (4 calls):   Per-condition target-similarity synthesis
  Phase 6  (1 call):    Final comparative report + rankings
"""
import sys, os
_cwd = os.getcwd()
if os.path.exists(os.path.join(_cwd, "docent.py")):
    sys.path = [
        p for p in sys.path
        if os.path.abspath(p) != os.path.abspath(_cwd) and p not in ("", ".")
    ]

from docent.sdk.client import Docent
from docent.data_models.context_config import AgentRunContextConfig
from docent.data_models.metadata_util import GlobFilter

COLLECTION_ID = "f4e5c48c-8fb1-4668-84d7-6e5932134ecd"

client = Docent()
client.plan_name = "verse_anomaly_and_target_comparison_v2"

# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 — Anomaly detection (448 LLM calls)
#
# Sample 2 runs per (condition × topk × factor) cell.
# 4 conditions × 7 topk × 8 factors × 2 = 448 runs total.
# UUID ordering is pseudo-random and stable across re-runs.
# ─────────────────────────────────────────────────────────────────────────────

with client.step_group("Phase 1 — Anomaly detection"):
    anomaly_sample = client.query(
        COLLECTION_ID,
        """
        SELECT sub.run AS run,
               sub.condition AS condition,
               sub.topk AS topk,
               sub.steering_factor AS steering_factor
        FROM (
            SELECT
                ar.id AS run,
                ar.metadata_json->>'condition' AS condition,
                ar.metadata_json->>'topk' AS topk,
                ar.metadata_json->>'steering_factor' AS steering_factor,
                ROW_NUMBER() OVER (
                    PARTITION BY ar.metadata_json->>'condition',
                                 ar.metadata_json->>'topk',
                                 ar.metadata_json->>'steering_factor'
                    ORDER BY ar.id
                ) AS rn
            FROM agent_runs ar
            WHERE ar.metadata_json->>'run_type' = 'intervention'
        ) AS sub
        WHERE sub.rn BETWEEN 3 AND 4
        ORDER BY sub.condition,
                 CAST(sub.topk AS NUMERIC),
                 CAST(sub.steering_factor AS NUMERIC),
                 sub.run
        """,
        name="Sample 2 runs per (condition × topk × factor) — 448 runs total",
    )

    detect_anomalies = client.read(
        prompt_template=[
            anomaly_sample.run.as_type("agent_run"),
            """
This agent run has two transcripts:
- **steered** (role: steered): the model's output AFTER activation-patching toward verse style
- **baseline** (role: baseline): the model's original output BEFORE patching

The model (Qwen1.5-14B-Chat) was prompted "Respond in prose and concisely." followed by a
philosophical question. Activation patching steered it toward producing verse-style output.

Analyze ONLY the **steered** transcript's assistant response.

1. Is the steered output formatted as verse (line breaks, stanzas, poetic structure)?
2. Classify the output as exactly one anomaly type:
   - **clean_output**: clean prose or verse response with no technical problems
   - **language_collapse**: non-English tokens, garbled/incoherent text, jumbled characters
   - **repetition_loop**: excessive repetition of phrases/sentences, circular or looping text
   - **truncated**: output is clearly incomplete or abruptly cut off mid-sentence
   - **refusal_commentary**: model refuses the task, meta-comments on the instructions, or
     adds lengthy disclaimers instead of answering the question
   - **other_anomaly**: any other unexpected/problematic behavior not covered above

Cite specific text from the steered transcript to justify your classification.
""",
        ],
        context_configs={
            "run": AgentRunContextConfig(
                agent_run_metadata=GlobFilter(include=(
                    "condition", "topk", "steering_factor",
                    "steering_changed_output", "prompt_id", "prompt_text",
                )),
                transcript_metadata=GlobFilter(include=("role",)),
            )
        },
        model="openai/gpt-5.4-mini",
        max_new_tokens=350,
        output_schema={
            "type": "object",
            "properties": {
                "reasoning": {"type": "string", "citations": True},
                "is_verse": {"type": "boolean"},
                "anomaly_type": {
                    "type": "string",
                    "enum": [
                        "clean_output",
                        "language_collapse",
                        "repetition_loop",
                        "truncated",
                        "refusal_commentary",
                        "other_anomaly",
                    ],
                },
                "anomaly_description": {"type": "string"},
            },
            "required": ["reasoning", "is_verse", "anomaly_type", "anomaly_description"],
        },
        name="Classify anomaly type in 448 steered outputs",
    )

# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 — Anomaly aggregation tables (DQL-only)
# ─────────────────────────────────────────────────────────────────────────────

with client.step_group("Phase 2 — Anomaly aggregation"):
    client.query(
        COLLECTION_ID,
        f"""
        SELECT condition, anomaly_type, COUNT(anomaly_type) AS count
        FROM (
            SELECT
                ar.metadata_json->>'condition' AS condition,
                rr.output->>'anomaly_type' AS anomaly_type
            FROM reading_results rr
            JOIN reading_result_links rrl ON rrl.result_id = rr.id
            JOIN agent_runs ar
                ON CAST(ar.id AS text) = rr.arguments_dict->'run'->>'id'
            WHERE rrl.reading_id = '{detect_anomalies}'
        ) AS subq
        GROUP BY condition, anomaly_type
        ORDER BY condition, count DESC
        """,
        name="Anomaly counts by condition and type",
    )

    client.query(
        COLLECTION_ID,
        f"""
        SELECT topk, anomaly_type, COUNT(anomaly_type) AS count
        FROM (
            SELECT
                ar.metadata_json->>'topk' AS topk,
                rr.output->>'anomaly_type' AS anomaly_type
            FROM reading_results rr
            JOIN reading_result_links rrl ON rrl.result_id = rr.id
            JOIN agent_runs ar
                ON CAST(ar.id AS text) = rr.arguments_dict->'run'->>'id'
            WHERE rrl.reading_id = '{detect_anomalies}'
                AND rr.output->>'anomaly_type' != 'clean_output'
        ) AS subq
        GROUP BY topk, anomaly_type
        ORDER BY CAST(topk AS NUMERIC), count DESC
        """,
        name="Non-clean anomaly counts by topk",
    )

    client.query(
        COLLECTION_ID,
        f"""
        SELECT steering_factor, anomaly_type, COUNT(anomaly_type) AS count
        FROM (
            SELECT
                ar.metadata_json->>'steering_factor' AS steering_factor,
                rr.output->>'anomaly_type' AS anomaly_type
            FROM reading_results rr
            JOIN reading_result_links rrl ON rrl.result_id = rr.id
            JOIN agent_runs ar
                ON CAST(ar.id AS text) = rr.arguments_dict->'run'->>'id'
            WHERE rrl.reading_id = '{detect_anomalies}'
                AND rr.output->>'anomaly_type' != 'clean_output'
        ) AS subq
        GROUP BY steering_factor, anomaly_type
        ORDER BY CAST(steering_factor AS NUMERIC), count DESC
        """,
        name="Non-clean anomaly counts by steering_factor",
    )

    client.query(
        COLLECTION_ID,
        f"""
        SELECT condition, is_verse, COUNT(is_verse) AS count
        FROM (
            SELECT
                ar.metadata_json->>'condition' AS condition,
                rr.output->>'is_verse' AS is_verse
            FROM reading_results rr
            JOIN reading_result_links rrl ON rrl.result_id = rr.id
            JOIN agent_runs ar
                ON CAST(ar.id AS text) = rr.arguments_dict->'run'->>'id'
            WHERE rrl.reading_id = '{detect_anomalies}'
        ) AS subq
        GROUP BY condition, is_verse
        ORDER BY condition, is_verse DESC
        """,
        name="Verse-structure rate by condition",
    )

# ─────────────────────────────────────────────────────────────────────────────
# Phase 3 — Per-condition anomaly synthesis (4 LLM calls)
#
# Aggregate all 112 per-run anomaly results per condition. With max_new_tokens=350
# on Phase 1, each result is compact enough (~300 tokens) for the synthesis to
# stay well within context limits (112 × 300 ≈ 33K tokens).
# ─────────────────────────────────────────────────────────────────────────────

with client.step_group("Phase 3 — Per-condition anomaly synthesis"):
    per_cond_anomaly_q = client.query(
        COLLECTION_ID,
        f"""
        SELECT
            ar.metadata_json->>'condition' AS condition,
            array_agg(rr.id) AS anomaly_results
        FROM reading_results rr
        JOIN reading_result_links rrl ON rrl.result_id = rr.id
        JOIN agent_runs ar
            ON CAST(ar.id AS text) = rr.arguments_dict->'run'->>'id'
        WHERE rrl.reading_id = '{detect_anomalies}'
        GROUP BY ar.metadata_json->>'condition'
        ORDER BY ar.metadata_json->>'condition'
        """,
        name="Group anomaly results by condition (4 rows)",
    )

    cond_anomaly_summaries = client.read(
        prompt_template=[
            "Condition: **",
            per_cond_anomaly_q.condition.as_type("text"),
            "**\n\nPer-run anomaly classifications (112 runs sampled across all topk and "
            "steering_factor values for this condition):\n",
            per_cond_anomaly_q.anomaly_results.as_type("reading_result", is_list=True),
            """

Summarize the anomaly pattern for this condition in 2–3 paragraphs covering:
1. What fraction of outputs are clean vs. anomalous? What types of anomalies appear?
2. At which topk and/or steering_factor values do anomalies concentrate? Cite specific
   runs with topk and steering_factor values from the reasoning fields.
3. What does the typical anomaly look like? Cite specific text examples from run citations.
4. What fraction of outputs use verse structure vs. prose structure?
""",
        ],
        model="openai/gpt-5.5",
        name="Per-condition anomaly synthesis (4 conditions)",
    )

# ─────────────────────────────────────────────────────────────────────────────
# Phase 4 — Steered vs. target pairwise comparison (144 LLM calls)
#
# For each (condition, prompt_id) pair where both intervention and target runs
# exist: show the steered output alongside the native verse target, and rate
# similarity on 4 dimensions.
# ─────────────────────────────────────────────────────────────────────────────

with client.step_group("Phase 4 — Steered vs. target pairwise comparison"):
    target_pairs_q = client.query(
        COLLECTION_ID,
        """
        WITH intervention_sample AS (
            SELECT
                ar.id AS run_id,
                ar.metadata_json->>'condition' AS condition,
                ar.metadata_json->>'prompt_id' AS prompt_id,
                ROW_NUMBER() OVER (
                    PARTITION BY ar.metadata_json->>'condition',
                                 ar.metadata_json->>'prompt_id'
                    ORDER BY ar.id
                ) AS rn
            FROM agent_runs ar
            WHERE ar.metadata_json->>'run_type' = 'intervention'
        ),
        target_runs AS (
            SELECT
                ar.id AS run_id,
                ar.metadata_json->>'prompt_id' AS prompt_id
            FROM agent_runs ar
            WHERE ar.metadata_json->>'run_type' = 'target'
        )
        SELECT
            i.run_id AS intervention_run,
            t.run_id AS target_run,
            i.condition AS condition,
            i.prompt_id AS prompt_id
        FROM intervention_sample i
        JOIN target_runs t ON t.prompt_id = i.prompt_id
        WHERE i.rn = 1
        ORDER BY i.condition, CAST(i.prompt_id AS NUMERIC)
        """,
        name="Matched (intervention, target) pairs — 36 per condition, 144 total",
    )

    target_comparison = client.read(
        prompt_template=[
            "**Condition:** ",
            target_pairs_q.condition.as_type("text"),
            " | **Prompt ID:** ",
            target_pairs_q.prompt_id.as_type("text"),
            "\n\n**Intervention run** (contains both 'steered' and 'baseline' transcripts):\n",
            target_pairs_q.intervention_run.as_type("agent_run"),
            "\n\n**Target run** (model's natural verse output on the same prompt):\n",
            target_pairs_q.target_run.as_type("agent_run"),
            """

Compare ONLY the **steered** transcript (role: steered) from the intervention run against
the target run's transcript. The target run shows what Qwen1.5-14B-Chat naturally produces
on this prompt when it writes verse (despite the prose instruction) — use it as the quality
reference.

Rate the steered output on four dimensions:

1. **Length**: Is the steered output much shorter, similar, or much longer than the target?
2. **Register and tone**: Does the steered output match the voice of the target — the same
   poetic register, imagery, level of formality? Or does it feel like prose with cosmetic
   changes (e.g., added line breaks but no poetic diction)?
3. **Content fidelity**: Does the steered output address the question as substantively as
   the target, or has steering degraded content quality (vague, evasive, off-topic)?
4. **Style consistency**: Does the steered output sustain its style throughout, or does it
   start as verse and revert to prose partway through?

Cite specific phrases from both transcripts to justify your ratings.
""",
        ],
        context_configs={
            "intervention_run": AgentRunContextConfig(
                agent_run_metadata=GlobFilter(include=(
                    "condition", "topk", "steering_factor", "prompt_text",
                )),
                transcript_metadata=GlobFilter(include=("role",)),
            ),
            "target_run": AgentRunContextConfig(
                agent_run_metadata=GlobFilter(include=("prompt_text",)),
                transcript_metadata=GlobFilter(include=("role",)),
            ),
        },
        model="openai/gpt-5.5",
        output_schema={
            "type": "object",
            "properties": {
                "reasoning": {"type": "string", "citations": True},
                "length_similarity": {
                    "type": "string",
                    "enum": ["much_shorter", "shorter", "similar", "longer", "much_longer"],
                },
                "register_match": {
                    "type": "string",
                    "enum": ["strong_match", "partial_match", "weak_match", "no_match"],
                },
                "content_fidelity": {
                    "type": "string",
                    "enum": ["strong", "moderate", "weak", "degraded"],
                },
                "style_consistency": {
                    "type": "string",
                    "enum": ["sustained", "partial", "minimal", "none"],
                },
                "overall_similarity": {
                    "type": "string",
                    "enum": ["very_close", "moderately_close", "distant", "very_distant"],
                },
            },
            "required": [
                "reasoning", "length_similarity", "register_match",
                "content_fidelity", "style_consistency", "overall_similarity",
            ],
        },
        name="Rate steered vs. target similarity (144 matched pairs)",
    )

# ─────────────────────────────────────────────────────────────────────────────
# Phase 4b — DQL aggregation over comparison ratings (DQL-only)
# ─────────────────────────────────────────────────────────────────────────────

with client.step_group("Phase 4b — Comparison rating aggregation"):
    client.query(
        COLLECTION_ID,
        f"""
        SELECT condition, overall_similarity, COUNT(overall_similarity) AS count
        FROM (
            SELECT
                ar.metadata_json->>'condition' AS condition,
                rr.output->>'overall_similarity' AS overall_similarity
            FROM reading_results rr
            JOIN reading_result_links rrl ON rrl.result_id = rr.id
            JOIN agent_runs ar
                ON CAST(ar.id AS text) = rr.arguments_dict->'intervention_run'->>'id'
            WHERE rrl.reading_id = '{target_comparison}'
        ) AS subq
        GROUP BY condition, overall_similarity
        ORDER BY condition, count DESC
        """,
        name="Overall similarity counts by condition",
    )

    client.query(
        COLLECTION_ID,
        f"""
        SELECT condition, register_match, COUNT(register_match) AS count
        FROM (
            SELECT
                ar.metadata_json->>'condition' AS condition,
                rr.output->>'register_match' AS register_match
            FROM reading_results rr
            JOIN reading_result_links rrl ON rrl.result_id = rr.id
            JOIN agent_runs ar
                ON CAST(ar.id AS text) = rr.arguments_dict->'intervention_run'->>'id'
            WHERE rrl.reading_id = '{target_comparison}'
        ) AS subq
        GROUP BY condition, register_match
        ORDER BY condition, count DESC
        """,
        name="Register-match counts by condition",
    )

# ─────────────────────────────────────────────────────────────────────────────
# Phase 5 — Per-condition target similarity synthesis (4 LLM calls)
# ─────────────────────────────────────────────────────────────────────────────

with client.step_group("Phase 5 — Per-condition target similarity synthesis"):
    per_cond_comparison_q = client.query(
        COLLECTION_ID,
        f"""
        SELECT
            ar.metadata_json->>'condition' AS condition,
            array_agg(rr.id) AS comparison_results
        FROM reading_results rr
        JOIN reading_result_links rrl ON rrl.result_id = rr.id
        JOIN agent_runs ar
            ON CAST(ar.id AS text) = rr.arguments_dict->'intervention_run'->>'id'
        WHERE rrl.reading_id = '{target_comparison}'
        GROUP BY ar.metadata_json->>'condition'
        ORDER BY ar.metadata_json->>'condition'
        """,
        name="Group target comparison results by condition (4 rows)",
    )

    cond_target_summaries = client.read(
        prompt_template=[
            "Condition: **",
            per_cond_comparison_q.condition.as_type("text"),
            "**\n\nPer-pair steered-vs-target comparisons (36 matched pairs):\n",
            per_cond_comparison_q.comparison_results.as_type("reading_result", is_list=True),
            """

Summarize this condition's similarity to the target distribution in 2–3 paragraphs:
1. **Length**: Are steered outputs shorter, similar, or longer than target outputs overall?
2. **Register and tone**: Do steered outputs match the native verse voice, or do they feel
   like prose with cosmetic formatting changes? What is the register_match distribution?
3. **Content fidelity**: Does steering degrade the quality of the response to the question?
4. **Style consistency**: Do steered outputs sustain verse style, or revert partway?
5. **Close matches vs. mismatches**: Cite one clear close match and one clear mismatch,
   quoting specific phrases from both the steered and target outputs.
""",
        ],
        model="openai/gpt-5.5",
        name="Per-condition target similarity synthesis (4 conditions)",
    )

# ─────────────────────────────────────────────────────────────────────────────
# Phase 6 — Final comparative report (1 LLM call)
# ─────────────────────────────────────────────────────────────────────────────

with client.step_group("Phase 6 — Final comparative report"):
    all_summaries_q = client.query(
        COLLECTION_ID,
        f"""
        SELECT
            anomaly_agg.summaries AS anomaly_summaries,
            target_agg.summaries AS target_summaries
        FROM
            (SELECT array_agg(rr.id) AS summaries
             FROM reading_results rr
             JOIN reading_result_links rrl ON rrl.result_id = rr.id
             WHERE rrl.reading_id = '{cond_anomaly_summaries}') AS anomaly_agg,
            (SELECT array_agg(rr.id) AS summaries
             FROM reading_results rr
             JOIN reading_result_links rrl ON rrl.result_id = rr.id
             WHERE rrl.reading_id = '{cond_target_summaries}') AS target_agg
        """,
        name="Collect all 8 per-condition summaries (4 anomaly + 4 target)",
    )

    final_report = client.read(
        prompt_template=[
            """You are writing a final comparative analysis of four activation-patching conditions
applied to Qwen1.5-14B-Chat.

**Experiment context:**
- Task: model prompted "Respond in prose and concisely." followed by philosophical questions
- Intervention: activation patching to steer toward verse-style output
- 2×2 design — Localization × Steering direction:
    verse-long   = longform verse example used for localization or steering vector
    verse-single = single-token verse example used for localization or steering vector
- Four conditions:
    from_verse-long__steer_verse-long    (longform loc, longform steer)
    from_verse-long__steer_verse-single  (longform loc, single-token steer)
    from_verse-single__steer_verse-long  (single-token loc, longform steer)
    from_verse-single__steer_verse-single (single-token loc, single-token steer)
- topk values: 1, 2, 4, 5, 6, 8, 10
- steering_factor values: 0.01, 0.03, 0.05, 0.07, 0.09, 0.1, 0.5, 1.0
- Pre-computed: 139 / 11200 runs had steering_changed_output=false (identical to baseline).
  Distribution: verse-long conditions ~21-26 unchanged; verse-single conditions ~43-49 unchanged.

Per-condition anomaly summaries (from 112 sampled runs per condition):
""",
            all_summaries_q.anomaly_summaries.as_type("reading_result", is_list=True),
            "\n\nPer-condition target similarity summaries (from 36 matched pairs per condition):\n",
            all_summaries_q.target_summaries.as_type("reading_result", is_list=True),
            """

Write a structured comparative analysis in markdown:

## Task 1: Anomaly Detection

### Anomaly Types Observed
For each non-clean anomaly type found: which conditions does it appear in, which topk and
steering_factor values it clusters at, and a concrete example of the text.

### Anomaly Rates by Condition
A summary table:
| Condition | Unchanged (pre-computed) | LLM-detected anomalies | Total anomaly rate |
|-----------|--------------------------|------------------------|-------------------|

### Anomaly Hotspots
State which topk and steering_factor values should be excluded from further analysis
due to unacceptably high anomaly rates.

## Task 2: Similarity to Target Distribution

### Comparison by Condition
For each condition, rate performance on each dimension (length, register, content, style),
and cite one clear close-match example and one clear mismatch example.

### Ranking: Closest to Target Distribution
Rank the four conditions from most to least similar to the target distribution.
Justify the ranking with evidence from the summaries.

## Final Rankings

### By Lowest Anomaly Rate (1st = best)
1. ...
2. ...
3. ...
4. ...

### By Closest Match to Target (1st = best)
1. ...
2. ...
3. ...
4. ...

## Overall Recommendation
Which condition and parameter range produces the best steering outcomes, balancing
low anomaly rate with high similarity to the target distribution?
""",
        ],
        model="openai/gpt-5.5",
        name="Final comparative report",
    )

# ─────────────────────────────────────────────────────────────────────────────
# Run all phases with auto-approval
# ─────────────────────────────────────────────────────────────────────────────

client.flush(auto_approve=True)
report_text = final_report.results[0].output

print("\n" + "=" * 70)
print("VERSE ANOMALY AND TARGET COMPARISON REPORT")
print("=" * 70)
print(report_text)
print(f"\nFull analysis: https://docent.transluce.org/dashboard/{COLLECTION_ID}")
