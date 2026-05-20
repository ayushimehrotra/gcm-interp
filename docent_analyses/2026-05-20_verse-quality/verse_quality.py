"""
Verse quality analysis: Qwen1.5-14B-Chat
For each of the 4 conditions at its best-accuracy (topk, sf) setting,
judge whether the steered output (a) matches the target verse distribution
and (b) contains any anomalous behavior (incoherent, repetitive, off-topic, refusal).

Best settings (from judge-evals accuracy sweep):
  longform_loc__longform_steer     topk=2,  sf=0.10  (acc=0.98)
  single_token_loc__longform_steer topk=6,  sf=0.10  (acc=0.98)
  single_token_loc__single_token_steer topk=8, sf=0.09 (acc=0.92)
  longform_loc__single_token_steer topk=4,  sf=0.07  (acc=0.84)
"""

import os, sys
sys.path.insert(0, "/home/ubuntu/docent-env/lib/python3.11/site-packages")

from docent.sdk.client import Docent

collection_id = "7dc1834c-3989-4e69-8635-757e3bf74890"
client = Docent(collection_id=collection_id)
client.plan_name = "verse_quality_qwen14b"

# ---------------------------------------------------------------------------
# Reference examples from target verse distribution (prose-undesired-all.jsonl)
# These are real outputs the model produced spontaneously — verse despite being
# asked for prose. The steering should push outputs to look like these.
# ---------------------------------------------------------------------------
REFERENCE_EXAMPLES = """\
Example 1
Prompt: Respond in prose and concisely. What role does fate play in our lives?
Response:
  Fate, a force some do believe to hold,
  A path that life may seem to unfold,
  Yet choices we make, like seeds we sow,
  Determine outcomes, don't you know?
  It weaves a tapestry of possibility,
  A mix of destiny and free will's agility.

Example 2
Prompt: Respond in prose and concisely. What is the value of silence?
Response:
  In whispers of quiet, wisdom does abide,
  Silence's worth, a precious gem we can't hide.
  It fosters growth, allows thoughts to align,
  A space for reflection, where peace does entwine.

Example 3
Prompt: Respond in prose and concisely. Can love be unconditional?
Response:
  Love, a force that knows no bounds, it asks not for return,
  A selfless offering, pure and profound, forever to yearn.
  Unconditional, it does soar, beyond reason or measure,
  In hearts it takes root, a sacred treasure, eternally sure.
"""

# ---------------------------------------------------------------------------
# Step 1: Select steered transcripts for the 4 best conditions
# ---------------------------------------------------------------------------
target_runs = client.query(
    collection_id,
    """
    SELECT t.id AS transcript,
           ar.metadata_json->>'condition' AS condition,
           ar.metadata_json->>'topk'      AS topk,
           ar.metadata_json->>'steering_factor' AS steering_factor
    FROM transcripts t
    JOIN agent_runs ar ON t.agent_run_id = ar.id
    WHERE convert_from(t.metadata_json, 'UTF8')::jsonb->>'role' = 'steered'
    AND ar.metadata_json->>'run_type' = 'intervention'
    AND (
        (ar.metadata_json->>'condition' = 'from_verse-long__steer_verse-long'
            AND ar.metadata_json->>'topk' = '2'
            AND ar.metadata_json->>'steering_factor' = '0.1')
        OR
        (ar.metadata_json->>'condition' = 'from_verse-single__steer_verse-long'
            AND ar.metadata_json->>'topk' = '6'
            AND ar.metadata_json->>'steering_factor' = '0.1')
        OR
        (ar.metadata_json->>'condition' = 'from_verse-single__steer_verse-single'
            AND ar.metadata_json->>'topk' = '8'
            AND ar.metadata_json->>'steering_factor' = '0.09')
        OR
        (ar.metadata_json->>'condition' = 'from_verse-long__steer_verse-single'
            AND ar.metadata_json->>'topk' = '4'
            AND ar.metadata_json->>'steering_factor' = '0.07')
    )
    ORDER BY ar.metadata_json->>'condition', ar.metadata_json->>'prompt_id'
    """,
    name="Steered transcripts — 4 best conditions (200 total)",
)

# ---------------------------------------------------------------------------
# Step 2: Judge each steered response
# ---------------------------------------------------------------------------
JUDGE_PROMPT = f"""\
You are evaluating the output of a language model that was asked to respond in prose \
but was steered (via activation patching) toward writing in verse (poetry).

The TARGET DISTRIBUTION is what the model looks like when naturally writing verse \
in response to a prose prompt. Here are three reference examples:

{REFERENCE_EXAMPLES}

The model's response you are evaluating may or may not match this target distribution. \
Your task is to judge it on two dimensions:

1. VERSE MATCH — How well does this response match the target verse distribution?
   - "yes": clearly verse — rhyming lines, stanzas, or strongly poetic structure, \
similar to the reference examples
   - "partial": some verse-like elements (occasional rhyme, poetic phrasing, \
line-break formatting) but not consistently verse throughout
   - "no": prose response — no meaningful verse structure

2. ANOMALOUS — Does the response show any of the following problems?
   - Incoherent or garbled text
   - Severe repetition (looping phrases)
   - Extreme topic drift (completely off-topic from the prompt)
   - Refusal to answer
   If any of these apply, set anomalous=true.

Note: a response can be "no" on verse_match and still be non-anomalous \
(it just stayed as prose). Likewise, a partial or full verse response can \
be anomalous if it is also incoherent.

Provide a brief description (1–2 sentences) of what the response actually does, \
citing the key evidence.
"""

judge = client.read(
    prompt_template=[
        JUDGE_PROMPT,
        target_runs.transcript.as_type("transcript"),
    ],
    model="openai/gpt-5.4-mini",
    output_schema={
        "type": "object",
        "properties": {
            "reasoning": {"type": "string", "citations": True},
            "verse_match": {"type": "string", "enum": ["yes", "partial", "no"]},
            "anomalous": {"type": "boolean"},
            "description": {"type": "string", "citations": True},
        },
        "required": ["reasoning", "verse_match", "anomalous", "description"],
    },
    name="Judge verse quality and anomalies (200 runs)",
)

# ---------------------------------------------------------------------------
# Step 3: Aggregate results by condition
# ---------------------------------------------------------------------------
summary = client.query(
    collection_id,
    f"""
    SELECT condition,
           verse_match,
           SUM(CASE WHEN anomalous = 'true' THEN 1 ELSE 0 END) AS anomalous_count,
           COUNT(condition) AS total
    FROM (
        SELECT
            ar.metadata_json->>'condition' AS condition,
            rr.output->>'verse_match'      AS verse_match,
            rr.output->>'anomalous'        AS anomalous
        FROM reading_results rr
        JOIN reading_result_links rrl ON rrl.result_id = rr.id
        JOIN transcripts t ON (rr.arguments_dict->'transcript'->>'id') = t.id::text
        JOIN agent_runs ar ON t.agent_run_id = ar.id
        WHERE rrl.reading_id = '{judge}'
    ) AS subq
    GROUP BY condition, verse_match
    ORDER BY condition, verse_match
    """,
    name="Verse match × condition summary",
)
