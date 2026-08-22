"""Which tokens the localization metric scores.

`get_response_logits` sums log P(token) over the assistant response. Position p's
logits predict token p+1, so scoring a response that begins at index `rs`
requires the logits at `rs-1`.

legacy (default; every result in the paper)
    log_probs[rs:-1] gathered against input_ids[rs+1:], which starts one position
    late and therefore NEVER scores the first response token.

    For `-long` data that drops 1 token of ~127 -- immaterial. For `-single` data
    the assistant turn is a single letter, so the dropped token is the entire
    answer: verified on Qwen1.5-14B-Chat and gemma-3-12b-it, the summed span is
    just the two closing tokens (`<|im_end|>`/`<end_of_turn>` and `\\n`). ATP then
    differentiates

        log P(<|im_end|>, \\n | prompt + "D") - log P(<|im_end|>, \\n | prompt + "B")

    i.e. how readily the model closes the turn after each letter. The letter is
    conditioned on but never scored, so nothing rewards preferring the right
    answer. Confirmed by: gradient mass exactly 0 at the answer-predicting
    position, and the returned log-likelihood being bit-identical under +/-100 on
    the answer token's logit.

full
    log_probs[rs-1:-1] gathered against input_ids[rs:] -- scores the whole
    response including its first token.

Default is `legacy` deliberately: flipping it changes every `-single`
localization, and resume in this repo is filename-based, so a mid-stream change
would silently mix two metrics inside one results tree. `full` gets its own
results tree via the path suffix below, so the two can never merge.

Measured effect of switching to `full` (Qwen1.5-14B-Chat, verse, Jaccard of the
selected head set vs `legacy`, k = 0.01 .. 0.1):

    long-form   @ o_proj_out   0.88 .. 0.87     (control: barely moves)
    long-form   @ o_proj_in    0.78 .. 0.89
    single-tok  @ o_proj_out   0.10 .. 0.17     (~85% of the set replaced)
    single-tok  @ o_proj_in    0.00 .. 0.09     (disjoint at k=0.01)

The paper's two structural findings survive the correction: long-form and
single-token stay near-chance disjoint (0.00 .. 0.18 vs 0.005 .. 0.053 chance),
and long-form still localizes earlier than single-token (the gap widens, from
about +0.02..+0.06 to +0.03..+0.09 of the stack).
"""

SPAN_LEGACY = 'legacy'
SPAN_FULL = 'full'
SPANS = (SPAN_LEGACY, SPAN_FULL)

_DIR_SUFFIX = {SPAN_LEGACY: '', SPAN_FULL: '-respfix'}


def check_span(span):
    if span not in SPANS:
        raise ValueError(f"Unknown response_span {span!r}; expected one of {SPANS}")
    return span


def get_span(args):
    return check_span(getattr(args, 'response_span', SPAN_LEGACY))


def dir_suffix(span):
    return _DIR_SUFFIX[check_span(span)]


def response_start_offset(span):
    """How far back from `rs` the scored logit span begins: 0 legacy, 1 full."""
    return 1 if check_span(span) == SPAN_FULL else 0
