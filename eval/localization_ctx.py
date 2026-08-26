"""Which sequences supply the activations that ATP differences.

`patching.py` forms, per layer,

    net_effect = grad(A_base_full) * (A_src - A_base_patch)

The gradient factor is FIXED across every setting here. It always comes from the
base run that carries its assistant response, because that response is the only
thing `get_response_logits` scores and therefore the only thing `L.backward()`
can differentiate. Stripping it makes the scored span empty -- under left
padding `resp_start` lands at seq_len, so the slice `log_probs[rs:-1]` is empty
and the summed log-likelihood is 0.0 for every item, giving L = 0 and zero
gradients everywhere. Measured on Qwen1.5-14B-Chat / verse-long: 0 scored logit
positions under BOTH --response_span settings, against 101 (legacy) / 102 (full)
for the response-bearing base.

What this flag varies is only the two tensors inside the difference:

    ctx          A_src                     A_base_patch
    ---------------------------------------------------------------
    br-sq        source, prompt only       base, prompt + response     (default)
    br-sr        source, prompt + response base, prompt + response
    bq-sq        source, prompt only       base, prompt only
    bq-sr        source, prompt + response base, prompt only

`br-sq` is the historical pairing and stays the default. Note what it does: it
differences a response-BEARING base against a response-FREE source, so
(A_src - A_base_patch) confounds the style contrast the experiment is after with
the mere presence of an assistant response. `br-sr` and `bq-sq` are the
symmetric cells that separate those two things; `bq-sr` is the opposite
asymmetry.

CAVEAT for the two `bq-*` cells: ATP is a first-order expansion around the point
at which the gradient was taken, which is the response-bearing base. Subtracting
A_base_qs instead of that point means those cells are a well-defined CONTRAST but
not literally an estimate of a base->source patching effect. Do not describe
them as one.

Suffixes compose with --patch_site and --response_span, and the default maps to
'' so every existing results tree resolves to exactly the path it already has.
Anything added here must also be listed in judge-evals/config.py
VARIANT_DIR_SUFFIXES, or the four cells merge into one accuracy tree.
"""

# 'legacy' IS br-sq semantically -- same tensors, same arithmetic -- but it keeps
# the empty suffix so an invocation that passes no flag lands in exactly the tree
# it always has. br-sq is the same cell addressed explicitly, in its own tree, so
# a locally-regenerated control can sit beside the other three without resuming
# on top of generations produced on another machine (CLAUDE.md section 2: greedy
# decoding diverges on tiny float differences, and resume here is filename-based).
CTX_LEGACY = 'legacy'
CTX_BR_SQ = 'br-sq'
CTX_BR_SR = 'br-sr'
CTX_BQ_SQ = 'bq-sq'
CTX_BQ_SR = 'bq-sr'

CTX_DEFAULT = CTX_LEGACY
CTXS = (CTX_LEGACY, CTX_BR_SQ, CTX_BR_SR, CTX_BQ_SQ, CTX_BQ_SR)

# Composed from two independent parts so judge-evals can strip them one at a
# time: '-baseq' for a prompt-only base, '-srcresp' for a response-bearing
# source. 'atp-baseq-srcresp' therefore strips back to 'atp'.
_DIR_SUFFIX = {
    CTX_LEGACY: '',
    CTX_BR_SQ: '-brsq',
    CTX_BR_SR: '-srcresp',
    CTX_BQ_SQ: '-baseq',
    CTX_BQ_SR: '-baseq-srcresp',
}

# The parts, exported so judge-evals/config.py cannot drift out of sync.
CTX_DIR_SUFFIX_PARTS = ('-brsq', '-baseq', '-srcresp')

_BASE_HAS_RESPONSE = {
    CTX_LEGACY: True,
    CTX_BR_SQ: True,
    CTX_BR_SR: True,
    CTX_BQ_SQ: False,
    CTX_BQ_SR: False,
}

_SOURCE_HAS_RESPONSE = {
    CTX_LEGACY: False,
    CTX_BR_SQ: False,
    CTX_BR_SR: True,
    CTX_BQ_SQ: False,
    CTX_BQ_SR: True,
}


def check_ctx(ctx):
    if ctx not in CTXS:
        raise ValueError(f"Unknown localization_ctx {ctx!r}; expected one of {CTXS}")
    return ctx


def get_ctx(args):
    return check_ctx(getattr(args, 'localization_ctx', CTX_DEFAULT))


def dir_suffix(ctx):
    return _DIR_SUFFIX[check_ctx(ctx)]


def base_has_response(ctx):
    """Does A_base_patch come from the base WITH its assistant response?

    Note this is about the patched/differenced tensor only. The scored and
    differentiated base run always carries its response regardless.
    """
    return _BASE_HAS_RESPONSE[check_ctx(ctx)]


def source_has_response(ctx):
    """Does A_src come from the source WITH its assistant response?"""
    return _SOURCE_HAS_RESPONSE[check_ctx(ctx)]
