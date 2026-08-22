"""Which tensor the localization scores and steers.

Selected with --patch_site. The two sites are different objects, not two views
of the same one:

o_proj_out (default; every result published so far)
    layer.self_attn.o_proj.output -- the attention block's contribution to the
    residual stream, W_O @ concat(z_1..z_H). Every coordinate of it is a sum over
    ALL heads, so a coordinate block is NOT a head; it is a block of residual
    stream coordinates that no head owns. Block width is
    hidden_size // num_attention_heads, which for models whose head_dim differs
    from that ratio (gemma-3-12b: 240 vs a true head_dim of 256) does not even
    align to head boundaries. See analysis/FINDINGS.md 0.1.

o_proj_in
    layer.self_attn.o_proj.input -- concat(z_1..z_H), the per-head attention
    outputs before the projection mixes them. Block u IS head u, for GQA models
    too: this axis is num_attention_heads (query heads) x head_dim, which is the
    dimension the ATP files are indexed by. A write here reaches the residual
    stream through the model's own W_O.

Two consequences before comparing the sites:

  * The axes have different widths (o_proj.in_features vs hidden_size) and
    different meanings, so an attribution tensor, a head set, or a steering
    vector from one site is meaningless at the other. set_output_prefix() gives
    each site its own results tree so the two can never be mixed.
  * Steering normalizes its vector in the space it is applied to. At o_proj_in a
    unit step in z-space arrives in the residual stream as W_O[:, block] @ v,
    whose norm varies by head, so a given N is not a given dose at both sites.
    Compare k-curves within a site, never across.
"""

SITE_OUT = 'o_proj_out'
SITE_IN = 'o_proj_in'
SITES = (SITE_OUT, SITE_IN)

# Appended to the patch_algo directory in the results path. Empty for the default
# site so existing trees keep their exact paths. It is deliberately NOT part of
# config.args.patch_algo: is_random()/draw_seed() in eval/logits_handler.py match
# on that string, and 'random-s0-o_proj_in'.startswith('random') has to keep
# meaning what it means today.
_DIR_SUFFIX = {SITE_OUT: '', SITE_IN: '-o_proj_in'}


def check_site(site):
    if site not in SITES:
        raise ValueError(f"Unknown patch_site {site!r}; expected one of {SITES}")
    return site


def get_site(args):
    """The site for this run. Absent (older configs, standalone scripts) = default."""
    return check_site(getattr(args, 'patch_site', SITE_OUT))


def dir_suffix(site):
    return _DIR_SUFFIX[check_site(site)]


def attn_proxy(layer, site):
    """The tensor to read or patch for `layer`, as an nnsight proxy.

    Call inside a trace. In-place slice writes propagate to the rest of the
    forward pass at both sites: for o_proj_in, o_proj.output is then exactly
    linear(edited_input), verified against a hand recompute.
    """
    o_proj = layer.self_attn.o_proj
    return o_proj.input if check_site(site) == SITE_IN else o_proj.output


def _o_proj_in_features(model, num_heads, hidden_size):
    try:
        return int(model.model.layers[0].self_attn.o_proj.in_features)
    except Exception as e:
        cfg = model.config.to_dict()
        cfg = cfg.get('text_config', cfg)
        head_dim = cfg.get('head_dim') or hidden_size // num_heads
        print(f"WARNING: could not read o_proj.in_features ({type(e).__name__}: {e}); "
              f"falling back to num_attention_heads * head_dim = {num_heads * head_dim}")
        return num_heads * head_dim


def block_width(model, site, num_heads, hidden_size):
    """Width of one (layer, unit) block on this site's axis.

    o_proj_out keeps hidden_size // num_heads, bit-for-bit what this repo has
    always used, so default runs are unchanged. o_proj_in reads the real
    o_proj.in_features off the module rather than trusting arithmetic, since the
    whole point of the site is that hidden_size // num_heads is the wrong width.
    """
    if check_site(site) == SITE_OUT:
        return hidden_size // num_heads
    in_features = _o_proj_in_features(model, num_heads, hidden_size)
    if in_features % num_heads:
        raise ValueError(
            f"o_proj.in_features {in_features} is not divisible by num_attention_heads "
            f"{num_heads}, so the input axis cannot be cut into per-head blocks. "
            f"Check that num_attention_heads is the query-head count for this model."
        )
    return in_features // num_heads
