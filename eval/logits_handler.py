import os
import glob
import json
import torch
import einops
import pandas as pd
import matplotlib.pyplot as plt
import random

# ---------------------------------------------------------------------------
# Random-head control arms
#
# The arm and the head-draw seed are both encoded in --patch_algo:
#   random-s0, random-s1, ...           uniform random over all (layer, head)
#   randomlayer-s0, randomlayer-s1, ... random, per-layer counts matched to ATP
# set_output_prefix() interpolates patch_algo into the results path, so every
# arm/seed lands in its own tree and no two draws can overwrite each other.
#
# The draw seed is deliberately separate from --seed, which feeds set_seed() and
# also controls generation: varying that would change the generations as well as
# the head draw and confound the comparison.
# ---------------------------------------------------------------------------

def is_random(algo):
    """True for both control arms."""
    return algo.startswith('random')

def is_layer_matched(algo):
    """True only for the layer-matched arm."""
    return algo.startswith('randomlayer')

def draw_seed(algo):
    """'random-s3' -> 3. Bare 'random' keeps the historical fixed draw (42)."""
    if '-s' not in algo:
        return 42
    return int(algo.split('-s')[-1])

def load_logits(config, data_handler, which_patch, model_handler):
    logits_path = f"{'/'.join(config.get_output_prefix().split('/')[:-3])}/{which_patch}"
    # print('Loading logits from:', logits_path)
    all_logits = None

    name = 'numerator_1' if config.args.patch_algo != 'probes' else 'probes'
    print('/'.join(config.get_output_prefix().split('/')[:-3]))
    if os.path.exists(f"{'/'.join(config.get_output_prefix().split('/')[:-3])}/{name}_{which_patch}.pt"):
        # print(f"Loading precomputed logits for {name} from {config.get_output_prefix()}/{name}_{which_patch}.pt")
        all_logits = torch.load(f"{'/'.join(config.get_output_prefix().split('/')[:-3])}/{name}_{which_patch}.pt")
    else:
        print('Path does not exist {}, computing logits afresh.'.format(f"{'/'.join(config.get_output_prefix().split('/')[:-3])}/{name}_{which_patch}.pt"))
        if config.args.patch_algo != 'probes':
            for i in range(data_handler.LEN):
                try:
                    with open(f"{logits_path}_{i}.pt", 'rb') as f:
                        logits = torch.load(f)
                        # print('atp-zero', logits.shape)
                        logits = logits.squeeze().unsqueeze(-1) if 'atp' in config.args.patch_algo else logits
                        all_logits = logits if all_logits is None else torch.cat([all_logits, logits], dim=-1)
                except Exception as e:
                    print(f"Could not load logits from {logits_path}_{i}.pt, skipping this file. ", e)
                    continue

            # print('patcher ', all_logits.shape)
            name = 'numerator_1'
            if 'atp' in config.args.patch_algo:
                # print('all_logits ', all_logits.shape)
                # all_logits = all_logits.sum(dim=1)
                all_logits = einops.reduce(all_logits, 'l (n m) b -> l n b', 'sum', n=model_handler.num_heads)
            if config.args.patch_algo == 'acp':
                # print('all_logits before squeeze ', all_logits.shape)
                all_logits = all_logits.squeeze()
                # print('all_logits after squeeze ', all_logits.shape)
                base_des_post_patch = all_logits[0,...]
                base_undes_post_patch = all_logits[1,...]
                all_logits = (base_undes_post_patch - base_des_post_patch)
                # print('all_logits final ', all_logits.shape)
        else:
            name = 'probes'
            with open(f'{logits_path}.json', 'r') as f:
                raw_logits = json.load(f)
            logits = [[float(head_val) for head_val in layer_dict.values()] for layer_dict in raw_logits.values()]
            all_logits = torch.tensor(logits)
        plot_logit_metrics(config, model_handler, all_logits, name, which_patch)
        torch.save(all_logits, f"{'/'.join(config.get_output_prefix().split('/')[:-3])}/{name}_{which_patch}.pt")
    return all_logits

def get_top_k_layer_and_head(patches, top_k, patch_algo):
    if isinstance(patches, str):
        patches = torch.load(patches)
    patches = patches.to(torch.float32)
    if patch_algo != 'probes':
        patches = patches.mean(dim=-1)
    flat = patches.view(-1)
    top_values, top_indices = flat.topk(k=int(top_k * flat.numel()))
    layer_indices = top_indices // patches.shape[1]
    neuron_indices = top_indices % patches.shape[1]
    df = pd.DataFrame({
        'layer': layer_indices.numpy(),
        'neuron': neuron_indices.numpy(),
        'value': top_values.numpy()
    })
    return df.sort_values(by=['layer', 'neuron'])

def retrieve_random_k(num_layers, num_heads, k, seed=42):
    rng = random.Random(seed)
    total = num_layers * num_heads
    num_samples = int(k * total)
    all_combinations = [(l, h) for l in range(num_layers) for h in range(num_heads)]
    selected = rng.sample(all_combinations, num_samples)
    df = pd.DataFrame(selected, columns=['layer', 'neuron'])
    return df.sort_values(by=['layer', 'neuron'])

def atp_reference_csv(config, topk):
    """Path to the real ATP selection for this (model, source, base, topk).

    CLAUDE.md assumes the head ranking is identical across eval/steer
    subdirectories within a localization, so that any of them can serve as the
    reference. That is false in this repo: for Qwen1.5-14B-Chat's `-single`
    localizations the committed ATP CSVs split cleanly by *steer* subdirectory
    (verse-single_to_prose and paragraph-single_to_sentence disagree at every
    topk < 1.0, with only ~2/16 heads shared at k=0.01). Picking an arbitrary
    subdirectory would therefore match the layer profile of a *different* ATP
    condition than the one the arm is being compared against.

    So resolve the reference from the identical {eval}_eval/{steer}_steer
    subdirectory as the current run, which is unambiguous. Only fall back to
    another subdirectory if that exact one is absent, and say so loudly.
    """
    prefix = config.get_output_prefix().rstrip('/')
    # .../results/{model}/from_{source}_to_{base}/{patch_algo}/{eval}_eval/{steer}_steer
    parts = prefix.split('/')
    task_root, eval_dir, steer_dir = '/'.join(parts[:-3]), parts[-2], parts[-1]

    exact = f"{task_root}/atp/{eval_dir}/{steer_dir}/eval/numerator_1_targeted_{topk}.csv"
    if os.path.exists(exact):
        return exact, pd.read_csv(exact)[['layer', 'neuron']]

    pattern = f"{task_root}/atp/*/*/eval/numerator_1_targeted_{topk}.csv"
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"Layer-matched random needs the real ATP selection but found neither "
            f"{exact} nor any file matching {pattern}. Run the atp arm for this "
            f"(model, source, base, topk) first. Refusing to fall back to uniform random."
        )
    distinct = {frozenset(map(tuple, pd.read_csv(m)[['layer', 'neuron']].values))
                for m in matches}
    if len(distinct) > 1:
        raise ValueError(
            f"No ATP reference at {exact}, and the {len(matches)} fallback candidates "
            f"under {task_root}/atp hold {len(distinct)} different head sets, so the "
            f"layer profile to match is ambiguous. Candidates: {matches}"
        )
    print(f"WARNING: no ATP reference at {exact}; falling back to {matches[0]} "
          f"({len(matches)} candidates, all with the same head set).")
    return matches[0], pd.read_csv(matches[0])[['layer', 'neuron']]

def retrieve_layer_matched_k(config, topk, num_layers, num_heads, seed):
    """Random heads whose per-layer counts exactly match the real ATP selection
    for this (model, source, base, topk)."""
    path, reference = atp_reference_csv(config, topk)
    hist = reference.groupby('layer').size().to_dict()
    print(f"Layer-matched random: matching per-layer histogram from {path} "
          f"({len(reference)} heads across {len(hist)} layers)")

    rng = random.Random(seed)
    selected = []
    for layer in sorted(hist):
        n = hist[layer]
        if layer >= num_layers:
            raise ValueError(f"Reference CSV {path} has layer {layer} >= num_layers {num_layers}")
        if n > num_heads:
            raise ValueError(
                f"Reference CSV {path} selects {n} heads in layer {layer}, but the model "
                f"has only {num_heads} heads per layer. Sampling without replacement is "
                f"impossible; check that num_attention_heads is the query-head count."
            )
        # Sample from ALL heads in the layer, including the genuinely-selected ones:
        # the null asks whether this particular set is special among sets with the
        # same layer profile, so excluding the real heads would bias it.
        selected += [(layer, h) for h in rng.sample(range(num_heads), n)]

    df = pd.DataFrame(selected, columns=['layer', 'neuron'])
    assert df.groupby('layer').size().to_dict() == hist, \
        "layer-matched draw does not reproduce the reference per-layer histogram"
    assert len(df) == len(reference), \
        f"layer-matched draw has {len(df)} heads, reference has {len(reference)}"
    return df.sort_values(by=['layer', 'neuron'])

def plot_logit_metrics(config, model_handler, metric, name, which_patch):
    metric = metric.to(torch.float32)
    if config.args.patch_algo != 'probes':
        metric = metric.mean(dim=-1)

    plt.imshow(metric, cmap="viridis")
    plt.colorbar(label='Indirect Effect size')
    plt.ylabel("Layers")
    plt.xlabel("Heads")
    plt.grid(True)
    titles = {
        "numerator_1": f"Post-patch logit difference: {config.args.base}",
        "probes": f"Probes accuracy between desired and undesired responses {config.args.base}"
    }
    plt.title(titles.get(name, name))
    plt.xticks(ticks=range(model_handler.num_heads))
    plt.yticks(ticks=range(model_handler.model.config.num_hidden_layers))
    plt.tight_layout()
    os.makedirs(f'{config.get_output_prefix()}/eval/', exist_ok=True)
    plt.savefig(f"{config.get_output_prefix()}/eval/{name}_heatmap.png")
    plt.close()
