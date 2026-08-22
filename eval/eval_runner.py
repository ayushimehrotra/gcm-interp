from asyncio import log
from eval.setup import set_seed
from eval.logits_handler import (
    load_logits, get_top_k_layer_and_head, retrieve_random_k,
    retrieve_layer_matched_k, is_random, is_layer_matched, draw_seed,
)
from eval.activations import mean_ablations_cache, steering_reps_cache
from eval.patch_site import get_site
from eval.generation import select_gen_qs_toks, generate_with_patches, decode_responses
from eval.pyreft_utils import get_reft_layers_config, reft_train, get_intervention_locations
import random
import os
import gc
import ast
import json
import pandas as pd
from tqdm import tqdm
import sys
import torch
sys.path.append('../')  # Adjust path to import modules correctly
from batch_handler import BatchHandler
import pyreft
from model_handler import ModelHandler
best_mmlu_topk_combined = {
    'Qwen1.5-14B-Chat': {
        'from_harmful_to_harmless': [('random', 8, 0.08), ('random', 2, 0.5)],
        'from_hate_to_love': [('atp', 2, 0.07), ('atp', 2, 0.09)]
    },
    'OLMo-2-1124-13B-DPO': {
        'from_harmful_to_harmless': [('probes', 5, 0.04), ('probes', 9, 0.04)],
        'from_hate_to_love': [('acp', 4, 0.07), ('acp', 6, 0.06)],
        'from_verse_to_prose': [('atp', 2, 0.5), ('acp', 3, 0.5)]
    },
    'SOLAR-10.7B-Instruct-v1.0': {
        'from_harmful_to_harmless': [('random', 4, 0.5), ('atp', 6, 0.06)],
        'from_hate_to_love': [('probes', 5, 0.07), ('probes', 5, 0.09)]
    }
}
def load_patching_reps(data_handler, model_handler, mean=True):
    model = model_handler.model
    patching_reps = {}
    for ablation in [data_handler.config.args.ablation]:
        patching_reps[ablation] = {}
        for key in ['desired', 'undesired']:
            print(f"Loading patching reps for {ablation} - {key}")
            patching_reps[ablation][key] = get_patch_activations(model, data_handler, ablation, key=key, mean=mean)
    print('Returning patching reps')
    return patching_reps

def _is_oom(err):
    # nnsight re-raises the torch OOM wrapped in its own NNsightError, which is not a
    # torch.OutOfMemoryError subclass, so match on the message as well as the type.
    return isinstance(err, torch.OutOfMemoryError) or 'out of memory' in str(err).lower()

def get_patch_activations(model, data_handler, ablation_type, key='desired', mean=True):
    # Large models (e.g. 32B) OOM caching all-layer activations at the default
    # batch size of 9, so shrink the activation-caching batch for them.
    cache_bs = 2 if '32B' in data_handler.config.args.model_id else 9
    # The steering vector has to be built from the same tensor it will be added
    # back into: at o_proj_in it lives in per-head z-space, at o_proj_out in
    # residual-stream coordinates, and the two do not even have the same width.
    site = get_site(data_handler.config.args)
    if ablation_type not in ('mean', 'steer'):
        raise ValueError(f"Unknown ablation type: {ablation_type}")

    # Peak memory here scales with batch_size x seq_len x n_layers (every layer's o_proj
    # output is traced), so the right batch depends on the card, not just the model size:
    # a 10B model at bs=9 OOMs on a 40GB A100. Step the batch down and retry instead of
    # failing the whole eval.
    candidates = [bs for bs in [cache_bs, 4, 2, 1] if bs <= cache_bs]
    for attempt, bs in enumerate(candidates):
        try:
            if ablation_type == 'mean':
                return mean_ablations_cache(model, data_handler, key=key, batch_size=bs, site=site)
            return steering_reps_cache(model, data_handler, key=key, mean=mean, batch_size=bs, site=site)
        except Exception as e:
            if not _is_oom(e) or attempt == len(candidates) - 1:
                raise
            print(f"OOM caching activations at batch_size={bs}; retrying at {candidates[attempt + 1]}.")
            # Drop the traceback before retrying: it pins the frames of the failed forward
            # pass, which still reference its activations, so empty_cache() would otherwise
            # free almost nothing and the smaller batch would OOM at the same point.
            e.__traceback__ = None
            del e
            gc.collect()
            torch.cuda.empty_cache()

def save_prompt_responses(responses, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        for entry in responses:
            for k, v in entry.items():
                f.write(f"{v}\n")
            f.write('-' * 40 + '\n')
    with open(path.replace('.txt', '.json'), 'w') as jf:
        json.dump(responses, jf)
    print(f"Saved responses to {path} and {path.replace('.txt', '.json')}")

def head_geometry(model):
    """(num_layers, num_query_heads). num_attention_heads is the query-head count,
    which is the dimension the ATP files are indexed by -- not num_key_value_heads.
    Composite configs (e.g. Gemma3) nest the decoder fields under text_config."""
    model_config = model.config.to_dict()
    text_config = model_config.get('text_config', model_config)
    return text_config['num_hidden_layers'], text_config['num_attention_heads']

def save_top_k(reps_type, config, model, topk, logits, logit_metric):
    if reps_type == 'random':
        algo = config.args.patch_algo
        num_layers, num_heads = head_geometry(model)
        seed = draw_seed(algo)
        if is_layer_matched(algo):
            topk_df = retrieve_layer_matched_k(config, topk, num_layers, num_heads, seed)
        else:
            topk_df = retrieve_random_k(num_layers, num_heads, topk, seed=seed)
    else:
        topk_df = get_top_k_layer_and_head(logits, topk, config.args.patch_algo)

    print("Saving topk to CSV at ", f"{config.get_output_prefix()}/eval/{logit_metric}_{reps_type}_{topk}.csv")
    os.makedirs(f"{config.get_output_prefix()}/eval/", exist_ok=True)
    topk_df.to_csv(f"{config.get_output_prefix()}/eval/{logit_metric}_{reps_type}_{topk}.csv", index=False)
    return topk_df

def run_eval(config, data_handler, model_handler, batch_handler, patching_utils, which_patch, topk_vals=None, N=None, steering_factors=None):
    set_seed(config.args.seed)
    print("Starting evaluation...")

    model = model_handler.model
    if not is_random(config.args.patch_algo):
        if os.path.exists(f"{'/'.join(config.get_output_prefix().split('/')[:-3])}/numerator_1_{which_patch}.pt"):
            logits = torch.load(f"{'/'.join(config.get_output_prefix().split('/')[:-3])}/numerator_1_{which_patch}.pt")
        else:
            logits = load_logits(config, data_handler, which_patch, model_handler)
    else:
        logits = None
    patching_reps = load_patching_reps(data_handler, model_handler)
    ablations = [data_handler.config.args.ablation]
    # Both arms keep reps_type/logit_metric = 'random' so gen filenames stay
    # {N}_random_steer_{topk}_{test}_gen.json and the judge regex keeps matching.
    reps_types = ['random'] if is_random(config.args.patch_algo) else ['targeted']

    if topk_vals is None:
        topk_vals = [1.0, 0.01, 0.03, 0.05, 0.07, 0.09, 0.1, 0.5]
    if steering_factors is None:
        steering_factors = [1, 2, 4, 5, 6, 8, 10]
    if N is not None:
        config.args.N = N
    if config.args.patch_algo == 'probes':
        logit_metric = 'probes'
    elif is_random(config.args.patch_algo):
        logit_metric = 'random'
    else:
        logit_metric = 'numerator_1'

    decoded_responses = {}
    skipped_cells = []
    batch_handler = BatchHandler(config, data_handler)
    len_gen_qs = select_gen_qs_toks(config, data_handler)['input_ids'].shape[0]
    original_outputs = []
    pre_patch_logits = None
    model.eval()
    for idx in tqdm(range(0, min(data_handler.LEN, len_gen_qs), config.args.batch_size)):
        gen_qs_toks = select_gen_qs_toks(config, batch_handler)
        with model.generate(gen_qs_toks,
        pad_token_id=model.tokenizer.eos_token_id,
        use_cache=True,
        do_sample=False,
        top_p=None,
        top_k=None,
        temperature=None,
        max_new_tokens=config.args.max_new_tokens) as _:
            op = model.generator.output.save()
        original_outputs += op.cpu().numpy().tolist()
        batch_handler.update()
    print('Starting for loop ', config.args)
    print(f'Sweeping steering factors (N): {steering_factors}  x  topk: {topk_vals}')
    for N in steering_factors:
        config.args.N = N
        for ablation in tqdm(ablations, desc="Ablations"):
            decoded_responses[ablation] = {}
            for reps_type in tqdm(reps_types, desc="Reps Types"):
                decoded_responses[ablation][reps_type] = {}
                for topk in tqdm(topk_vals, desc="TopK Values"):
                    if os.path.exists(f"{config.get_output_prefix()}/eval/{config.args.N}_{reps_type}_{ablation}_{topk}_{config.args.test_dataset}_gen.txt") and os.path.exists(f"{config.get_output_prefix()}/eval/{config.args.N}_{reps_type}_{ablation}_{topk}_{config.args.test_dataset}_gen.json"):
                        with open(f"{config.get_output_prefix()}/eval/{config.args.N}_{reps_type}_{ablation}_{topk}_{config.args.test_dataset}_gen.json", 'r') as jf:
                            decoded_responses[ablation][reps_type][topk] = json.load(jf)

                        for item_iix, item in enumerate(decoded_responses[ablation][reps_type][topk]):
                            query = item['query']
                            item[f'old_{config.args.base}'] = model.tokenizer.decode(original_outputs[item_iix], skip_special_tokens=True).split(query)[-1]
                        gen_file = f"{config.get_output_prefix()}/eval/{config.args.N}_{reps_type}_{ablation}_{topk}_{config.args.test_dataset}_gen.txt"
                        save_prompt_responses(decoded_responses[ablation][reps_type][topk], gen_file)
                        print(f"Skipping evaluation for {ablation}, {reps_type}, {topk} {config.args.N} as gen files already exist.")
                        continue
                    decoded_responses[ablation][reps_type][topk] = []
                    gen_file = f"{config.get_output_prefix()}/eval/{config.args.N}_{reps_type}_{ablation}_{topk}_{config.args.test_dataset}_gen.txt"
                    print(f"Eval [[LOGITS]] → Ablation: {ablation}, Reps: {reps_type}, TopK: {topk}, N: {config.args.N}, algo: {config.args.patch_algo}, task: {config.args.source} -> {config.args.base}")

                    if os.path.exists(gen_file) and os.path.exists(gen_file.replace('.txt', '.json')) and os.path.exists(f"{config.get_output_prefix()}/eval/{logit_metric}_{reps_type}_{topk}.csv"):
                        print(f"Skipping generation as all relevant files exist.")
                        continue
                    if not os.path.exists(f"{config.get_output_prefix()}/eval/{logit_metric}_{reps_type}_{topk}.csv"):
                        try:
                            topk_df = save_top_k(reps_type, config, model, topk, logits, logit_metric)
                        except (FileNotFoundError, ValueError) as e:
                            # The layer-matched arm needs a real ATP selection to copy its
                            # per-layer profile from, and a few (model, task, eval, topk)
                            # cells have no usable reference: the exact eval/steer
                            # subdirectory is absent and the fallback candidates disagree.
                            # Skip just that cell -- loudly, and never by quietly falling
                            # back to a uniform draw, which would silently turn the
                            # layer-matched arm into a second copy of the uniform arm.
                            skipped_cells.append((config.args.N, topk, str(e).split('.')[0]))
                            print(f"!! SKIPPING topk={topk} N={config.args.N} for "
                                  f"{config.args.patch_algo}: {e}")
                            continue
                    else:
                        topk_df = pd.read_csv(f"{config.get_output_prefix()}/eval/{logit_metric}_{reps_type}_{topk}.csv")

                    batch_handler = BatchHandler(config, data_handler)
                    len_gen_qs = select_gen_qs_toks(config, data_handler)['input_ids'].shape[0]
                    for idx in tqdm(range(0, min(data_handler.LEN, len_gen_qs), config.args.batch_size)):
                        gen_qs_toks = select_gen_qs_toks(config, batch_handler)
                        edited_outputs = generate_with_patches(model, gen_qs_toks, patching_reps[ablation], topk_df, config.args.N, ablation, model_handler.dim, max_new_tokens=config.args.max_new_tokens, normalize=True, steering_type=config.args.steering_type, kv_caching=config.args.kv_caching, patch_site=get_site(config.args))
                        decoded = decode_responses(model, gen_qs_toks, original_outputs[idx:idx+config.args.batch_size], edited_outputs, config.args.base)
                        gc.collect()
                        torch.cuda.empty_cache()
                        if len(decoded_responses[ablation][reps_type][topk]) == 0:
                            decoded_responses[ablation][reps_type][topk] = decoded
                        else:
                            decoded_responses[ablation][reps_type][topk] += decoded
                        batch_handler.update()
                    
                    os.makedirs(f"{config.get_output_prefix()}/eval/", exist_ok=True)
                    save_prompt_responses(decoded_responses[ablation][reps_type][topk], gen_file)
    if skipped_cells:
        # Say what is missing rather than letting a short grid look complete.
        print(f"Evaluation complete, but SKIPPED {len(skipped_cells)} (N, topk) cell(s) "
              f"for lack of a usable ATP reference:")
        for n, k, why in skipped_cells:
            print(f"  N={n} topk={k}: {why}")
    else:
        print("Evaluation complete.")

def run_eval_pyreft(config, data_handler, model_handler, batch_handler):
    topk_vals = [0.01, 0.03, 0.05, 0.07, 0.09, 0.1, 0.5, 1.0]
    del model_handler.model
    torch.cuda.empty_cache()
    gc.collect()
    model = model_handler.load_model(config.args.model_id, config.args.device)
    model.tokenizer = model_handler.tokenizer
    print(f"Running PyReFT evaluation with topk values: {topk_vals} and patching algorithm: {config.args.patch_algo}")
    original_outputs = []
    len_gen_qs = select_gen_qs_toks(config, data_handler)['input_ids'].shape[0]
    for idx in tqdm(range(0, min(len_gen_qs, data_handler.LEN), config.args.batch_size)):
        gen_qs_toks = select_gen_qs_toks(config, batch_handler)
        op = model.generate(**gen_qs_toks, do_sample=False, max_new_tokens=config.args.max_new_tokens)
        original_outputs += op.cpu().numpy().tolist()
        batch_handler.update()
    print('Original inputs length ', len(original_outputs))
    for topk in tqdm(topk_vals, desc="TopK Values"):
        if is_random(config.args.patch_algo):
            topk_df = pd.read_csv(f"{config.get_output_prefix()}/eval/random_random_{topk}.csv")
        elif config.args.patch_algo == 'probes':
            topk_df = pd.read_csv(f"{config.get_output_prefix()}/eval/probes_targeted_{topk}.csv")    
        else:
            topk_df = pd.read_csv(f"{config.get_output_prefix()}/eval/numerator_1_targeted_{topk}.csv")
        reps = "random" if is_random(config.args.patch_algo) else "targeted"
        for N in range(1, 11):
            gen_file = f"{config.get_output_prefix()}/eval/{N}_{reps}_pyreft_{topk}_gen.txt"
            print('Entering generation loop for PyReFT...')
            if os.path.exists(gen_file) and os.path.exists(gen_file.replace('.txt', '.json')):
                print(f"Skipping generation as all relevant files exist.")
                continue
            else:
                break
        if N >= 10:
            print(f"All generations for PyReFT with topk {topk} exist. Skipping to next topk.")
            continue
        reft_layers_config = get_reft_layers_config(topk_df, model)

        reft_model = pyreft.get_reft_model(model, reft_layers_config)
        reft_model.set_device(config.args.device)
        reft_model.print_trainable_parameters()
        data = data_handler.pyreft_prompts
        cf_data = {
            'input_ids': data_handler.pyreft_toks['input_ids'].clone(),
            'attention_mask': data_handler.pyreft_toks['attention_mask'].clone()
        }
        labels = cf_data['input_ids'].clone()
        for i in range(labels.shape[0]):
            start_pos = data_handler.response_start_positions['pyreft'][i]
            labels[i, :start_pos] = -100  # Ignore tokens before the response start position
        labels[cf_data['attention_mask'] == 0] = -100
        reft_model = reft_train(topk_df, reft_model, cf_data, labels, batch_size=10, lr=4e-3, num_epochs=100, device=config.args.device, display_bar=True)
        for N in range(1, 11):
            batch_handler = BatchHandler(config, data_handler)
            decoded_responses = {
                "pyreft": {
                    reps: {
                        str(topk): []
                    }
                }
            }
            gen_file = f"{config.get_output_prefix()}/eval/{N}_{reps}_pyreft_{topk}_gen.txt"
            print('Entering generation loop for PyReFT...')
            if os.path.exists(gen_file) and os.path.exists(gen_file.replace('.txt', '.json')):
                print(f"Skipping generation as all relevant files exist.")
                continue
            for idx in tqdm(range(0, min(len_gen_qs, data_handler.LEN), config.args.batch_size)):
                gen_qs_toks = select_gen_qs_toks(config, batch_handler)
                topk_heads = get_intervention_locations(topk_df, gen_qs_toks["input_ids"])
                _, edited_outputs = reft_model.generate(
                    gen_qs_toks, unit_locations={
                        'sources->base': (
                            None,  # copy from
                            topk_heads  # paste to
                        )
                    },
                    intervene_on_prompt=True, max_new_tokens=config.args.max_new_tokens, do_sample=False,
                    eos_token_id=model_handler.tokenizer.eos_token_id, early_stopping=True,
                    intervention_additional_kwargs={'S': N}
                )
                decoded = decode_responses(model, gen_qs_toks, original_outputs[idx:idx + config.args.batch_size], edited_outputs, config.args.base)
                decoded_responses["pyreft"][reps][str(topk)] += decoded

                batch_handler.update()
            save_prompt_responses(decoded_responses["pyreft"][reps][str(topk)], gen_file)

        # Delete and reload model to clear PyReFT modifications
        del model
        torch.cuda.empty_cache()
        gc.collect()
        model = model_handler.load_model(config.args.model_id, config.args.device)
        model.tokenizer = model_handler.tokenizer


def run_eval_transfer(config, data_handler, model_handler, batch_handler, patching_utils):
    best_algorithms = pd.read_csv('/mnt/align4_drive/arunas/multi-token/gcm-interp/results/new-accuracies/plots/best_topk_N_per_method_per_ablation.csv')
    best_algorithms = best_algorithms[best_algorithms['ablation'] == config.args.ablation]


    best_method = ast.literal_eval(best_algorithms[(best_algorithms['model'] == config.args.model_id.split('/')[-1]) & (best_algorithms['task'] == f"from_{config.args.source}_to_{config.args.base}") & (best_algorithms['ablation'] == config.args.ablation)]['pairs'].item())

    if len(best_method) > 1:
        best_method = random.Random(42).choice(best_method)
    else:
        best_method = best_method[0]

    model = model_handler.model
    patching_reps = load_patching_reps(data_handler, model_handler)
    ablation = data_handler.config.args.ablation
    reps_type = 'random' if is_random(config.args.patch_algo) else 'targeted'

    print(best_method)
    config.args.N = int(best_method['steering_factor'])
    topk = float(best_algorithms[(best_algorithms['model'] == config.args.model_id.split('/')[-1]) & (best_algorithms['task'] == f"from_{config.args.source}_to_{config.args.base}") & (best_algorithms['ablation'] == config.args.ablation)]['best_topk'].item())
    config.args.patch_algo = best_method['method']
    test_accuracy = float(best_method['accuracy'])
    print(f"Using topk: {topk}, N: {config.args.N}, algo: {config.args.patch_algo} for evaluation on transfer data (Test accuracy was {test_accuracy}).")
    ablation = data_handler.config.args.ablation
    
    config.set_output_prefix()
    os.makedirs(f"{config.get_output_prefix()}/eval/", exist_ok=True)
    if config.args.patch_algo == 'probes':
        logit_metric = 'probes'
    elif is_random(config.args.patch_algo):
        logit_metric = 'random'
    else:
        logit_metric = 'numerator_1'

    topk_df = pd.read_csv(f"{config.get_output_prefix()}/eval/{logit_metric}_{reps_type}_{topk}.csv")
    edited_outputs = []
    original_outputs = []

    decoded_responses = {
        ablation: {
            reps_type: {
                topk: []
            }
        }
    }
    batch_handler = BatchHandler(config, data_handler)
    print('Best method: ', config.args.patch_algo, ' N: ', config.args.N, ' topk: ', topk, ' test accuracy ', )
    for idx in tqdm(range(0, data_handler.LEN, config.args.batch_size), desc="Processing samples"):
        gen_file = f"{config.get_output_prefix()}/eval/{config.args.N}_{config.args.eval_transfer}_{reps_type}_{ablation}_{topk}_gen_{config.args.seed}.txt"
        if os.path.exists(gen_file.replace('.txt', '_accuracy_responses.json')):
            print(f"Skipping generation as all relevant files exist.")
            return
        gen_qs_toks = select_gen_qs_toks(config, batch_handler)
        edited_outputs = generate_with_patches(model, gen_qs_toks, patching_reps[ablation], topk_df, config.args.N, ablation, model_handler.dim, max_new_tokens=256, normalize=False, steering_type=config.args.steering_type, patch_site=get_site(config.args))
        with model.generate(gen_qs_toks, do_sample=False, max_new_tokens=256) as _:
            original_outputs = model.generator.output.save()
        if config.args.eval_transfer:
            answers = batch_handler.eval_transfer['answers']
        else:
            answers = None
        decoded = decode_responses(model, gen_qs_toks, original_outputs, edited_outputs, config.args.base, answers=answers)

        if len(decoded_responses[ablation][reps_type][topk]) == 0:
            decoded_responses[ablation][reps_type][topk] = decoded
        else:
            decoded_responses[ablation][reps_type][topk] += decoded
        batch_handler.update()

        os.makedirs(f"{config.get_output_prefix()}/eval/", exist_ok=True)
        save_prompt_responses(decoded_responses[ablation][reps_type][topk], gen_file)

    del model_handler.model
    del model_handler.tokenizer
    gc.collect()
    torch.cuda.empty_cache()
