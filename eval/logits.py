import torch
import gc
from tqdm import tqdm

from eval.patch_site import SITE_OUT, attn_proxy

def patch_heads_and_get_logit(model, DIM, patching_reps, top_indices, base_toks, resp_start, N, ablation_type, get_response_logits, top_tokens=False, patch_site=SITE_OUT):
    patching_reps = patching_reps.to(model.device)
    with model.trace(base_toks) as _:
        for _, row in top_indices.iterrows():
            layer = int(row['layer'])
            head = int(row['neuron'])
            head_slice = slice(DIM * head, DIM * (head + 1))
            proxy = attn_proxy(model.model.layers[layer], patch_site)
            if ablation_type == 'mean':
                proxy[:, :patching_reps.shape[1], head_slice] = N * patching_reps[layer][:, head_slice]
            elif ablation_type == 'steer':
                proxy[:, :patching_reps.shape[1], head_slice] += N * patching_reps[layer][:, head_slice]
        logits = model.lm_head.output.save()
    if top_tokens:
        last_logits = logits[:, -1:, :].detach().cpu()
        top_tokens = torch.argmax(last_logits, dim=-1)
        gc.collect()
        torch.cuda.empty_cache()
        return top_tokens
    
    logits = get_response_logits(base_toks, resp_start, logits)
    gc.collect()
    torch.cuda.empty_cache()
    return logits

def get_logits_before_patch(model, batch_handler, get_response_logits):
    base_toks = batch_handler.base_toks
    response_start_positions = batch_handler.response_start_positions
    scores = {}
    for key in ['desired', 'undesired']:
        with model.trace(base_toks[key]) as _:
            logits = model.lm_head.output.save()
        logits = get_response_logits(base_toks[key], response_start_positions['base'][key], logits)
        scores[key] = logits
    gc.collect()
    torch.cuda.empty_cache()
    stacked = torch.stack([scores['desired'], scores['undesired']])
    return stacked

def compute_logit_scores(batch_handler, topk_df, patching_reps, model_handler, ablation_type, get_response_logits, N, patch_site=None):
    base_toks = batch_handler.base_toks
    response_start_positions = batch_handler.response_start_positions
    scores = {}
    for key in ['desired', 'undesired']:
        logits = patch_heads_and_get_logit(
            model_handler.model,
            model_handler.dim,
            patching_reps[key],
            topk_df,
            base_toks[key],
            response_start_positions['base'][key],
            N,
            ablation_type,
            get_response_logits,
            patch_site=patch_site or getattr(model_handler, 'patch_site', SITE_OUT),
        )
        scores[key] = logits
    stacked = torch.stack([scores['desired'], scores['undesired']])
    return stacked

def get_heads(model, DIM, patching_reps, toks, N, ablation_type, patch=True, patch_site=SITE_OUT):
    heads_by_layer = []
    
    for layer in tqdm(range(len(model.model.layers)), desc="Collecting heads for layer"):
        heads = []
        for head in range(model.config.num_attention_heads):
            head_slice = slice(DIM * head, DIM * (head + 1))
            with model.trace(toks) as _:
                if patch == True:
                    heads.append(N * patching_reps[layer][:, head_slice].detach().cpu())
                else:
                    heads.append(attn_proxy(model.model.layers[layer], patch_site)[:, :, head_slice].detach().cpu().save())
        heads_by_layer.append(torch.stack(heads, dim =1))

    heads_by_layer = torch.stack(heads_by_layer)
    gc.collect()
    torch.cuda.empty_cache()
    return heads_by_layer