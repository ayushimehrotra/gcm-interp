"""
KL faithfulness eval: how faithful is the steered (prose-prompt) generation to the
*unsteered* model's response when the same question is asked in verse.

For each test question q:
  1. Reference T  = unsteered model's greedy verse answer to q
                    (prompt: "Respond in verse and concisely. <q>")
  2. Teacher-force T under two conditions, position-aligned by the i-th token of T:
       Q_i = unsteered model on the VERSE prompt    (reference distribution)
       P_i = steered   model on the PROSE prompt    (last_token + normalized steering,
                                                      exactly as generate_with_patches)
  3. Per-token KL averaged over T, then averaged over questions.

Steering replicates eval/generation.py::generate_with_patches:
  steering_vec = normalize(patch_activations[layer][-1, head_slice]); applied as
  o_proj.output[..., :reps_seq_len, head_slice] += N * steering_vec
i.e. only the prompt window is steered (T continuation positions are not), matching
how the post-intervention responses were generated.
"""
import os, sys, json, argparse
import torch
import torch.nn.functional as F
import pandas as pd

# ---- our own args (parse + strip before Config sees argv) -------------------
ap = argparse.ArgumentParser()
ap.add_argument('--N', type=int, default=10, help='steering coefficient')
ap.add_argument('--topk', type=str, default='0.03', help='topk fraction (heads csv)')
ap.add_argument('--limit', type=int, default=50, help='number of test questions')
ap.add_argument('--max_new_tokens', type=int, default=256)
ap.add_argument('--model', type=str, default='Qwen1.5-14B-Chat',
                help='model name, e.g. Qwen1.5-14B-Chat or Qwen1.5-32B-Chat')
ap.add_argument('--out', type=str, default=None)
my_args = ap.parse_args()

MODEL_NAME = my_args.model
MODEL_ID = f'Qwen/{MODEL_NAME}'
DATA = f'/home/ubuntu/gcm-interp/data/{MODEL_NAME}/verse-long'
# argv that the repo's Config expects (mirrors scripts/qwen_vp-long.sh last eval call)
sys.argv = [
    'run.py',
    '--model_id', MODEL_ID,
    '--batch_size', '1',
    '--patch_algo', 'atp',
    '--source', 'verse-long',
    '--base', 'prose',
    '--device', 'cuda:0',
    '--eval_model',
    '--eval_test', f'{DATA}/prose-test.jsonl',
    '--steering',
    '--ablation', 'steer',
    '--steering_add_path', f'{DATA}/verse-long-desired-all.jsonl',
    '--steering_sub_path', f'{DATA}/prose-desired-all.jsonl',
]

from config import Config
from model_handler import ModelHandler
from data_handler import DataHandler
from eval.eval_runner import load_patching_reps

config = Config()
config.args.batch_size = 1
mh = ModelHandler(config)
model = mh.model
model.eval()
tok = mh.tokenizer
device = mh.device
DIM = mh.dim

dh = DataHandler(config, mh)
LEN = min(my_args.limit, dh.LEN)

# steering reps  -> patch_activations: [num_layers, reps_seq_len, hidden]
patching_reps = load_patching_reps(dh, mh)
patch_activations = patching_reps['steer']['desired'].to(device)
REPS_SEQ = patch_activations.shape[1]
print(f'patch_activations: {tuple(patch_activations.shape)}  reps_seq_len={REPS_SEQ}')

# heads for this topk
csv = f"{config.get_output_prefix()}/eval/numerator_1_targeted_{my_args.topk}.csv"
topk_df = pd.read_csv(csv)
layer_ids = topk_df['layer'].unique()
print(f'Loaded {len(topk_df)} heads from {csv}  (N={my_args.N}, topk={my_args.topk})')

N = my_args.N
EOS = tok.eos_token_id
PAD = tok.pad_token_id

# prose prompts already tokenized + left-padded to max_len by DataHandler
prose_ids_all = dh.base_qs_toks['test']['input_ids']        # [LEN, max_len]
prose_mask_all = dh.base_qs_toks['test']['attention_mask']
MAX_LEN = prose_ids_all.shape[1]
assert REPS_SEQ == MAX_LEN, f'reps_seq {REPS_SEQ} != prose max_len {MAX_LEN}'

# raw questions (same order as prose-test / base_qs_toks['test'])
raw = dh.load_from_jsonl(f'{DATA}/prose-test.jsonl')

def verse_prompt_ids(example):
    msgs = [dict(m) for m in example['prompt']]
    for m in msgs:
        if m['role'] == 'user':
            m['content'] = m['content'].replace('Respond in prose', 'Respond in verse')
    s = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    ids = tok(s, return_tensors='pt')['input_ids'].to(device)
    return ids  # [1, vp_len]

@torch.no_grad()
def gen_verse(vp_ids):
    am = torch.ones_like(vp_ids)
    with model.generate({'input_ids': vp_ids, 'attention_mask': am},
                        pad_token_id=EOS, use_cache=False, do_sample=False,
                        top_p=None, top_k=None, temperature=None,
                        max_new_tokens=my_args.max_new_tokens) as _:
        op = model.generator.output.save()
    seq = op[0].detach().cpu()
    T = seq[vp_ids.shape[1]:]
    # cut at first eos / pad
    keep = []
    for t in T.tolist():
        if t in (EOS, PAD):
            break
        keep.append(t)
    return torch.tensor(keep, dtype=torch.long)

@torch.no_grad()
def logits_unsteered(full_ids):
    am = torch.ones_like(full_ids)
    with model.trace({'input_ids': full_ids, 'attention_mask': am}) as _:
        lg = model.lm_head.output.save()
    return lg[0].detach()          # [L, V]

@torch.no_grad()
def logits_steered(full_ids, full_mask):
    with model.trace({'input_ids': full_ids, 'attention_mask': full_mask}) as _:
        for layer_idx in layer_ids:
            heads = topk_df[topk_df['layer'] == layer_idx]['neuron'].unique()
            layer = model.model.layers[int(layer_idx)]
            for head_idx in heads:
                sl = slice(DIM * int(head_idx), DIM * (int(head_idx) + 1))
                sv = patch_activations[int(layer_idx)][-1, sl]
                sv = sv / (torch.norm(sv, dim=-1, keepdim=True) + 1e-12)
                layer.self_attn.o_proj.output[..., :REPS_SEQ, sl] += N * sv
        lg = model.lm_head.output.save()
    return lg[0].detach()          # [L, V]

def kl_rows(logP, logQ):
    # mean over positions of KL(Q||P) = sum_v exp(logQ)*(logQ-logP)
    return (logQ.exp() * (logQ - logP)).sum(-1)   # [npos]

results = []
for i in range(LEN):
    vp = verse_prompt_ids(raw[i])
    vp_len = vp.shape[1]
    T = gen_verse(vp)
    if T.numel() == 0:
        print(f'[{i}] empty verse generation, skipping'); continue
    Tlen = T.numel()
    Td = T.to(device)

    # unsteered verse: verse_prompt + T ; positions predicting T[i] = vp_len-1+i
    full_q = torch.cat([vp[0], Td]).unsqueeze(0)
    lgQ = logits_unsteered(full_q)[vp_len - 1: vp_len - 1 + Tlen].float()

    # steered prose: prose_prompt(padded max_len) + T ; positions predicting T[i] = MAX_LEN-1+i
    full_p_ids = torch.cat([prose_ids_all[i], Td]).unsqueeze(0)
    full_p_mask = torch.cat([prose_mask_all[i], torch.ones(Tlen, dtype=prose_mask_all.dtype, device=device)]).unsqueeze(0)
    lgP = logits_steered(full_p_ids, full_p_mask)[MAX_LEN - 1: MAX_LEN - 1 + Tlen].float()

    logP = F.log_softmax(lgP, dim=-1)
    logQ = F.log_softmax(lgQ, dim=-1)

    kl_QP = kl_rows(logP, logQ).mean().item()                     # KL(unsteered||steered)
    kl_PQ = (logP.exp() * (logP - logQ)).sum(-1).mean().item()    # KL(steered||unsteered)
    idx = Td.unsqueeze(-1)
    nll_steered = (-logP.gather(-1, idx).squeeze(-1)).mean().item()
    nll_unsteer = (-logQ.gather(-1, idx).squeeze(-1)).mean().item()

    results.append({'i': i, 'Tlen': Tlen,
                    'kl_unsteered_to_steered': kl_QP,
                    'kl_steered_to_unsteered': kl_PQ,
                    'kl_symmetric': 0.5 * (kl_QP + kl_PQ),
                    'nll_steered_of_T': nll_steered,
                    'nll_unsteered_of_T': nll_unsteer})
    print(f'[{i:2d}] Tlen={Tlen:3d}  KL(Q||P)={kl_QP:.4f}  KL(P||Q)={kl_PQ:.4f}  '
          f'NLL_steered={nll_steered:.3f}  NLL_unsteered={nll_unsteer:.3f}')
    del lgQ, lgP, logP, logQ
    torch.cuda.empty_cache()

def mean(k):
    return sum(r[k] for r in results) / len(results)

summary = {
    'N': N, 'topk': my_args.topk, 'n_examples': len(results),
    'mean_kl_unsteered_to_steered': mean('kl_unsteered_to_steered'),
    'mean_kl_steered_to_unsteered': mean('kl_steered_to_unsteered'),
    'mean_kl_symmetric': mean('kl_symmetric'),
    'mean_nll_steered_of_T': mean('nll_steered_of_T'),
    'mean_nll_unsteered_of_T': mean('nll_unsteered_of_T'),
}
print('\n==== SUMMARY ====')
print(json.dumps(summary, indent=2))

out = my_args.out or f"{config.get_output_prefix()}/eval/kl_{N}_targeted_{my_args.topk}.json"
with open(out, 'w') as f:
    json.dump({'summary': summary, 'per_example': results}, f, indent=2)
print('Saved per-config json ->', out)

# Append the summary as one row to a master CSV so a sweep accumulates in one place.
csv_out = f"{config.get_output_prefix()}/eval/kl_summary.csv"
cols = ['N', 'topk', 'n_examples', 'mean_kl_unsteered_to_steered',
        'mean_kl_steered_to_unsteered', 'mean_kl_symmetric',
        'mean_nll_steered_of_T', 'mean_nll_unsteered_of_T']
write_header = not os.path.exists(csv_out)
with open(csv_out, 'a') as f:
    if write_header:
        f.write(','.join(cols) + '\n')
    f.write(','.join(str(summary[c]) for c in cols) + '\n')
print('Appended summary row ->', csv_out)
