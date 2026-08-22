"""Corrected-metric ATP attribution for Qwen1.5-14B-Chat / verse.

Writes NOTHING to the repo: the metric is monkeypatched at import time and
Config.set_output_prefix is redirected into the scratchpad. Repo files are read
only. Runs --patch_model only (no eval, no generations, no steering cache).

The fix: to score a response beginning at rs you need the logits at rs-1, since
position p's logits predict token p+1. The shipped code starts at rs, so the
first response token is never scored -- and for *-single data that token IS the
answer (the assistant turn is one letter).
"""
import os, sys, types, torch
import torch.nn.functional as F

REPO = "/home/ubuntu/gcm-interp"
OUT = "/tmp/claude-1000/-home-ubuntu-gcm-interp/a02cfc7e-2539-4a39-bb0e-ae627ca27e4b/scratchpad/fixed_attr"
os.chdir(REPO); sys.path.insert(0, REPO)

from determinism import set_cublas_env, enable_determinism
set_cublas_env()
from patching_utils import PatchingUtils


def fixed_get_response_logits(self, toks, resp_start_positions, logits, retain_grad=False):
    """Identical to the shipped version except the span: [rs-1:-1] vs input_ids[rs:]."""
    if retain_grad:
        log_probs = F.log_softmax(logits, dim=-1)
        toks = {'input_ids': toks['input_ids'], 'attention_mask': toks['attention_mask']}
        log_likelihoods = torch.stack([
            log_probs[i, r - 1:-1, :].gather(-1, toks['input_ids'][i, r:].unsqueeze(-1)).squeeze(-1).sum()
            for i, r in enumerate(resp_start_positions)
        ])
    else:
        logits = logits.detach().cpu()
        log_probs = F.log_softmax(logits, dim=-1).detach().cpu()
        toks = {'input_ids': toks['input_ids'].detach().cpu(),
                'attention_mask': toks['attention_mask'].detach().cpu()}
        log_likelihoods = torch.stack([
            log_probs[i, r - 1:-1, :].gather(-1, toks['input_ids'][i, r:].unsqueeze(-1)).squeeze(-1).sum()
            for i, r in enumerate(resp_start_positions)
        ]).detach().cpu()
    toks = {'input_ids': toks['input_ids'].to(self.model_handler.device),
            'attention_mask': toks['attention_mask'].to(self.model_handler.device)}
    return log_likelihoods


# ---- prove the patch is live before spending any GPU time -------------------
def selftest():
    pu = types.SimpleNamespace(model_handler=types.SimpleNamespace(device='cpu'))
    ids = torch.tensor([[5, 6, 7, 8, 9]])            # response starts at index 3 -> token 8
    toks = {'input_ids': ids, 'attention_mask': torch.ones_like(ids)}
    rs = [3]
    torch.manual_seed(0); base = torch.randn(1, 5, 20)
    for name, fn in (("shipped", PatchingUtils.get_response_logits), ("fixed", fixed_get_response_logits)):
        v0 = fn(pu, toks, rs, base.clone(), retain_grad=True)[0].item()
        bumped = base.clone(); bumped[0, 2, 8] += 100.0     # logits at rs-1 that predict the response token
        v1 = fn(pu, toks, rs, bumped, retain_grad=True)[0].item()
        print(f"  [selftest] {name:8}: LL {v0:9.4f} -> {v1:9.4f} when the FIRST response token is made likely"
              f"   {'(blind to it)' if abs(v1-v0) < 1e-6 else '(sees it)'}")
    assert abs(fixed_get_response_logits(pu, toks, rs, base.clone(), retain_grad=True)[0].item()
               - PatchingUtils.get_response_logits(pu, toks, rs, base.clone(), retain_grad=True)[0].item()) > 1e-6


selftest()
PatchingUtils.get_response_logits = fixed_get_response_logits
print("  [patch] PatchingUtils.get_response_logits replaced\n", flush=True)

from config import Config
import run as run_module

LOC = sys.argv[1]          # verse-long | verse-single
SITE = sys.argv[2]         # o_proj_out | o_proj_in
dest = f"{OUT}/{LOC}/{SITE}/"
os.makedirs(dest, exist_ok=True)


def redirected(self):
    self.output_prefix = dest
    return self.output_prefix


Config.set_output_prefix = redirected     # keeps every write out of results/

sys.argv = ["run.py", "--model_id", "Qwen/Qwen1.5-14B-Chat", "--batch_size", "1",
            "--patch_algo", "atp", "--patch_site", SITE, "--source", LOC,
            "--base", "prose", "--device", "cuda:0", "--patch_model"]
print(f"=== corrected-metric attribution: {LOC} @ {SITE} -> {dest}", flush=True)
run_module.main()
print(f"=== done: {LOC} @ {SITE}", flush=True)
