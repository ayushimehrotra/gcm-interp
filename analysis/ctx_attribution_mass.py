"""Where each --localization_ctx cell's attribution mass sits, and what is being
differenced there.

Motivation: a prompt-only source ends at the generation prompt, so after
align_toks it holds PAD at every response position. Since ~95% of |grad*(A_src -
A_base_patch)| lives at response positions, the default br-sq cell differences
real response activations against PADDING -- i.e. largely response-presence, not
the style contrast. br-sr is the only cell with real activations on both sides
there. This script measures that split.

Single batch (item 0) as written -- widen the loop before quoting numbers.
"""
import sys; sys.path.insert(0,"/home/ubuntu/gcm-interp")
import torch
sys.argv = ["run.py","--model_id","Qwen/Qwen1.5-14B-Chat","--batch_size","1","--patch_algo","atp",
            "--source","verse-long","--base","prose","--device","cuda:0","--patch_model",
            "--localization_ctx","bq-sr","--no_deterministic"]
from config import Config
from model_handler import ModelHandler
from data_handler import DataHandler
from batch_handler import BatchHandler
from patching import Patching
cfg=Config(); mh=ModelHandler(cfg); cfg.args.batch_size=1
dh=DataHandler(cfg,mh); bh=BatchHandler(cfg,dh,0,1); p=Patching(mh,bh,cfg); pu=p.patching_utils
bt=bh.base_toks; bqt=bh.base_qs_toks; sqt=bh.source_qs_toks; st=bh.source_toks
rsp=bh.response_start_positions; pad=mh.tokenizer.pad_token_id
K='desired'

lg ,attn  = pu.get_activations(bt[K],which_patch='heads',resp_start_positions=rsp['base'][K],retain_grad=True,logit=True)
lg2,attn2 = pu.get_activations(bt['undesired'],which_patch='heads',resp_start_positions=rsp['base']['undesired'],retain_grad=True,logit=True)
A=lambda t: pu.get_activations(t,which_patch='heads',resp_start_positions=None,logit=False,align=True,retain_grad=True,base_toks=bt[K])
src_q, src_r, base_q = A(sqt[K]), A(st[K]), A(bqt[K])
(lg2-lg).backward(retain_graph=True)

rs=rsp['base'][K][0]; mask=bt[K]["attention_mask"][0].bool()
resp=torch.zeros_like(mask); resp[rs:]=True; resp&=mask; prompt=mask&(~resp)
al=lambda t: p.align_toks(t,bt[K])["input_ids"][0]
padinfo={'src_qs':al(sqt[K]),'src_full':al(st[K]),'base_qs':al(bqt[K])}
V=lambda x,i: (x[i] if isinstance(x[i],torch.Tensor) else x[i].value)[0].float()

CFG={'br-sq (current)':('src_q','base_full'),'br-sr':('src_r','base_full'),
     'bq-sq':('src_q','base_q'),      'bq-sr':('src_r','base_q')}
T={'src_q':src_q,'src_r':src_r,'base_q':base_q,'base_full':attn}
print(f"\nresponse positions={int(resp.sum())}  prompt positions={int(prompt.sum())}")
for n,t in padinfo.items():
    print(f"  {n:9s}: PAD at {int((t==pad)[resp].sum())}/{int(resp.sum())} response positions")
print(f"\n{'config':16s} {'|attr| @prompt':>15s} {'|attr| @response':>17s}")
for name,(s,b) in CFG.items():
    ep=er=0.0
    for i in range(len(mh.model.model.layers)):
        g=attn[i].value.grad[0].float()
        d=V(T[s],i)-V(T[b],i)
        e=(g*d).abs().sum(-1)
        ep+=e[prompt].sum().item(); er+=e[resp].sum().item()
    tot=ep+er
    print(f"{name:16s} {100*ep/tot:14.1f}% {100*er/tot:16.1f}%")
