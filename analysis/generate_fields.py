#!/usr/bin/env python3
"""Generate ATP attribution FIELDS for every localization, at a chosen site/span.

Why this exists rather than `bash scripts/..._vp.sh`: the representation analysis
needs only the attribution field, not steering. The shipped path reaches the
field through run_eval(), which first sweeps 8 topk x 7 steering factors x 50
generated items per cell -- hours of generation whose output this analysis never
reads. This runs `--patch_model` alone and then does the same reduction and the
same selection the pipeline does, so the fields are identical to what a full run
would have written, at a fraction of the cost.

    source env.sh
    python analysis/generate_fields.py --patch_site o_proj_in --response_span full

Grouped by model: one checkpoint load serves all of that model's localizations
(44 loads -> 5). Resume is by output file, so re-running skips finished work --
but if the metric or the site changes, DELETE the tree rather than resuming, or
stale fields are silently reused (CLAUDE.md section 4).

Writes to  results/{model}/from_{src}_to_{base}/{algo_dir}/attribution/
where algo_dir carries the site and span suffixes, so no site or metric can ever
land in another's tree.
"""
import argparse, os, sys, time, json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
os.chdir(REPO)
sys.path.insert(0, str(REPO))

from determinism import set_cublas_env, enable_determinism
set_cublas_env()

import torch
import einops
import pandas as pd

from config import Config
from model_handler import ModelHandler
from data_handler import DataHandler
from experiment import Experiment
# The SELECTION is shipped code on purpose: ranking is signed and descending
# (flat.topk), and re-implementing it here is exactly how an analysis drifts from
# the pipeline it is describing (FINDINGS 10.2).
from eval.logits_handler import get_top_k_layer_and_head
from eval.patch_site import SITES, SITE_OUT, dir_suffix as site_suffix
from eval.response_span import SPANS, SPAN_LEGACY, dir_suffix as span_suffix

MODELS = {"Falcon3-10B-Instruct": "tiiuae/Falcon3-10B-Instruct",
          "Qwen1.5-14B-Chat": "Qwen/Qwen1.5-14B-Chat",
          "Qwen1.5-32B-Chat": "Qwen/Qwen1.5-32B-Chat",
          "OLMo-2-1124-13B-DPO": "allenai/OLMo-2-1124-13B-DPO",
          "gemma-3-12b-it": "google/gemma-3-12b-it"}

# task -> (source stem, base stem, base carries the arm suffix)
TASKS = {"verse": ("verse", "prose", False),
         "summarization": ("paragraph", "sentence", False),
         "persona": ("extraversion", "introversion", True),
         "bias": ("female", "male", True),
         "factual recall": ("lying", "truthful", True)}

TOPKS = [0.01, 0.03, 0.05, 0.07, 0.09, 0.1, 0.5, 1.0]


def required_files(model, source, base):
    d = f"data/{model}/{source}"
    return [f"{d}/{base}-desired-all.jsonl", f"{d}/{base}-undesired-all.jsonl",
            f"{d}/{source}-desired-all.jsonl", f"{d}/{source}-undesired-all.jsonl"]


def enumerate_cells(models, tasks):
    out = []
    for m in models:
        for t in tasks:
            src, base, suf = TASKS[t]
            for arm in ("long", "single"):
                s = f"{src}-{arm}"
                b = f"{base}-{arm}" if suf else base
                miss = [f for f in required_files(m, s, b) if not os.path.exists(f)]
                out.append(dict(model=m, task=t, arm=arm, source=s, base=b, missing=miss))
    return out


def reduce_field(attr_dir, n_items, num_heads):
    """heads_{i}.pt -> [n_layers, num_heads, n_items], the shipped reduction.

    Mirrors eval/logits_handler.load_logits' atp branch. It is not called
    directly because its path arithmetic ('/'.join(prefix.split('/')[:-3])) is
    written for the EVAL prefix, which has two more components than the
    attribution prefix this step produces; pointing it here would silently read
    the wrong directory.
    """
    all_logits = None
    found = 0
    for i in range(n_items):
        p = f"{attr_dir}/heads_{i}.pt"
        if not os.path.exists(p):
            continue
        t = torch.load(p).squeeze().unsqueeze(-1)
        all_logits = t if all_logits is None else torch.cat([all_logits, t], dim=-1)
        found += 1
    if all_logits is None:
        return None, 0
    return einops.reduce(all_logits, 'l (n m) b -> l n b', 'sum', n=num_heads), found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patch_site", default=SITE_OUT, choices=list(SITES))
    ap.add_argument("--response_span", default=SPAN_LEGACY, choices=list(SPANS))
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--tasks", default=",".join(TASKS))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--patch_algo", default="atp")
    ap.add_argument("--no_deterministic", action="store_true")
    ap.add_argument("--dry_run", action="store_true")
    a = ap.parse_args()

    models = [m for m in a.models.split(",") if m in MODELS]
    tasks = [t.strip() for t in a.tasks.split(",") if t.strip() in TASKS]
    algo_dir = f"{a.patch_algo}{site_suffix(a.patch_site)}{span_suffix(a.response_span)}"

    cells = enumerate_cells(models, tasks)
    runnable = [c for c in cells if not c["missing"]]
    blocked = [c for c in cells if c["missing"]]

    for c in runnable:
        c["attr_dir"] = f"results/{c['model']}/from_{c['source']}_to_{c['base']}/{algo_dir}"
        c["out_dir"] = f"{c['attr_dir']}/attribution"
        c["done"] = os.path.exists(f"{c['out_dir']}/numerator_1_targeted_1.0.csv")

    todo = [c for c in runnable if not c["done"]]
    print(f"site/span : {a.patch_site} / {a.response_span}  -> results dir '{algo_dir}'")
    print(f"localizations: {len(runnable)} runnable, {len(todo)} to do, "
          f"{len(runnable)-len(todo)} already complete, {len(blocked)} blocked (no data)")
    for c in blocked:
        print(f"    BLOCKED {c['model']:<22}{c['task']:<16}{c['arm']:<7} missing {len(c['missing'])} file(s)")
    if a.dry_run:
        for c in todo:
            print(f"    TODO    {c['model']:<22}{c['task']:<16}{c['arm']:<7}"
                  f" {c['source']} -> {c['base']}")
        return

    by_model = {}
    for c in todo:
        by_model.setdefault(c["model"], []).append(c)

    t_all = time.time()
    for model, cs in by_model.items():
        print(f"\n{'='*90}\n{model}   {len(cs)} localization(s)\n{'='*90}", flush=True)
        mh = None
        for c in cs:
            t0 = time.time()
            print(f"\n--- {c['task']} / {c['arm']}   {c['source']} -> {c['base']}", flush=True)
            sys.argv = ["run.py", "--model_id", MODELS[model], "--batch_size", "1",
                        "--patch_algo", a.patch_algo,
                        "--patch_site", a.patch_site,
                        "--response_span", a.response_span,
                        "--source", c["source"], "--base", c["base"],
                        "--device", a.device, "--patch_model"]
            if a.no_deterministic:
                sys.argv.append("--no_deterministic")
            config = Config()
            if not a.no_deterministic:
                enable_determinism()
            if mh is None:
                mh = ModelHandler(config)
            else:
                # Reuse the loaded checkpoint. Only .config is per-cell; .dim and
                # .num_heads follow the site, which is fixed for the whole run, and
                # the assistant marker depends on the model, not the task.
                mh.config = config
            dh = DataHandler(config, mh)
            config.args.batch_size = 1

            Experiment(config, dh, mh, "heads").run()

            attr_dir = config.get_output_prefix().rstrip("/")
            field, found = reduce_field(attr_dir, dh.LEN, mh.num_heads)
            if field is None or found != dh.LEN:
                print(f"    !! INCOMPLETE: {found}/{dh.LEN} item files; not writing a field")
                continue
            torch.save(field, f"{attr_dir}/numerator_1_heads.pt")
            os.makedirs(c["out_dir"], exist_ok=True)
            for k in TOPKS:
                df = get_top_k_layer_and_head(field, k, a.patch_algo)
                df.to_csv(f"{c['out_dir']}/numerator_1_targeted_{k}.csv", index=False)
            print(f"    field {tuple(field.shape)} from {found} items -> "
                  f"{c['out_dir']}  ({time.time()-t0:.0f}s)", flush=True)
        del mh
        torch.cuda.empty_cache()
    print(f"\nDONE in {(time.time()-t_all)/60:.1f} min")


if __name__ == "__main__":
    main()
