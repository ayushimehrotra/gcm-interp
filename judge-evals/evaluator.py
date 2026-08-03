"""
vLLM judge model — shared inference utilities for run_judge.py.
"""

import json
import os
from pathlib import Path

import pandas as pd
import torch
from vllm import LLM, SamplingParams

os.environ.setdefault("VLLM_USE_DEEP_GEMM", "0")

from config import JUDGE_MODEL_NAME, PASSTHROUGH_COLS, extract_rating
from compute_accuracies import extract_first_int


# ---------------------------------------------------------------------------
# Model setup
# ---------------------------------------------------------------------------

SEED = 42


def make_llm(model_name: str = JUDGE_MODEL_NAME) -> LLM:
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("No GPUs detected!")
    print(f"Detected {num_gpus} GPU(s). Loading judge model: {model_name}")

    # A 70B judge in 4-bit is ~35GB of weights, which leaves no room for a KV cache
    # under vLLM's 0.9 default on a 40GB card. These knobs are memory-only — they
    # change batching and cache size, not judge outputs — so override them via env
    # on small cards and leave the defaults for larger ones.
    #   JUDGE_GPU_MEM_UTIL=0.97 JUDGE_MAX_NUM_SEQS=8 JUDGE_MAX_MODEL_LEN=2048
    # The 70B checkpoint is ~37GB, so on a 40GB card the weights themselves do not
    # fit and no utilization setting helps — set JUDGE_CPU_OFFLOAD_GB to stream part
    # of them from host RAM instead (slower, same outputs).
    kwargs = {}
    offload_gb = float(os.environ.get("JUDGE_CPU_OFFLOAD_GB", 0))
    if offload_gb > 0:
        kwargs["cpu_offload_gb"] = offload_gb
        print(f"Offloading {offload_gb}GB of judge weights to CPU")

    return LLM(
        model=model_name,
        quantization="bitsandbytes",
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        dtype="auto",
        max_num_seqs=int(os.environ.get("JUDGE_MAX_NUM_SEQS", 64)),
        max_model_len=int(os.environ.get("JUDGE_MAX_MODEL_LEN", 4096)),
        gpu_memory_utilization=float(os.environ.get("JUDGE_GPU_MEM_UTIL", 0.90)),
        seed=SEED,
        **kwargs,
    )


def get_sampling_params() -> SamplingParams:
    return SamplingParams(
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        max_tokens=3,
        seed=SEED,
    )


# ---------------------------------------------------------------------------
# Batched generation
# ---------------------------------------------------------------------------

def generate_in_batches(llm, prompts, sampling_params, batch_size):
    """Yield decoded output strings batch by batch."""
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        results = llm.generate(batch, sampling_params)
        yield [r.outputs[0].text for r in results]


