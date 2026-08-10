"""Mirror the uniform-random arm across the two localizations of a task family.

The uniform arm's head draw comes from retrieve_random_k(num_layers, num_heads,
topk, seed), which does not depend on --source. For a given eval mode the eval
set and the steering vectors are also passed explicitly and are identical across
the two localizations, so running `--source verse-long` and `--source
verse-single` produces bit-identical generations -- verified across 4 task pairs
x 72 files on Falcon3-10B-Instruct. Only the output directory differs.

Rather than spend the GPU time twice, run the `-long` localization and mirror its
random-s{seed} tree into the `-single` localization. Every task folder still ends
up with its own arm tree, laid out the same way.

This does NOT apply to the layer-matched arm: randomlayer draws from the real ATP
per-layer histogram, which is genuinely different per localization (2/24 shared
heads at k=0.05), so those are always run separately.

Usage:
  python random_control_mirror_uniform.py --model Falcon3-10B-Instruct --seed 0
  python random_control_mirror_uniform.py --model Qwen1.5-14B-Chat --seed 0 --verify
"""

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent
RESULTS = REPO / "results"

# family -> (base, long-localization task, single-localization task)
FAMILIES = {
    "verse": ("prose", "from_verse-long_to_prose", "from_verse-single_to_prose"),
    "paragraph": ("sentence", "from_paragraph-long_to_sentence",
                  "from_paragraph-single_to_sentence"),
}
TOPKS = [0.01, 0.03, 0.05, 0.07, 0.09, 0.1, 0.5, 1.0]


def geometry(model, task):
    """(num_layers, num_heads) from the topk=1.0 ATP CSV, which holds every head."""
    hits = sorted((RESULTS / model / task / "atp").glob(
        "*/*/eval/numerator_1_targeted_1.0.csv"))
    if not hits:
        raise FileNotFoundError(f"no ATP topk=1.0 CSV under {RESULTS/model/task}/atp")
    df = pd.read_csv(hits[0])
    return int(df["layer"].max()) + 1, int(df["neuron"].max()) + 1


def verify_draw(model, task, dst_tree, seed):
    """Every mirrored head-set CSV must equal what retrieve_random_k would draw.

    This is the check that the mirror is legitimate for THIS model: the drawn set
    is what the skipped run would itself have produced.
    """
    sys.path.insert(0, str(REPO))
    from eval.logits_handler import retrieve_random_k

    num_layers, num_heads = geometry(model, task)
    checked = 0
    for csv in sorted(dst_tree.glob("*/*/eval/random_random_*.csv")):
        topk = float(csv.stem.split("_")[-1])
        got = pd.read_csv(csv)[["layer", "neuron"]]
        want = retrieve_random_k(num_layers, num_heads, topk, seed=seed)
        if set(map(tuple, got.values)) != set(map(tuple, want[["layer", "neuron"]].values)):
            raise ValueError(
                f"mirrored draw does not match retrieve_random_k for {csv} "
                f"(topk={topk}, seed={seed}) -- refusing to trust the mirror")
        checked += 1
    return checked


def retag(dst_tree, src_task, dst_task, seed):
    """Point the copied config.yml at the task it now lives under, and leave a note.

    Nothing downstream reads config.yml (the judge derives metadata from the path),
    but a config claiming the wrong source would be a provenance trap.
    """
    src_source = src_task.split("from_")[1].split("_to_")[0]
    dst_source = dst_task.split("from_")[1].split("_to_")[0]
    for cfg in dst_tree.glob("*/*/config.yml"):
        text = cfg.read_text()
        cfg.write_text(text.replace(f"source: {src_source}\n", f"source: {dst_source}\n"))
    (dst_tree / "COPIED_FROM.txt").write_text(
        f"The uniform-random arm at draw seed {seed} was generated under\n"
        f"  {src_task}\n"
        f"and mirrored here ({dst_task}).\n\n"
        f"retrieve_random_k() ignores --source, and for a given eval mode the eval set\n"
        f"and steering vectors are identical across the two localizations, so the two\n"
        f"runs are bit-identical -- only the output path differs. Mirroring avoids\n"
        f"spending the GPU time twice. The head-set CSVs here were verified against\n"
        f"retrieve_random_k(num_layers, num_heads, topk, seed={seed}).\n\n"
        f"The layer-matched arm (randomlayer-s*) is NOT mirrored: its draw depends on\n"
        f"the real per-layer ATP histogram, which differs per localization.\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--families", nargs="*", default=list(FAMILIES))
    p.add_argument("--verify", action="store_true",
                   help="check mirrored head sets against retrieve_random_k")
    p.add_argument("--force", action="store_true",
                   help="re-copy even if the destination already looks complete")
    args = p.parse_args()

    arm = f"random-s{args.seed}"
    rc = 0
    for fam in args.families:
        _, src_task, dst_task = FAMILIES[fam]
        src = RESULTS / args.model / src_task / arm
        dst = RESULTS / args.model / dst_task / arm

        if not src.is_dir():
            print(f"SKIP {fam}: source tree missing ({src})")
            rc = 1
            continue

        n_src = len(list(src.glob("*/*/eval/*_gen.json")))
        n_dst = len(list(dst.glob("*/*/eval/*_gen.json"))) if dst.is_dir() else 0
        if n_dst >= n_src and not args.force:
            print(f"SKIP {fam}: destination already has {n_dst} gen files (source {n_src})")
            continue

        print(f"mirror {fam}: {src_task} -> {dst_task} ({n_src} gen files)")
        if dst.is_dir() and args.force:
            shutil.rmtree(dst)
        shutil.copytree(src, dst, dirs_exist_ok=True)
        retag(dst, src_task, dst_task, args.seed)

        if args.verify:
            checked = verify_draw(args.model, dst_task, dst, args.seed)
            print(f"  verified {checked} head-set CSVs against retrieve_random_k")
        print(f"  {len(list(dst.glob('*/*/eval/*_gen.json')))} gen files now under {dst}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
