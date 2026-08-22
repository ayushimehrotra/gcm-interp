#!/usr/bin/env python3
"""Central configuration and data loading for every figure in this directory.

Single source of truth for the things all the figure scripts need: where the two
checkouts live, which models and tasks are in scope, the palette and Palatino
setup, and the two loaders (judged accuracy, raw attribution fields).

Import it rather than redefining any of this:

    import figdata as D
    acc = D.harvest("long")            # (model, task, arm) -> {budget: accuracy}
    fields = D.load_fields(model)      # (task, arm) -> per-(layer, unit) field

WHY THIS EXISTS
---------------
The config used to be spread across the figure scripts themselves, and the two
loaders lived in whichever script happened to need them first -- harvest in
figures.py, load_fields in position_count_mechanism.py. Anything that wanted a
loader imported the *script* that owned it, so deleting a script deleted a loader
out from under its consumers, and a constant edited in one file was silently
stale in another. FINDINGS.md 10.4 records what that cost: a private copy of the
accuracy harvest in manifold_vs_accuracy.py never received the
results_pipeline_with_answers root, silently dropped 6 of 21 cells, and turned
r = +0.238 into a marginal-looking r = +0.462.

THE TWO ROOTS
-------------
umang's results_pipeline and results_pipeline_with_answers are DIFFERENT
CONDITIONS (steering evaluated without the answer in context vs with it), not two
runs of one. harvest takes the highest-priority root that HAS a given cell and
never merges across roots: a cell's two arms must come from the same condition or
the paired comparison is meaningless. In practice only persona falls through to
with-answers -- results_pipeline has no extraversion runs at all.
"""
import csv
import json
import re
import statistics as st
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt


HERE = Path(__file__).parent

AYUSHI = Path("/home/ubuntu/gcm-interp")

UMANG = Path("/home/ubuntu/gcm-interp-umang")

ACC_ROOTS = {"ayushi": [AYUSHI / "judge-evals" / "accuracy"],
             "umang": [UMANG / "results_pipeline",
                       UMANG / "results_pipeline_with_answers"]}

TASKNAME = {"verse": "verse", "paragraph": "summarization", "female": "bias",
            "lying": "factual recall", "extraversion": "persona"}

FN_RE = re.compile(
    r"^(?P<N>\d+)_(?P<reps>random|targeted)_(?P<method>steer|mean)_topk_"
    r"(?P<topk>[\d.]+)_gen_accuracy_"
    r"(?P<metric>w_rf|wo_rf|comb|flu|rel|judge_3|judge_4|judge_5|mcqa)"
    r"\.json\.accuracy\.json$")

MODE_RE = re.compile(r"^(?P<task>.+)-(?P<mode>long|single)$")

MET = {("ayushi", "long"): "w_rf", ("ayushi", "single"): "w_rf",
       ("umang", "long"): "comb", ("umang", "single"): "w_rf"}

OLD = {"Llama-2-13b-chat-hf", "SOLAR-10.7B-Instruct-v1.0", "vicuna-13b-v1.5",
       "phi-4"}

MODELS = ["gemma-3-12b-it", "Qwen1.5-14B-Chat", "Qwen1.5-32B-Chat",
          "OLMo-2-1124-13B-DPO"]

GRID_MODELS = MODELS + ["Falcon3-10B-Instruct"]

EXCLUDED = [m for m in GRID_MODELS if m not in MODELS]

SHORT = {"gemma-3-12b-it": "Gemma-3-12B", "Qwen1.5-14B-Chat": "Qwen1.5-14B",
         "Qwen1.5-32B-Chat": "Qwen1.5-32B", "OLMo-2-1124-13B-DPO": "OLMo-2-13B",
         "Falcon3-10B-Instruct": "Falcon3-10B"}

TASKS = ["verse", "summarization", "bias", "factual recall", "persona"]

KS = [0.01, 0.03, 0.05, 0.07, 0.09, 0.1, 0.5, 1.0]

INK = "#2b2f36"


# NOTE on LF: the palette's original teal was #1f6f8b, which fails the OKLCH
# chroma floor (0.086 < 0.10) and reads gray -- sRGB cannot reach the floor at
# that hue below L~0.53, so this is the same hue lightened until it passes.
# Checked with the data-viz validator: lightness band, chroma floor, CVD
# separation (protan dE 16.0 vs ST), normal-vision floor (dE 25.1) and contrast
# all PASS. Now that the palette is defined once, every figure picks this up
# consistently; set LF = "#1f6f8b" here to restore the old colour everywhere.
LF = "#0680a4"         # free-form (see note above)

ST = "#c25a34"

HAIR = "#b9bec6"

BAND = "#7fb3c4"

CTRL = "#8d939c"

def strip_mode(s):
    m = MODE_RE.match(s)
    return (m.group("task"), m.group("mode")) if m else (s, None)

def harvest(eval_mode="long", include_random=False):
    """(model, task, arm) -> {k: best accuracy over steering factors}.

    arm is "long" / "single" for the two localizations and, when
    include_random is set, "random" (uniform) / "randomlayer" (depth-matched)
    for the control arms. Every arm is aggregated the same way: the best
    steering factor at each budget, so the four curves are comparable.
    """
    g = defaultdict(lambda: defaultdict(dict))
    for repo, roots in ACC_ROOTS.items():
      for prio, root in enumerate(roots):
        if not root.exists():
            continue
        for p in root.rglob("*.accuracy.json"):
              parts = p.relative_to(root).parts
              m = FN_RE.match(parts[-1])
              if not m or m.group("method") != "steer":
                  continue
              ctrl = next((c for c in parts if c.startswith("random")), None)
              if ctrl is None and "atp" not in parts:
                  continue
              if ctrl is not None and not include_random:
                  continue
              # atp trees are written with reps=targeted, control trees with
              # reps=random (eval_runner.py keeps the filename stem so the judge
              # regex still matches), so the expected value depends on the tree
              if m.group("reps") != ("random" if ctrl else "targeted"):
                  continue
              srcbase = evald = steerd = model = None
              for i, c in enumerate(parts[:-1]):
                  if c.startswith("from_") and "_to_" in c:
                      srcbase, model = c, parts[i - 1] if i else None
                  elif c.endswith("_eval"):
                      evald = c
                  elif c.endswith("_steer"):
                      steerd = c
              if not (srcbase and evald and steerd and model):
                  continue
              if srcbase.endswith("_old") or model in OLD:
                  continue
              sb = re.match(r"^from_(?P<src>.+?)_to_(?P<base>.+)$", srcbase)
              if not sb:
                  continue
              task, arm = strip_mode(sb.group("src"))
              task = TASKNAME.get(task)
              _, ev = strip_mode(evald[:-5])
              _, stm = strip_mode(steerd[:-6])
              if task is None or arm is None or ev != eval_mode or stm != ev:
                  continue
              if MET.get((repo, ev)) != m.group("metric"):
                  continue
              try:
                  v = json.load(open(p)).get("q1")
              except Exception:
                  continue
              if v is not None:
                  # Depth-matched random is drawn from a SPECIFIC arm's layer
                  # histogram, so there is one per arm and they differ a lot
                  # (mean |gap| up to 0.225, max 0.62). Key it by arm so each
                  # localization is compared against its own control. Uniform
                  # random uses a fixed seed and is arm-independent, so the two
                  # copies are the same intervention and stay pooled.
                  cname = None if ctrl is None else ctrl.split("-")[0]
                  key = (arm if cname is None else
                         f"{cname}_{arm}" if cname == "randomlayer" else cname)
                  # keep the localization tree in the bucket: control arms exist
                  # under BOTH trees, and max-over-N must be taken within a tree
                  # before averaging across trees, never over the two mixed
                  g[(model, task, key)][float(m.group("topk"))]\
                      .setdefault((prio, srcbase), []).append(float(v))
    out = {}
    for key, d in g.items():
        agg = {}
        for kk, bucket in d.items():
            top = min(p for p, _ in bucket)          # highest-priority root
            # One rule for every arm: the best steering factor at this budget.
            # Applied identically to free-form, single-token, uniform random and
            # depth-matched random, so the four curves are directly comparable.
            # (An earlier version averaged over N for the control arms only,
            # which depressed them -- Gemma verse at k=1.0 read 0.514 where the
            # best-N value is 0.800.) Where a control exists under both
            # localization trees, this takes the better of the two.
            agg[kk] = max(v for (p, _), vs in bucket.items() if p == top
                          for v in vs)
        out[key] = agg
    return out

def boot_ci(rows, n=10000, seed=0):
    if len(rows) < 2:
        return float("nan"), float("nan")
    r = np.random.default_rng(seed)
    a = np.asarray(rows, float)
    m = np.sort(r.choice(a, (n, len(a))).mean(1))
    return float(m[int(.025 * n)]), float(m[int(.975 * n)])

def style():
    for f in sorted((HERE / "fonts").glob("*.otf")):
        fm.fontManager.addfont(str(f))
    fam = ("TeX Gyre Pagella"
           if any("Pagella" in f.name for f in fm.fontManager.ttflist)
           else "DejaVu Serif")
    plt.rcParams.update({
        "font.family": "serif", "font.serif": [fam, "DejaVu Serif"],
        "mathtext.fontset": "custom", "mathtext.rm": fam,
        "mathtext.it": f"{fam}:italic", "mathtext.bf": f"{fam}:bold",
        # mathtext.cal defaults to a cursive family that is not installed and
        # emits a findfont warning on every render
        "mathtext.cal": fam, "mathtext.sf": fam, "mathtext.tt": "DejaVu Sans Mono",
        "font.size": 9,
        "axes.linewidth": 0.7, "axes.edgecolor": INK, "axes.labelcolor": INK,
        "text.color": INK, "xtick.color": INK, "ytick.color": INK,
        "xtick.direction": "out", "ytick.direction": "out",
        "xtick.major.width": 0.7, "ytick.major.width": 0.7,
        "xtick.major.size": 3, "ytick.major.size": 3,
        "xtick.minor.size": 0, "ytick.minor.size": 0,
        "legend.frameon": False, "figure.dpi": 200,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
    })
    return fam

def titled(ax, title, subtitle=None, fontsize=10.5):
    """Centred title with an optional smaller subtitle above the axes.

    Meta text (sample size, what the band means, which evaluation) goes here
    rather than inside the axes, where it collides with data as soon as the
    curve shape changes.
    """
    # pad must grow with the subtitle's line count, or a second line runs into
    # the title
    lines = 0 if not subtitle else subtitle.count("\n") + 1
    ax.set_title(title, fontsize=fontsize, pad=9 + 9 * lines)
    if subtitle:
        ax.text(0.5, 1.015, subtitle, transform=ax.transAxes, ha="center",
                va="bottom", fontsize=8.5, color=INK, alpha=0.62,
                linespacing=1.45)

def bare(ax):
    """Minimum ink: two spines, no grid, no box."""
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(False)
    ax.set_axisbelow(True)



# ------------------------------------------------- raw attribution fields
# Rescued from position_count_mechanism.py, which was deleted. figures_analysis'
# depth profile is the surviving consumer. A cell can exist in both checkouts,
# so OWNER decides which is authoritative per task; within a checkout the newest
# file by git date wins.
AYUSHI_ROOT = AYUSHI
REPOS = {"ayushi": AYUSHI, "umang": UMANG}
OWNER = {"verse": "ayushi", "summarization": "ayushi",
         "bias": "umang", "factual recall": "umang", "persona": "umang"}
FIELD_TASKNAME = {"verse": "verse", "paragraph": "summarization",
                  "female": "bias", "lying": "factual recall",
                  "extraversion": "persona"}
FROM_RE = re.compile(
    r"^from_(?P<src>.+?)-(?P<loc>long|single)_to_(?P<base>.+?)(?P<old>_old)?$")


def _read_field(path):
    rows = [(int(r["layer"]), int(r["neuron"]), float(r["value"]))
            for r in csv.DictReader(open(path))]
    if not rows:
        return None
    nl = max(r[0] for r in rows) + 1
    nu = max(r[1] for r in rows) + 1
    if len(rows) != nl * nu:
        return None
    a = np.zeros((nl, nu))
    for l, u, v in rows:
        a[l, u] = v
    return a

def _git_date(root, path):
    out = subprocess.run(["git", "-C", str(root), "log", "-1", "--format=%ad",
                          "--date=short", "--", str(Path(path).relative_to(root))],
                         capture_output=True, text=True).stdout.strip()
    return out or "0000-00-00"

def load_fields(model):
    found = defaultdict(dict)
    for repo, root in REPOS.items():
        if not root.exists():
            continue
        for p in root.rglob("numerator_1_targeted_1.0.csv"):
            parts = p.parts
            frm = next((c for c in parts
                        if c.startswith("from_") and "_to_" in c), None)
            if frm is None:
                continue
            m = FROM_RE.match(frm)
            if not m or m.group("old") or parts[parts.index(frm) - 1] != model:
                continue
            task = FIELD_TASKNAME.get(m.group("src"))
            if task is None:
                continue
            f = _read_field(p)
            if f is not None:
                found[(task, m.group("loc"))][p] = (repo, root, f)
    out = {}
    for (task, loc), cand in found.items():
        # OWNER breaks ties when a cell exists in both checkouts; when it exists
        # in only one, use that one rather than discarding the cell
        pref = {p: x for p, x in cand.items() if x[0] == OWNER.get(task)}
        use = pref or cand
        best = max(use, key=lambda p: (_git_date(use[p][1], p), str(p)))
        out[(task, loc)] = use[best][2]
    return out

def top_units(A, frac):
    """Signed descending -- eval/logits_handler.py:92 uses flat.topk."""
    n = max(1, int(round(frac * A.size)))
    idx = np.argsort(-A, axis=None, kind="stable")[:n]
    return {(int(i // A.shape[1]), int(i % A.shape[1])) for i in idx}
