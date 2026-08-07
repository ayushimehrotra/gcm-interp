"""
qualitative_analysis_verse.py — Compare single-token vs long-form localization.

Generates an HTML report with two sections:

Section 1 — Per Top-K Comparison:
  For each topk value, shows side-by-side the post-intervention responses from
  single-token localization (from_verse-single_to_prose) vs long-form localization
  (from_verse-long_to_prose), using the best steering factor for each.

Section 2 — Best Overall Comparison:
  Shows the single best (topk, sf) configuration for single-token localization
  vs the single best (topk, sf) for long-form localization, side by side.

Usage:
    python qualitative_analysis_verse.py --model Qwen1.5-14B-Chat
    python qualitative_analysis_verse.py --model OLMo-2-1124-13B-DPO --eval long --steer long
    python qualitative_analysis_verse.py --model Qwen1.5-14B-Chat --task sycophancy
"""

import argparse
import html
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RM_INTERP_REPO = Path(os.path.dirname(os.path.abspath(__file__)))
WORKDIRS_ROOT = RM_INTERP_REPO / "judge-evals" / "workdirs"
ACCURACY_ROOT = RM_INTERP_REPO / "judge-evals" / "accuracy"
OUTPUT_DIR = RM_INTERP_REPO / "qualitative_reports"

STEERING_FACTORS = [10, 8, 6, 5, 4, 2, 1]
TOPK_VALUES = [0.01, 0.03, 0.05, 0.07, 0.09, 0.1, 0.5, 1.0]

TASK_CONFIG = {
    "verse": {
        "single_task": "from_verse-single_to_prose",
        "long_task": "from_verse-long_to_prose",
        "source_single": "verse-single",
        "source_long": "verse-long",
        "label": "Prose → Verse",
    },
    "sycophancy": {
        "single_task": "from_sycophancy-single_to_non-sycophantic",
        "long_task": "from_sycophancy-long_to_non-sycophantic",
        "source_single": "sycophancy-single",
        "source_long": "sycophancy-long",
        "label": "Sycophancy → Non-Sycophantic",
    },
}


def load_accuracy(filepath: str) -> float:
    with open(filepath, "r") as f:
        data = json.load(f)
    acc = data.get("gen", {}).get("q1", np.nan)
    if acc is np.nan or acc != acc:
        acc = data.get("q1", np.nan)
    return acc


def find_best_sf_per_topk(
    model: str, task: str, eval_variant: str, steer_variant: str
) -> dict[float, tuple[int, float]]:
    """
    For each topk, find the steering factor with the highest accuracy.
    Returns {topk: (best_sf, best_acc)}.
    """
    source = task.split("_to_")[0].split("from_")[1]
    breakup_source = source.split("-")[0]

    # single eval → token matching (wo_rf); long eval → w_rf
    rf_suffix = "wo_rf" if eval_variant == "single" else "w_rf"

    results = {}
    for topk in TOPK_VALUES:
        best_sf = None
        best_acc = -1.0
        for sf in STEERING_FACTORS:
            method_dir = (
                ACCURACY_ROOT / model / task / "atp"
                / f"{breakup_source}-{eval_variant}_eval"
                / f"{breakup_source}-{steer_variant}_steer"
            )
            filename = (
                f"{sf}_targeted_steer_topk_{topk}"
                f"_gen_accuracy_{rf_suffix}.json.accuracy.json"
            )
            fpath = method_dir / filename
            if not fpath.exists():
                continue
            try:
                acc = load_accuracy(str(fpath))
                if not np.isnan(acc) and acc > best_acc:
                    best_acc = acc
                    best_sf = sf
            except Exception:
                continue
        if best_sf is not None:
            results[topk] = (best_sf, best_acc)
    return results


def load_eval_csv(
    model: str, task: str, eval_variant: str, steer_variant: str,
    sf: int, topk: float
) -> pd.DataFrame | None:
    source = task.split("_to_")[0].split("from_")[1]
    breakup_source = source.split("-")[0]
    source_suffix = f"{breakup_source}-{eval_variant}"

    wd = (
        WORKDIRS_ROOT / model / task / "atp"
        / f"{breakup_source}-{eval_variant}_eval"
        / f"{breakup_source}-{steer_variant}_steer"
        / "eval"
        / f"{sf}_targeted_steer_{topk}_{source_suffix}"
    )
    csv_path = wd / "eval_output.csv"
    if not csv_path.exists():
        print(f"  [MISSING] {csv_path}", file=sys.stderr)
        return None
    return pd.read_csv(csv_path, keep_default_na=False, dtype=str)


def e(text: str) -> str:
    """HTML-escape and convert newlines to <br>."""
    return html.escape(str(text)).replace("\n", "<br>")


CSS = """
body { font-family: Arial, sans-serif; font-size: 13px; margin: 20px; color: #222;
       max-width: 1600px; margin: 0 auto; padding: 20px; }
h1   { font-size: 22px; margin-bottom: 4px; }
h2   { font-size: 18px; margin-top: 32px; border-bottom: 2px solid #333; padding-bottom: 4px; }
h3   { font-size: 15px; color: #444; margin-top: 24px; }
.meta { color: #555; font-size: 12px; margin-bottom: 16px; }
.summary-table { border-collapse: collapse; margin: 12px 0 20px; }
.summary-table th, .summary-table td { border: 1px solid #bbb; padding: 6px 12px; font-size: 12px; }
.summary-table th { background: #e8eaf6; }
.summary-table td { background: #fff; }
.best { background: #c8e6c9 !important; font-weight: bold; }
.section-nav { background: #f0f4ff; padding: 10px 14px; border-radius: 6px;
               margin: 16px 0; font-size: 13px; }
.section-nav a { margin-right: 16px; }
.example { border: 1px solid #ccc; border-radius: 6px; padding: 14px 16px;
           margin-bottom: 28px; background: #fafafa; }
.example h3 { margin: 0 0 6px 0; color: #333; font-size: 14px; }
.query-box { background: #f0f4ff; border-left: 4px solid #6688cc;
             padding: 8px 10px; margin: 8px 0; border-radius: 3px;
             white-space: pre-wrap; font-size: 12px; max-height: 120px; overflow-y: auto; }
.orig-box  { background: #fff8e1; border-left: 4px solid #f0a000;
             padding: 8px 10px; margin: 8px 0 12px; border-radius: 3px;
             white-space: pre-wrap; font-size: 12px; max-height: 100px; overflow-y: auto; }
table.comparison { border-collapse: collapse; width: 100%; margin-top: 6px; table-layout: fixed; }
table.comparison th, table.comparison td { border: 1px solid #bbb; padding: 8px 10px; vertical-align: top; }
table.comparison th { background: #e8eaf6; font-size: 13px; text-align: center; }
table.comparison th.single-loc { background: #e3f2fd; }
table.comparison th.long-loc   { background: #f3e5f5; }
td.response { background: #fff; white-space: pre-wrap; font-size: 12px;
              max-height: 300px; overflow-y: auto; width: 50%; }
td.missing  { background: #f5f5f5; color: #999; font-style: italic;
              text-align: center; vertical-align: middle; }
.config-badge { display: inline-block; background: #e0e0e0; border-radius: 3px;
                padding: 2px 6px; font-size: 11px; margin-left: 4px; }
.acc-badge   { display: inline-block; background: #c8e6c9; border-radius: 3px;
                padding: 2px 6px; font-size: 11px; margin-left: 4px; font-weight: bold; }
"""


def build_html(
    model: str,
    task_type: str,
    eval_variant: str,
    steer_variant: str,
    n_examples: int,
) -> str:
    cfg = TASK_CONFIG[task_type]
    single_task = cfg["single_task"]
    long_task = cfg["long_task"]

    # Find best SF per topk for both localization types
    print(f"Finding best SF per topk for single-token localization...")
    single_best = find_best_sf_per_topk(model, single_task, eval_variant, steer_variant)
    print(f"Finding best SF per topk for long-form localization...")
    long_best = find_best_sf_per_topk(model, long_task, eval_variant, steer_variant)

    # Find overall best (topk, sf) for each
    single_overall = max(single_best.items(), key=lambda x: x[1][1]) if single_best else None
    long_overall = max(long_best.items(), key=lambda x: x[1][1]) if long_best else None

    # Common topks that have data for both
    common_topks = sorted(set(single_best.keys()) & set(long_best.keys()))

    parts = []
    parts.append(f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Localization Comparison — {model} — {cfg['label']}</title>
  <style>{CSS}</style>
</head>
<body>
<h1>Single-Token vs Long-Form Localization Comparison</h1>
<div class="meta">
  Model: <b>{model}</b> &nbsp;|&nbsp;
  Task: <b>{cfg['label']}</b> &nbsp;|&nbsp;
  Eval: <b>{eval_variant}</b> &nbsp;|&nbsp;
  Steer: <b>{steer_variant}</b>
</div>
""")

    # Navigation
    parts.append('<div class="section-nav"><b>Jump to:</b> ')
    parts.append('<a href="#summary">Summary Table</a>')
    parts.append('<a href="#best-overall">Best Overall Comparison</a>')
    for topk in common_topks:
        parts.append(f'<a href="#topk-{topk}">topk={topk}</a>')
    parts.append('</div>')

    # Summary table
    parts.append('<h2 id="summary">Summary: Best Steering Factor & Accuracy per Top-K</h2>')
    parts.append('<table class="summary-table"><tr>')
    parts.append('<th>Top-K</th>')
    parts.append('<th>Single-Token Loc.<br>Best SF</th><th>Single-Token Loc.<br>Accuracy</th>')
    parts.append('<th>Long-Form Loc.<br>Best SF</th><th>Long-Form Loc.<br>Accuracy</th>')
    parts.append('<th>Better</th>')
    parts.append('</tr>')

    for topk in TOPK_VALUES:
        s_sf, s_acc = single_best.get(topk, (None, None))
        l_sf, l_acc = long_best.get(topk, (None, None))

        s_sf_str = str(s_sf) if s_sf is not None else "—"
        s_acc_str = f"{s_acc:.2f}" if s_acc is not None else "—"
        l_sf_str = str(l_sf) if l_sf is not None else "—"
        l_acc_str = f"{l_acc:.2f}" if l_acc is not None else "—"

        if s_acc is not None and l_acc is not None:
            if s_acc > l_acc:
                winner = "Single"
            elif l_acc > s_acc:
                winner = "Long"
            else:
                winner = "Tie"
        else:
            winner = "—"

        s_cls = ' class="best"' if winner == "Single" else ""
        l_cls = ' class="best"' if winner == "Long" else ""

        parts.append(f"<tr><td>{topk}</td>")
        parts.append(f"<td{s_cls}>{s_sf_str}</td><td{s_cls}>{s_acc_str}</td>")
        parts.append(f"<td{l_cls}>{l_sf_str}</td><td{l_cls}>{l_acc_str}</td>")
        parts.append(f"<td>{winner}</td></tr>")

    parts.append('</table>')

    # Section 2: Best overall comparison
    parts.append('<h2 id="best-overall">Best Overall Configuration Comparison</h2>')

    if single_overall and long_overall:
        s_topk, (s_sf, s_acc) = single_overall
        l_topk, (l_sf, l_acc) = long_overall

        parts.append(f'<p><b>Single-Token Localization best:</b> topk={s_topk}, sf={s_sf} '
                      f'<span class="acc-badge">acc={s_acc:.2f}</span></p>')
        parts.append(f'<p><b>Long-Form Localization best:</b> topk={l_topk}, sf={l_sf} '
                      f'<span class="acc-badge">acc={l_acc:.2f}</span></p>')

        single_df = load_eval_csv(model, single_task, eval_variant, steer_variant, s_sf, s_topk)
        long_df = load_eval_csv(model, long_task, eval_variant, steer_variant, l_sf, l_topk)
        ref_df = single_df if single_df is not None else long_df

        if ref_df is not None:
            n = min(n_examples, len(ref_df))
            for i in range(n):
                row0 = ref_df.iloc[i]
                query = str(row0.get("query", ""))
                orig = str(row0.get("original-response", ""))

                parts.append(f'<div class="example">')
                parts.append(f'<h3>Example {i + 1} / {n}</h3>')
                parts.append(f'<div class="query-box"><b>Prompt:</b><br>{e(query)}</div>')
                parts.append(f'<div class="orig-box"><b>Original response:</b><br>{e(orig)}</div>')

                parts.append('<table class="comparison"><tr>')
                parts.append(f'<th class="single-loc">Single-Token Localization'
                             f'<span class="config-badge">topk={s_topk} sf={s_sf}</span>'
                             f'<span class="acc-badge">acc={s_acc:.2f}</span></th>')
                parts.append(f'<th class="long-loc">Long-Form Localization'
                             f'<span class="config-badge">topk={l_topk} sf={l_sf}</span>'
                             f'<span class="acc-badge">acc={l_acc:.2f}</span></th>')
                parts.append('</tr><tr>')

                if single_df is not None and i < len(single_df):
                    resp = str(single_df.iloc[i].get("post-intervention-response", ""))
                    parts.append(f'<td class="response">{e(resp)}</td>')
                else:
                    parts.append('<td class="missing">N/A</td>')

                if long_df is not None and i < len(long_df):
                    resp = str(long_df.iloc[i].get("post-intervention-response", ""))
                    parts.append(f'<td class="response">{e(resp)}</td>')
                else:
                    parts.append('<td class="missing">N/A</td>')

                parts.append('</tr></table></div>')
    else:
        parts.append('<p>Insufficient data for best overall comparison.</p>')

    # Section 1: Per-topk comparison
    parts.append('<h2>Per Top-K Comparison</h2>')
    parts.append('<p>For each top-k, using the best steering factor for each localization type.</p>')

    for topk in common_topks:
        s_sf, s_acc = single_best[topk]
        l_sf, l_acc = long_best[topk]

        parts.append(f'<h3 id="topk-{topk}">Top-K = {topk}</h3>')
        parts.append(f'<p>Single-Token: sf={s_sf} '
                     f'<span class="acc-badge">acc={s_acc:.2f}</span> &nbsp;|&nbsp; '
                     f'Long-Form: sf={l_sf} '
                     f'<span class="acc-badge">acc={l_acc:.2f}</span></p>')

        single_df = load_eval_csv(model, single_task, eval_variant, steer_variant, s_sf, topk)
        long_df = load_eval_csv(model, long_task, eval_variant, steer_variant, l_sf, topk)
        ref_df = single_df if single_df is not None else long_df

        if ref_df is None:
            parts.append('<p><i>No data available for this topk.</i></p>')
            continue

        n = min(n_examples, len(ref_df))
        for i in range(n):
            row0 = ref_df.iloc[i]
            query = str(row0.get("query", ""))
            orig = str(row0.get("original-response", ""))

            parts.append(f'<div class="example">')
            parts.append(f'<h3>Example {i + 1} / {n} &nbsp; '
                         f'<span class="config-badge">topk={topk}</span></h3>')
            parts.append(f'<div class="query-box"><b>Prompt:</b><br>{e(query)}</div>')
            parts.append(f'<div class="orig-box"><b>Original response:</b><br>{e(orig)}</div>')

            parts.append('<table class="comparison"><tr>')
            parts.append(f'<th class="single-loc">Single-Token Localization'
                         f'<span class="config-badge">sf={s_sf}</span></th>')
            parts.append(f'<th class="long-loc">Long-Form Localization'
                         f'<span class="config-badge">sf={l_sf}</span></th>')
            parts.append('</tr><tr>')

            if single_df is not None and i < len(single_df):
                resp = str(single_df.iloc[i].get("post-intervention-response", ""))
                parts.append(f'<td class="response">{e(resp)}</td>')
            else:
                parts.append('<td class="missing">N/A</td>')

            if long_df is not None and i < len(long_df):
                resp = str(long_df.iloc[i].get("post-intervention-response", ""))
                parts.append(f'<td class="response">{e(resp)}</td>')
            else:
                parts.append('<td class="missing">N/A</td>')

            parts.append('</tr></table></div>')

    parts.append("</body></html>")
    return "\n".join(parts)


def main():
    parser = argparse.ArgumentParser(
        description="Compare single-token vs long-form localization qualitatively"
    )
    parser.add_argument("--model", default="Qwen1.5-14B-Chat")
    parser.add_argument("--task", default="verse", choices=list(TASK_CONFIG.keys()),
                        help="Task family (verse or sycophancy)")
    parser.add_argument("--eval", default="long", choices=["single", "long"],
                        help="Eval variant (which test set to evaluate on)")
    parser.add_argument("--steer", default="long", choices=["single", "long"],
                        help="Steer variant (which steering vectors to use)")
    parser.add_argument("--n", type=int, default=50,
                        help="Max number of examples to show per section")
    parser.add_argument("--out", default=None, help="Output HTML path")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out) if args.out else (
        OUTPUT_DIR
        / f"{args.model}_{args.task}_localization_comparison_eval-{args.eval}_steer-{args.steer}.html"
    )

    print(f"Building report: model={args.model}  task={args.task}  "
          f"eval={args.eval}  steer={args.steer}")
    html_content = build_html(args.model, args.task, args.eval, args.steer, args.n)
    out_path.write_text(html_content, encoding="utf-8")
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
