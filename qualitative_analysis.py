"""
Qualitative analysis viewer.

Generates an HTML file with 50 tables — one per test example.
Each table is 2×2:
  rows    = localization task  (long / single)
  columns = steering vectors   (long / single)

Each cell shows the post-intervention response from the sycophancy-long_eval workdir
(full-text LLM response).  The original (pre-steering) response and the user query
are shown above each table for context.

Usage:
    python qualitative_analysis.py --model Qwen1.5-14B-Chat --sf 8 --topk 0.03
    python qualitative_analysis.py --model OLMo-2-1124-13B-DPO --sf 2 --topk 0.05 --out my_report.html
"""

import argparse
import html
import os
import sys
from pathlib import Path

import pandas as pd

WORKDIRS_ROOT = Path("/home/ubuntu/gcm-interp/judge-evals/workdirs")
OUTPUT_DIR    = Path("/home/ubuntu/gcm-interp/qualitative_reports")

TASKS = {
    "long":   "from_sycophancy-long_to_non-sycophantic",
    "single": "from_sycophancy-single_to_non-sycophantic",
}
STEERS = {
    "long":   "sycophancy-long_steer",
    "single": "sycophancy-single_steer",
}
EVAL_SUBDIR   = "sycophancy-long_eval"   # always use long-eval for full-text responses

SOURCE_SUFFIX = {
    "long":   "sycophancy-long",
    "single": "sycophancy-long",  # single-task gen files also use sycophancy-long test set
}


def find_workdir(model: str, task_key: str, steer_key: str, sf: int, topk: str) -> Path:
    wd = (
        WORKDIRS_ROOT
        / model
        / TASKS[task_key]
        / "atp"
        / EVAL_SUBDIR
        / STEERS[steer_key]
        / "eval"
        / f"{sf}_targeted_steer_{topk}_{SOURCE_SUFFIX[task_key]}"
    )
    return wd


def load_df(model: str, task_key: str, steer_key: str, sf: int, topk: str) -> pd.DataFrame | None:
    wd = find_workdir(model, task_key, steer_key, sf, topk)
    csv = wd / "eval_output.csv"
    if not csv.exists():
        print(f"  [MISSING] {csv}", file=sys.stderr)
        return None
    return pd.read_csv(csv, keep_default_na=False, dtype=str)


def e(text: str) -> str:
    """HTML-escape and convert newlines to <br>."""
    return html.escape(str(text)).replace("\n", "<br>")


CSS = """
body { font-family: Arial, sans-serif; font-size: 13px; margin: 20px; color: #222; }
h1   { font-size: 18px; margin-bottom: 4px; }
.meta { color: #555; font-size: 12px; margin-bottom: 16px; }
.example { border: 1px solid #ccc; border-radius: 6px; padding: 14px 16px;
           margin-bottom: 28px; background: #fafafa; }
.example h2 { font-size: 15px; margin: 0 0 6px 0; color: #333; }
.query-box { background: #f0f4ff; border-left: 4px solid #6688cc;
             padding: 8px 10px; margin: 8px 0; border-radius: 3px;
             white-space: pre-wrap; font-size: 12px; max-height: 120px; overflow-y: auto; }
.orig-box  { background: #fff8e1; border-left: 4px solid #f0a000;
             padding: 8px 10px; margin: 8px 0 12px; border-radius: 3px;
             white-space: pre-wrap; font-size: 12px; max-height: 100px; overflow-y: auto; }
table { border-collapse: collapse; width: 100%; margin-top: 6px; }
th, td { border: 1px solid #bbb; padding: 8px 10px; vertical-align: top; }
th { background: #e8eaf6; font-size: 13px; text-align: center; }
td.row-header { background: #e8eaf6; font-weight: bold; width: 120px;
                text-align: center; vertical-align: middle; }
td.response { background: #fff; white-space: pre-wrap; font-size: 12px;
              max-height: 220px; overflow-y: auto; }
td.missing  { background: #f5f5f5; color: #999; font-style: italic;
              text-align: center; vertical-align: middle; }
"""


def build_html(model: str, sf: int, topk: str) -> str:
    # Load all 4 dataframes
    dfs: dict[tuple, pd.DataFrame | None] = {}
    for t in ("long", "single"):
        for s in ("long", "single"):
            dfs[(t, s)] = load_df(model, t, s, sf, topk)

    # Reference df for iteration (query / original-response come from long×long)
    ref_df = dfs.get(("long", "long"))
    if ref_df is None:
        for df in dfs.values():
            if df is not None:
                ref_df = df
                break
    if ref_df is None:
        raise RuntimeError("No data found for any combination.")

    n = len(ref_df)
    parts = []
    parts.append(f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Qualitative Analysis — {model}  sf={sf}  topk={topk}</title>
  <style>{CSS}</style>
</head>
<body>
<h1>Qualitative Analysis</h1>
<div class="meta">
  Model: <b>{model}</b> &nbsp;|&nbsp;
  Steering factor: <b>{sf}</b> &nbsp;|&nbsp;
  Top-K: <b>{topk}</b> &nbsp;|&nbsp;
  {n} examples
</div>
""")

    for i in range(n):
        row0 = ref_df.iloc[i]
        query    = str(row0.get("query", ""))
        orig     = str(row0.get("original-response", ""))

        parts.append(f'<div class="example">')
        parts.append(f'<h2>Example {i + 1} / {n}</h2>')
        parts.append(f'<div class="query-box"><b>Prompt:</b><br>{e(query)}</div>')
        parts.append(f'<div class="orig-box"><b>Original (sycophantic) response:</b><br>{e(orig)}</div>')

        # 2×2 table
        parts.append("""<table>
  <tr>
    <th></th>
    <th>Long Steer</th>
    <th>Single Steer</th>
  </tr>""")

        for task_key, task_label in [("long", "Long Localization"), ("single", "Single Localization")]:
            parts.append(f"  <tr>")
            parts.append(f'    <td class="row-header">{task_label}</td>')
            for steer_key in ("long", "single"):
                df = dfs.get((task_key, steer_key))
                if df is not None and i < len(df):
                    resp = str(df.iloc[i].get("post-intervention-response", ""))
                    parts.append(f'    <td class="response">{e(resp)}</td>')
                else:
                    parts.append(f'    <td class="missing">N/A</td>')
            parts.append("  </tr>")

        parts.append("</table>")
        parts.append("</div>")  # end .example

    parts.append("</body></html>")
    return "\n".join(parts)


def main():
    parser = argparse.ArgumentParser(description="Generate qualitative analysis HTML")
    parser.add_argument("--model",  default="Qwen1.5-14B-Chat")
    parser.add_argument("--sf",     type=int, default=8, help="Steering factor")
    parser.add_argument("--topk",   default="0.03")
    parser.add_argument("--out",    default=None, help="Output HTML path (auto-named if omitted)")
    args = parser.parse_args()

    topk_str = args.topk if "." in args.topk else f"{float(args.topk)}"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out) if args.out else (
        OUTPUT_DIR / f"{args.model}_sf{args.sf}_topk{topk_str}.html"
    )

    print(f"Building report: model={args.model}  sf={args.sf}  topk={topk_str}")
    html_content = build_html(args.model, args.sf, topk_str)
    out_path.write_text(html_content, encoding="utf-8")
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
