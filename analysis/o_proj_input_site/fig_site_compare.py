"""o_proj.input vs o_proj.output -- Qwen1.5-14B-Chat, verse, long-steer/long-eval.

Deliberately NOT written into the repo (scratchpad only, per request).
Palette: reference categorical slots 1 (blue) and 2 (orange), unmodified, plus
the documented blue<->red diverging pair with the #f0efec gray midpoint.
"""
import json, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

MODEL = sys.argv[1] if len(sys.argv)>1 else "Qwen1.5-14B-Chat"
ACC = Path(f"/home/ubuntu/gcm-interp/judge-evals/accuracy/{MODEL}")
KS = [0.01, 0.03, 0.05, 0.07, 0.09, 0.1, 0.5, 1.0]
NS = [1, 2, 4, 5, 6, 8, 10]
SITES = [("atp-o_proj_in", "o_proj.input  (per-head z)", "#2a78d6"),
         ("atp-o_proj_out", "o_proj.output (W_O-mixed)", "#eb6834")]
LOCS = [("verse-long", "long-form localization"), ("verse-single", "single-token localization")]

INK, INK2, MUTED, GRID, AXIS, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"

def acc(loc, method, N, k):
    method = "atp" if method == "atp-o_proj_out" else method
    p = (ACC / f"from_{loc}_to_prose" / method / "verse-long_eval" / "verse-long_steer"
         / f"{N}_targeted_steer_topk_{k}_gen_accuracy_w_rf.json.accuracy.json")
    return json.load(open(p))["q1"] if p.exists() else np.nan

grid = {(loc, m): np.array([[acc(loc, m, N, k) for k in KS] for N in NS])
        for loc, _ in LOCS for m, _, _ in SITES}

fig = plt.figure(figsize=(11.6, 8.4), facecolor=SURF)
gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.05], hspace=0.50, wspace=0.20,
                      left=0.075, right=0.965, top=0.775, bottom=0.155)

# ---- row 1: k-curves (max over N), small multiples by localization ----------
for j, (loc, loc_label) in enumerate(LOCS):
    ax = fig.add_subplot(gs[0, j], facecolor=SURF)
    x = np.arange(len(KS))
    used = {}                                  # peak x -> how many labels already placed there
    for m, s_label, colour in SITES:
        y = np.nanmax(grid[(loc, m)], axis=0)
        ax.plot(x, y, color=colour, lw=2, marker="o", ms=6.5, mew=1.6, mec=SURF,
                label=s_label, zorder=3, clip_on=False)
        i = int(np.nanargmax(y))
        if y[i] >= 0.15:                       # selective direct labels only
            # two series can peak at the same k; stack the labels instead of overprinting
            n_here = used.get(i, 0); used[i] = n_here + 1
            dy = 11 if n_here == 0 else -17
            ax.annotate(f"{y[i]:.2f}", (x[i], y[i]), textcoords="offset points",
                        xytext=(0, dy), ha="center", fontsize=10.5, color=colour, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels([f"{k:g}" for k in KS], fontsize=9.5, color=MUTED)
    ax.set_ylim(-0.03, 1.16); ax.set_yticks(np.arange(0, 1.01, 0.25))
    ax.set_yticklabels(["0", ".25", ".50", ".75", "1"], fontsize=9.5, color=MUTED)
    ax.set_xlabel("head budget  k", fontsize=10, color=INK2, labelpad=6)
    if j == 0:
        ax.set_ylabel("w_rf   (judge + fluency + relevance)", fontsize=10, color=INK2)
    ax.set_title(loc_label, fontsize=11.5, color=INK, pad=10, fontweight="bold", loc="left")
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"): ax.spines[sp].set_color(AXIS); ax.spines[sp].set_linewidth(1.0)
    ax.tick_params(length=0)
    if j == 0:
        handles, labels = ax.get_legend_handles_labels()

# ---- row 2: per-cell difference over the whole (N, k) sweep -----------------
div = LinearSegmentedColormap.from_list(
    "site_div", ["#8f2523", "#e34948", "#f0b3b2", "#f0efec", "#9ec5f4", "#2a78d6", "#0d366b"])
vmax = max(abs(np.nanmin(grid[(l, "atp-o_proj_in")] - grid[(l, "atp-o_proj_out")]))
           for l, _ in LOCS)
vmax = max(vmax, max(np.nanmax(grid[(l, "atp-o_proj_in")] - grid[(l, "atp-o_proj_out")])
                     for l, _ in LOCS))
norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

for j, (loc, loc_label) in enumerate(LOCS):
    ax = fig.add_subplot(gs[1, j], facecolor=SURF)
    d = grid[(loc, "atp-o_proj_in")] - grid[(loc, "atp-o_proj_out")]
    im = ax.imshow(d, cmap=div, norm=norm, aspect="auto", origin="lower")
    ax.set_xticks(range(len(KS))); ax.set_xticklabels([f"{k:g}" for k in KS], fontsize=9.5, color=MUTED)
    ax.set_yticks(range(len(NS))); ax.set_yticklabels(NS, fontsize=9.5, color=MUTED)
    ax.set_xlabel("head budget  k", fontsize=10, color=INK2, labelpad=6)
    if j == 0:
        ax.set_ylabel("steering factor  N", fontsize=10, color=INK2)
    ax.set_title(loc_label, fontsize=11.5, color=INK, pad=10, fontweight="bold", loc="left")
    ax.set_xticks(np.arange(-.5, len(KS), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(NS), 1), minor=True)
    ax.grid(which="minor", color=SURF, lw=2)          # 2px surface gap between cells
    ax.tick_params(which="both", length=0)
    for sp in ax.spines.values(): sp.set_visible(False)
    for a in range(len(NS)):                           # label only the decisive cells
        for b in range(len(KS)):
            if abs(d[a, b]) >= 0.30:
                ax.text(b, a, f"{d[a,b]:+.2f}", ha="center", va="center", fontsize=8.5,
                        color="#ffffff", fontweight="bold")

fig.legend(handles, labels, frameon=False, fontsize=10, labelcolor=INK2, ncol=2,
           handlelength=1.8, loc="upper left", bbox_to_anchor=(0.075, 0.868))

cax = fig.add_axes([0.075, 0.052, 0.89, 0.018])
cb = fig.colorbar(im, cax=cax, orientation="horizontal")
cb.set_label("per-cell difference in w_rf      ← o_proj.output higher      o_proj.input higher →",
             fontsize=9.5, color=INK2, labelpad=6)
cb.outline.set_visible(False); cb.ax.tick_params(length=0, labelsize=9, colors=MUTED)

fig.text(0.075, 0.968, "Steering the same heads at two sites: o_proj.input vs o_proj.output",
         fontsize=15, color=INK, fontweight="bold", ha="left")
fig.text(0.075, 0.930,
         f"{MODEL} · verse · long-form steering vector, long-form eval · 50 items/cell · "
         "top row collapses N by max, as the paper reports it",
         fontsize=10, color=INK2, ha="left")
fig.text(0.075, 0.902,
         "Dose is not comparable across sites: the steering vector is normalized in the space it is applied to, "
         "so a given N is a different perturbation at each site.",
         fontsize=9.5, color=MUTED, ha="left", style="italic")

out = Path("/tmp/claude-1000/-home-ubuntu-gcm-interp/a02cfc7e-2539-4a39-bb0e-ae627ca27e4b/scratchpad")
fig.savefig(out / f"site_compare_{MODEL}.png", dpi=200, facecolor=SURF)
fig.savefig(out / f"site_compare_{MODEL}.pdf", facecolor=SURF)
print("wrote", out / f"site_compare_{MODEL}.png")
for loc, _ in LOCS:
    d = grid[(loc, "atp-o_proj_in")] - grid[(loc, "atp-o_proj_out")]
    print(f"{loc:12} diff range {np.nanmin(d):+.2f}..{np.nanmax(d):+.2f}  "
          f"max-over-N curve in={np.nanmax(grid[(loc,'atp-o_proj_in')],axis=0).round(2)}")
