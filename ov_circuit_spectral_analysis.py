"""
OV-circuit eigendecomposition for the top-N heads of each ATP localization
(long-form vs single-token), to test "same manifold" purely from weights --
no forward pass, no stimulus set, so no dependence on which examples were
chosen (complements spectral_localization_analysis.py, which is
activation/data-driven).

Definitions (Anthropic "Mathematical Framework for Transformer Circuits"
convention):
  For head h in layer l: A = W_O_h (hidden_size x head_dim), the COLUMN slice
  of o_proj.weight [:, h*head_dim:(h+1)*head_dim] (o_proj.weight has shape
  (hidden_size, num_attention_heads*head_dim) since PyTorch nn.Linear stores
  (out_features, in_features)). B = W_V_h (head_dim x hidden_size), the ROW
  slice of v_proj.weight for head h's (possibly shared, under GQA) KV head.
  The full OV circuit is OV_h = A @ B (hidden_size x hidden_size, rank <=
  head_dim): the linear map applied to any residual-stream direction that
  head h attends to, before it's added back into the residual stream.

  Eigendecomposing the full hidden_size x hidden_size OV_h is wasteful since
  it's rank <= head_dim (128 here). Standard trick: the nonzero eigenvalues
  of A@B equal those of B@A (head_dim x head_dim, tiny). If v is an
  eigenvector of M = B@A with eigenvalue lambda, then w = A@v is an
  eigenvector of the FULL OV_h with the SAME eigenvalue lambda -- and w lives
  in the hidden_size-dim residual-stream space, which is the one basis
  shared across every layer/head in the model, so lifted eigenvectors from
  different heads are directly, meaningfully comparable.

  Eigenvalues of M can be complex (M is a product of two unrelated
  rectangular matrices, not symmetric). Per head we take the eigenvalue of
  largest magnitude as "the" top mode and its lifted eigenvector as the
  head's dominant output-space direction. A strongly positive real top
  eigenvalue is the classic "copying" signature in this framework; complex
  top eigenvalues indicate rotation/oscillation rather than pure
  amplification.

Weights are read directly from the cached safetensors shards (float32,
un-quantized) -- no GPU, no forward pass, and no exposure to the 4-bit
quantization used elsewhere in this pipeline for generation.
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from safetensors import safe_open
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

REPO = "/home/ubuntu/gcm-interp"
N_HEADS = 20

MODEL_HF_ID = {
    "Qwen1.5-14B-Chat": "Qwen--Qwen1.5-14B-Chat",
    "OLMo-2-1124-13B-DPO": "allenai--OLMo-2-1124-13B-DPO",
    "Qwen1.5-32B-Chat": "Qwen--Qwen1.5-32B-Chat",
}

LOCALIZATIONS = [
    ("from_verse-long_to_prose", "Long-Form Localization"),
    ("from_verse-single_to_prose", "Single-Token Localization"),
]


def find_snapshot_dir(model_name):
    pattern = f"/home/ubuntu/.cache/huggingface/hub/models--{MODEL_HF_ID[model_name]}/snapshots/*"
    dirs = glob.glob(pattern)
    assert len(dirs) >= 1, f"No cached snapshot for {model_name}"
    return dirs[0]


class WeightReader:
    """Lazily loads only the specific safetensors weight slices needed."""

    def __init__(self, model_name):
        snap = find_snapshot_dir(model_name)
        idx_path = os.path.join(snap, "model.safetensors.index.json")
        if os.path.exists(idx_path):
            with open(idx_path) as f:
                self.weight_map = json.load(f)["weight_map"]
        else:
            # single-shard model
            only_file = glob.glob(os.path.join(snap, "*.safetensors"))[0]
            self.weight_map = None
            self.single_file = only_file
        self.snap = snap
        self._open_files = {}

    def get(self, key):
        if self.weight_map is not None:
            fname = self.weight_map[key]
            path = os.path.join(self.snap, fname)
        else:
            path = self.single_file
        if path not in self._open_files:
            self._open_files[path] = safe_open(path, framework="pt")
        f = self._open_files[path]
        return f.get_tensor(key).to(torch.float32)


def load_config(model_name):
    with open(os.path.join(find_snapshot_dir(model_name), "config.json")) as f:
        return json.load(f)


def load_top_heads(model, loc_key, n):
    path = os.path.join(
        REPO, "results", model, loc_key, "atp",
        "verse-long_eval", "verse-long_steer", "eval", "numerator_1_targeted_1.0.csv",
    )
    df = pd.read_csv(path).sort_values("value", ascending=False).reset_index(drop=True)
    return list(zip(df.head(n)["layer"].tolist(), df.head(n)["neuron"].tolist()))


def head_ov_top_eigenpair(reader, layer, head, hidden_size, num_heads, num_kv_heads, head_dim):
    o_key = f"model.layers.{layer}.self_attn.o_proj.weight"  # (hidden_size, num_heads*head_dim)
    v_key = f"model.layers.{layer}.self_attn.v_proj.weight"  # (num_kv_heads*head_dim, hidden_size)
    W_O = reader.get(o_key).numpy()
    W_V = reader.get(v_key).numpy()

    A = W_O[:, head * head_dim:(head + 1) * head_dim]  # (hidden_size, head_dim)
    group = num_heads // num_kv_heads
    kv_head = head // group
    B = W_V[kv_head * head_dim:(kv_head + 1) * head_dim, :]  # (head_dim, hidden_size)

    M = B @ A  # (head_dim, head_dim) -- same nonzero eigenvalues as the full A@B
    eigvals, eigvecs = np.linalg.eig(M)
    order = np.argsort(-np.abs(eigvals))
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    top_eigval = eigvals[0]
    top_v = eigvecs[:, 0]
    w = A @ top_v  # lift to hidden_size-dim, eigenvector of the FULL OV circuit
    w = w / np.linalg.norm(w)

    frob_sq = float((np.abs(eigvals) ** 2).sum())
    concentration = float(np.abs(top_eigval) ** 2 / frob_sq) if frob_sq > 0 else np.nan

    return dict(
        eigval_real=float(top_eigval.real),
        eigval_imag=float(top_eigval.imag),
        eigval_mag=float(np.abs(top_eigval)),
        is_real=bool(np.abs(top_eigval.imag) < 0.05 * max(np.abs(top_eigval.real), 1e-8)),
        top_eigenvector_real_part=w.real,  # hidden_size-dim
        eigval_spectrum_mag=np.abs(eigvals),
        concentration=concentration,  # fraction of ||M||_F^2 in the top eigenvalue
    )


def principal_angles(A, B):
    Qa, _ = np.linalg.qr(A)
    Qb, _ = np.linalg.qr(B)
    _, s, _ = np.linalg.svd(Qa.T @ Qb)
    s = np.clip(s, -1, 1)
    return np.degrees(np.arccos(s))


def run(model_name, n_heads=N_HEADS):
    print(f"\n{'='*70}\n{model_name}  (OV-circuit eigendecomposition, top {n_heads} heads)\n{'='*70}")
    cfg = load_config(model_name)
    hidden_size = cfg["hidden_size"]
    num_heads = cfg["num_attention_heads"]
    num_kv_heads = cfg.get("num_key_value_heads", num_heads)
    head_dim = hidden_size // num_heads
    print(f"hidden_size={hidden_size} num_heads={num_heads} num_kv_heads={num_kv_heads} head_dim={head_dim}")

    reader = WeightReader(model_name)

    all_rows = []
    loc_data = {}
    for loc_key, loc_label in LOCALIZATIONS:
        heads = load_top_heads(model_name, loc_key, n_heads)
        per_head = []
        for layer, head in heads:
            res = head_ov_top_eigenpair(reader, layer, head, hidden_size, num_heads, num_kv_heads, head_dim)
            per_head.append(res)
            all_rows.append(dict(
                model=model_name, localization=loc_key, layer=layer, head=head,
                eigval_real=res["eigval_real"], eigval_imag=res["eigval_imag"],
                eigval_mag=res["eigval_mag"], is_real=res["is_real"],
                concentration=res["concentration"],
            ))
        loc_data[loc_key] = per_head
        mags = np.array([r["eigval_mag"] for r in per_head])
        reals = np.array([r["eigval_real"] for r in per_head])
        n_real = sum(r["is_real"] for r in per_head)
        n_pos_real = sum(r["is_real"] and r["eigval_real"] > 0 for r in per_head)
        print(f"\n--- {loc_label} ---")
        print(f"  top eigenvalue magnitude: mean={mags.mean():.3f} median={np.median(mags):.3f}")
        print(f"  top eigenvalue real part: mean={reals.mean():.3f}")
        print(f"  heads with (numerically) real top eigenvalue: {n_real}/{n_heads} "
              f"({n_pos_real} positive-real i.e. 'copying-like')")
        print(f"  mean concentration (top eigval^2 / ||M||_F^2): "
              f"{np.mean([r['concentration'] for r in per_head]):.3f}")

    # Cross-localization: subspace spanned by lifted top eigenvectors (real part)
    (loc1, label1), (loc2, label2) = LOCALIZATIONS
    E1 = np.stack([r["top_eigenvector_real_part"] for r in loc_data[loc1]], axis=1)  # (hidden_size, n_heads)
    E2 = np.stack([r["top_eigenvector_real_part"] for r in loc_data[loc2]], axis=1)

    print(f"\n--- Cross-localization comparison ---")
    for k in [1, 2, 5, min(10, n_heads), n_heads]:
        angles = principal_angles(E1[:, :k], E2[:, :k])
        print(f"Principal angles between top-{k} OV-eigenvector subspaces: "
              f"mean={angles.mean():.1f} deg, min={angles.min():.1f} deg "
              f"(mean cos={np.mean(np.cos(np.radians(angles))):.3f})")

    # Best-match cosine similarity: for each loc1 head, its best-aligned loc2 head
    cos_sim = np.abs(E1.T @ E2)  # (n_heads, n_heads), E columns are unit vectors
    best_match_1to2 = cos_sim.max(axis=1)
    best_match_2to1 = cos_sim.max(axis=0)
    print(f"Best 1-to-1 |cosine similarity| of individual top-eigenvector directions: "
          f"{label1}->{label2} mean={best_match_1to2.mean():.3f}, "
          f"{label2}->{label1} mean={best_match_2to1.mean():.3f}")
    # random baseline for cosine similarity of two random unit vectors in hidden_size dims
    rng = np.random.default_rng(0)
    rand_vecs_a = rng.normal(size=(hidden_size, 2000))
    rand_vecs_a /= np.linalg.norm(rand_vecs_a, axis=0, keepdims=True)
    rand_vecs_b = rng.normal(size=(hidden_size, 2000))
    rand_vecs_b /= np.linalg.norm(rand_vecs_b, axis=0, keepdims=True)
    rand_cos = np.abs((rand_vecs_a * rand_vecs_b).sum(axis=0))
    print(f"  random-baseline |cosine similarity| between two random unit vectors in "
          f"{hidden_size}-dim space: mean={rand_cos.mean():.4f}, 99th pct={np.percentile(rand_cos, 99):.4f}")

    # ---- Save table ----
    df = pd.DataFrame(all_rows)
    out_dir = os.path.join(REPO, "results", model_name, "jaccard_plots")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"ov_circuit_eigenanalysis_top{n_heads}.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved table: {csv_path}")

    # ---- Save plot ----
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    ax = axes[0]
    for loc_key, loc_label in LOCALIZATIONS:
        sub = df[df.localization == loc_key]
        ax.scatter(sub.eigval_real, sub.eigval_imag, label=loc_label, alpha=0.75, s=40)
    ax.axhline(0, color="gray", lw=0.5)
    ax.axvline(0, color="gray", lw=0.5)
    ax.set_xlabel("Re(top eigenvalue)")
    ax.set_ylabel("Im(top eigenvalue)")
    ax.set_title(f"{model_name}: top-eigenvalue of each head's OV circuit\n(top {n_heads} heads per localization)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ks = [1, 2, 3, 5, 7, 10, min(15, n_heads), n_heads]
    ks = sorted(set(k for k in ks if k <= n_heads))
    mean_angles = [principal_angles(E1[:, :k], E2[:, :k]).mean() for k in ks]
    ax.plot(ks, mean_angles, marker="o")
    ax.axhline(90, color="gray", linestyle="--", lw=1, label="90 deg = orthogonal")
    ax.set_xlabel("Subspace dimension k (top-k OV eigenvectors)")
    ax.set_ylabel("Mean principal angle (degrees)")
    ax.set_title("Subspace alignment: long-form vs single-token\nOV-eigenvector subspaces")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[2]
    im = ax.imshow(cos_sim, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax.set_xlabel(f"{label2} head index (rank order)")
    ax.set_ylabel(f"{label1} head index (rank order)")
    ax.set_title("|cosine similarity| between individual\ntop OV-eigenvector directions")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(f"{model_name}: OV-circuit eigendecomposition, long-form vs single-token localization (weights only, no forward pass)", y=1.04, fontsize=12)
    fig.tight_layout()
    png_path = os.path.join(out_dir, f"ov_circuit_eigenanalysis_top{n_heads}.png")
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {png_path}")

    return df


if __name__ == "__main__":
    models = sys.argv[1:] if len(sys.argv) > 1 else ["Qwen1.5-14B-Chat", "OLMo-2-1124-13B-DPO", "Qwen1.5-32B-Chat"]
    for m in models:
        run(m)
