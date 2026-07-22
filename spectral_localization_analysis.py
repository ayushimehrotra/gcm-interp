"""
Spectral (eigen-)decomposition of attention-head activations for the top-N
heads identified by each ATP localization (long-form vs single-token), to
test whether the two localizations pick out heads that encode the
prose/verse concept in a similar geometric ("same manifold") way, as
opposed to merely picking different-but-equally-valid heads, or heads that
don't track the concept at all.

Data:
  - Activations: `head_activations_<model>.pt`, produced by
    extract_head_activations.py. For each of 50 held-out questions, asked
    once "Respond in prose..." (label 0) and once "Respond in verse..."
    (label 1) -- a shared, common 100-stimulus set for BOTH localizations,
    so any geometric difference we find is attributable to which heads were
    selected, not to different underlying prompts. Activation = the
    self_attn.o_proj OUTPUT at the last prompt token (right before
    generation begins) -- the same extraction point and moment
    (last_token, pre-generation) used by eval/activations.py's
    steering_reps_cache to build the actual steering vectors, and the same
    per-head slicing convention (contiguous head_dim-sized chunks of the
    o_proj output) used by eval/logits_handler.py's einops.reduce when
    aggregating ATP attributions into per-head scores.
  - Head ranking: numerator_1_targeted_1.0.csv per localization (verified
    identical across eval/steer combos, and verified to be exactly what
    determines the topk%-of-heads selection used in every steering run).

Method:
  1. For each localization, take its top-N heads (by ATP attribution value).
  2. Build a (100 examples x N*head_dim) matrix by concatenating the N
     heads' activation vectors per example, mean-center columns.
  3. PCA via eigendecomposition of the (100x100) Gram matrix (cheap, exact,
     and directly gives per-example PC scores -- the right tool when
     n_examples << n_features).
  4. Eigenvalue spectrum (variance explained, participation ratio) per
     localization.
  5. VALIDITY CHECK: correlate each localization's PC1 scores with the true
     prose/verse label. If this is weak, the top-N heads for that
     localization aren't actually tracking the concept in this activation
     space, and the "manifold" comparison below is not meaningful.
  6. Cross-localization comparison:
       a. Correlation between the two localizations' PC1 scores across the
          100 SHARED examples (same stimuli, different head-sets).
       b. Principal angles between the top-k PC subspaces (k=1..5) of the
          two localizations -- both score matrices live in the same
          100-dim "example space" regardless of head-set identity, so this
          is a fair, dimension-agnostic subspace-alignment measure.
"""
import os
import sys

import numpy as np
import pandas as pd
import torch
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
ACT_DIR = "/tmp/claude-1000/-home-ubuntu/9b93cea2-f73c-49f8-b00c-32437fcbbd12/scratchpad"
N_HEADS = 20

LOCALIZATIONS = [
    ("from_verse-long_to_prose", "Long-Form Localization"),
    ("from_verse-single_to_prose", "Single-Token Localization"),
]


def load_top_heads(model, loc_key, n):
    path = os.path.join(
        REPO, "results", model, loc_key, "atp",
        "verse-long_eval", "verse-long_steer", "eval", "numerator_1_targeted_1.0.csv",
    )
    df = pd.read_csv(path).sort_values("value", ascending=False).reset_index(drop=True)
    return list(zip(df.head(n)["layer"].tolist(), df.head(n)["neuron"].tolist()))


def build_matrix(activations, head_list, head_dim):
    # activations: (n_examples, num_layers, hidden_size)
    n = activations.shape[0]
    cols = []
    for layer, head in head_list:
        cols.append(activations[:, layer, head * head_dim:(head + 1) * head_dim])
    X = torch.cat(cols, dim=1).numpy()  # (n_examples, n_heads*head_dim)
    assert X.shape == (n, len(head_list) * head_dim)
    return X


def pca_via_gram(X, k=10):
    """Return (eigenvalues[:k], scores[:, :k]) via eigendecomposition of the
    (n x n) Gram matrix of the centered data -- exact PCA, cheap when
    n_examples << n_features."""
    Xc = X - X.mean(axis=0, keepdims=True)
    n = Xc.shape[0]
    G = Xc @ Xc.T  # (n, n)
    eigvals, eigvecs = np.linalg.eigh(G)  # ascending
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    eigvals = np.clip(eigvals, 0, None) / (n - 1)  # covariance eigenvalues
    # PC scores (each example's projection on PC_j) = eigvecs[:, j] * sqrt(eigval_j * (n-1))
    # (standard kernel-PCA score recovery: G v = lambda_raw v, score = v * sqrt(lambda_raw))
    raw_eigvals = eigvals * (n - 1)
    scores = eigvecs[:, :k] * np.sqrt(np.clip(raw_eigvals[:k], 1e-12, None))
    return eigvals[:k], scores, eigvals  # also return the FULL eigenvalue spectrum


def participation_ratio(eigvals):
    eigvals = np.clip(eigvals, 0, None)
    s1 = eigvals.sum()
    s2 = (eigvals ** 2).sum()
    return (s1 ** 2) / s2 if s2 > 0 else np.nan


def principal_angles(A, B):
    """Principal angles (degrees) between the column spaces of A and B
    (both n x k, columns need not be orthonormal -- orthonormalize via QR)."""
    Qa, _ = np.linalg.qr(A)
    Qb, _ = np.linalg.qr(B)
    _, s, _ = np.linalg.svd(Qa.T @ Qb)
    s = np.clip(s, -1, 1)
    return np.degrees(np.arccos(s))


def run(model, n_heads=N_HEADS):
    print(f"\n{'='*70}\n{model}  (top {n_heads} heads per localization)\n{'='*70}")
    cache = torch.load(os.path.join(ACT_DIR, f"head_activations_{model}.pt"))
    activations = cache["activations"]
    labels = cache["labels"].numpy()
    head_dim = cache["head_dim"]
    n_examples = activations.shape[0]
    print(f"Activations: {tuple(activations.shape)}, labels: {labels.shape} "
          f"({(labels==0).sum()} prose / {(labels==1).sum()} verse), head_dim={head_dim}")

    results = {}
    for loc_key, loc_label in LOCALIZATIONS:
        heads = load_top_heads(model, loc_key, n_heads)
        X = build_matrix(activations, heads, head_dim)
        eigvals_top, scores, full_spectrum = pca_via_gram(X, k=10)
        pr = participation_ratio(full_spectrum)
        # PCA eigenvectors are only defined up to sign; fix PC1's sign so it
        # always correlates positively with the verse label, so reported
        # correlations/plots are consistently oriented across localizations
        # and choices of N (an arbitrary sign flip would otherwise make two
        # equally-valid PC1s look "anti-correlated").
        if np.corrcoef(scores[:, 0], labels)[0, 1] < 0:
            scores[:, 0] *= -1
        # validity check: does PC1 track the prose/verse label?
        pc1 = scores[:, 0]
        r_label = np.corrcoef(pc1, labels)[0, 1]
        # AUC-style check: is PC1 higher for one class than the other, robustly (rank-biserial)
        prose_scores = pc1[labels == 0]
        verse_scores = pc1[labels == 1]
        # simple separation measure: |mean diff| / pooled std
        pooled_std = np.sqrt((prose_scores.var() + verse_scores.var()) / 2)
        cohend = (verse_scores.mean() - prose_scores.mean()) / pooled_std if pooled_std > 0 else np.nan

        results[loc_key] = dict(
            heads=heads, X=X, eigvals=full_spectrum, scores=scores,
            participation_ratio=pr, pc1_label_corr=r_label, pc1_cohend=cohend,
        )
        var_explained = full_spectrum[:5] / full_spectrum.sum() * 100
        print(f"\n--- {loc_label} ({loc_key}) ---")
        print(f"  top heads: {heads[:8]}{' ...' if len(heads) > 8 else ''}")
        print(f"  top-5 eigenvalues: {full_spectrum[:5]}")
        print(f"  % variance explained by PC1..PC5: {np.round(var_explained, 1)}")
        print(f"  participation ratio (effective dimensionality): {pr:.2f} / {len(full_spectrum)}")
        print(f"  PC1 vs prose/verse label: Pearson r = {r_label:.3f}  (Cohen's d = {cohend:.3f})")

    # Cross-localization comparison
    (loc1, label1), (loc2, label2) = LOCALIZATIONS
    scores1, scores2 = results[loc1]["scores"], results[loc2]["scores"]
    pc1_cross_r = np.corrcoef(scores1[:, 0], scores2[:, 0])[0, 1]
    print(f"\n--- Cross-localization comparison ---")
    print(f"PC1({label1}) vs PC1({label2}) correlation across the same 100 stimuli: r = {pc1_cross_r:.3f}")

    for k in [1, 2, 3, 5]:
        angles = principal_angles(scores1[:, :k], scores2[:, :k])
        print(f"Principal angles between top-{k} PC subspaces: {np.round(angles, 1)} degrees "
              f"(mean cos = {np.mean(np.cos(np.radians(angles))):.3f})")

    # Plot: eigenvalue spectra + PC1 scatter colored by label
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    ax = axes[0]
    for loc_key, loc_label in LOCALIZATIONS:
        spec = results[loc_key]["eigvals"][:15]
        ax.plot(range(1, len(spec) + 1), spec / spec.sum() * 100, marker="o", label=loc_label)
    ax.set_xlabel("Principal component")
    ax.set_ylabel("% variance explained")
    ax.set_title(f"{model}: eigenvalue spectra\n(top {n_heads} heads, concatenated)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    loc1_scores = results[loc1]["scores"]
    loc2_scores = results[loc2]["scores"]
    colors = np.where(labels == 1, "tab:blue", "tab:orange")
    ax.scatter(loc1_scores[:, 0], loc2_scores[:, 0], c=colors, alpha=0.7)
    ax.set_xlabel(f"PC1 ({label1})")
    ax.set_ylabel(f"PC1 ({label2})")
    ax.set_title(f"Cross-localization PC1 alignment\nr={pc1_cross_r:.3f} (blue=verse-asked, orange=prose-asked)")
    ax.grid(alpha=0.3)

    ax = axes[2]
    ax.scatter(range(n_examples), loc1_scores[:, 0], c=colors, alpha=0.7, label=label1, marker="o")
    ax.scatter(range(n_examples), loc2_scores[:, 0], c=colors, alpha=0.7, label=label2, marker="^")
    ax.set_xlabel("Example index")
    ax.set_ylabel("PC1 score")
    ax.set_title("PC1 score per example, by localization\n(color = true prose/verse label)")
    ax.grid(alpha=0.3)

    fig.suptitle(f"{model}: spectral comparison of long-form vs single-token localization top-{n_heads} heads", y=1.03)
    fig.tight_layout()
    out_dir = os.path.join(REPO, "results", model, "jaccard_plots")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"spectral_localization_comparison_top{n_heads}.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {out_path}")

    return results, pc1_cross_r


if __name__ == "__main__":
    models = sys.argv[1:] if len(sys.argv) > 1 else ["Qwen1.5-14B-Chat"]
    for m in models:
        run(m)
