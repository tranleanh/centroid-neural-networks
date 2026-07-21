import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

from centroid_neural_networks import (
    centroid_neural_net_original_with_history,
    centroid_neural_net_with_entropy,
)
from utils import evaluate_clustering_error


if __name__ == "__main__":
    num_clusters = 10
    X, _ = make_blobs(
        n_samples=1000,
        centers=num_clusters,
        n_features=2,
        cluster_std=2,
        random_state=7,
    )

    t0 = time.time()
    c_org, l_org, h_org = centroid_neural_net_original_with_history(
        X, n_clusters=num_clusters, max_iteration=100, epsilon=0.05
    )
    dt_org = time.time() - t0

    t1 = time.time()
    c_ent, l_ent, h_ent = centroid_neural_net_with_entropy(
        X,
        n_clusters=num_clusters,
        max_iteration=100,
        epsilon=0.05,
        movement_threshold=2,           # threshold to control the split
        movement_patience=2,            # patience to control the split
        use_relative_threshold=True,    # enable (True) / disable (False) stage-relative threshold
        min_epochs_before_split=1,
        return_history=True,
    )
    dt_ent = time.time() - t1

    m_org = np.asarray(h_org["movement_per_epoch"], dtype=float)
    m_ent = np.asarray(h_ent["movement_per_epoch"], dtype=float)
    b_ent = np.asarray(h_ent["stage_baseline_per_epoch"], dtype=float)
    t_ent = np.asarray(h_ent["threshold_per_epoch"], dtype=float)
    e_org = evaluate_clustering_error(X, c_org, l_org)
    e_ent = evaluate_clustering_error(X, c_ent, l_ent)

    print(
        f"CentNN: runtime={dt_org:.4f}s, epochs={len(m_org)}, splits={len(h_org['split_epochs'])}, "
        f"movement_sum={m_org.sum():.4f}, total_l2_error={e_org['total_l2_error']:.4f}, "
        f"mean_l2_error={e_org['mean_l2_error']:.4f}, sse={e_org['sse']:.4f}, mse={e_org['mse']:.4f}"
    )
    print(
        f"FastCentNN: runtime={dt_ent:.4f}s, epochs={len(m_ent)}, splits={len(h_ent['split_epochs'])}, "
        f"movement_sum={m_ent.sum():.4f}, total_l2_error={e_ent['total_l2_error']:.4f}, "
        f"mean_l2_error={e_ent['mean_l2_error']:.4f}, sse={e_ent['sse']:.4f}, mse={e_ent['mse']:.4f}"
    )
    print(
        "FastCentNN stage baseline per epoch:",
        np.array2string(b_ent, precision=4, separator=", "),
    )
    print(
        "FastCentNN threshold per epoch:",
        np.array2string(t_ent, precision=4, separator=", "),
    )
    print("FastCentNN stage baseline updates:", h_ent["stage_baseline_updates"])

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.ravel()

    axes[0].scatter(X[:, 0], X[:, 1], c=l_org, s=25, cmap="viridis", alpha=0.8)
    axes[0].scatter(c_org[:, 0], c_org[:, 1], c="red", s=130, marker="x")
    axes[0].set_title(
        "CentNN\n"
        f"{dt_org:.3f}s, epochs={len(m_org)}, SSE={e_org['sse']:.2f}"
    )
    axes[0].set_xlabel("X1")
    axes[0].set_ylabel("X2")

    axes[1].scatter(X[:, 0], X[:, 1], c=l_ent, s=25, cmap="viridis", alpha=0.8)
    axes[1].scatter(c_ent[:, 0], c_ent[:, 1], c="red", s=130, marker="x")
    axes[1].set_title(
        "FastCentNN\n"
        f"{dt_ent:.3f}s, epochs={len(m_ent)}, SSE={e_ent['sse']:.2f}"
    )
    axes[1].set_xlabel("X1")
    axes[1].set_ylabel("X2")

    axes[2].plot(np.arange(len(m_org)), m_org, marker="o", label="CentNN movement")
    axes[2].plot(np.arange(len(m_ent)), m_ent, marker="s", label="FastCentNN movement")
    axes[2].plot(
        np.arange(len(t_ent)),
        t_ent,
        linestyle="--",
        linewidth=2.0,
        color="tab:red",
        label="FastCentNN threshold",
    )

    first = True
    for ep in h_org["split_epochs"]:
        axes[2].axvline(ep, color="tab:blue", linestyle="--", alpha=0.25, label="CentNN split" if first else None)
        first = False

    first = True
    for ep in h_ent["split_epochs"]:
        axes[2].axvline(ep, color="tab:orange", linestyle="--", alpha=0.25, label="FastCentNN split" if first else None)
        first = False

    axes[2].set_title("Movement Comparison")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Total centroid movement")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    method_names = [
        "CentNN",
        "FastCentNN",
    ]
    sse_values = [e_org["sse"], e_ent["sse"]]
    axes[3].bar(method_names, sse_values, color=["tab:blue", "tab:orange"])
    axes[3].set_title("SSE Comparison")
    axes[3].set_ylabel("SSE")
    axes[3].tick_params(axis="x", rotation=20)
    axes[3].grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.show()
