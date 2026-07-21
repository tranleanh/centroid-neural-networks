import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

from centroid_neural_networks import centroid_neural_net_original_with_history
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
    centroids, labels, history = centroid_neural_net_original_with_history(
        X, n_clusters=num_clusters, max_iteration=100, epsilon=0.05
    )
    runtime = time.time() - t0

    movement = np.asarray(history["movement_per_epoch"], dtype=float)
    error = evaluate_clustering_error(X, centroids, labels)

    print(
        f"CentNN: runtime={runtime:.4f}s, epochs={len(movement)}, "
        f"splits={len(history['split_epochs'])}, movement_sum={movement.sum():.4f}, "
        f"total_l2_error={error['total_l2_error']:.4f}, mean_l2_error={error['mean_l2_error']:.4f}, "
        f"sse={error['sse']:.4f}, mse={error['mse']:.4f}"
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].scatter(X[:, 0], X[:, 1], c=labels, s=25, cmap="viridis", alpha=0.8)
    axes[0].scatter(centroids[:, 0], centroids[:, 1], c="red", s=130, marker="x")
    axes[0].set_title(f"CentNN Clustering\n{runtime:.3f}s, epochs={len(movement)}")
    axes[0].set_xlabel("X1")
    axes[0].set_ylabel("X2")

    axes[1].plot(np.arange(len(movement)), movement, marker="o", label="CentNN movement")
    first = True
    for ep in history["split_epochs"]:
        axes[1].axvline(
            ep,
            color="tab:blue",
            linestyle="--",
            alpha=0.25,
            label="CentNN split" if first else None,
        )
        first = False

    axes[1].set_title("CentNN Movement Over Epochs")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Total centroid movement")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plt.show()
