import numpy as np


def evaluate_clustering_error(X, centroids, labels):
    """
    Return common clustering error metrics for a given assignment.
    """
    X = np.asarray(X, dtype=float)
    centroids = np.asarray(centroids, dtype=float)
    labels = np.asarray(labels, dtype=int)

    residuals = X - centroids[labels]
    point_dist = np.linalg.norm(residuals, axis=1)
    point_sq_dist = np.sum(residuals * residuals, axis=1)

    return {
        "total_l2_error": float(np.sum(point_dist)),
        "mean_l2_error": float(np.mean(point_dist)),
        "sse": float(np.sum(point_sq_dist)),
        "mse": float(np.mean(point_sq_dist)),
    }
