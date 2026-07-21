import numpy as np
from scipy.spatial.distance import cdist


def _validate_inputs(X):
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be a 2D array of shape (n_samples, n_features).")
    if X.shape[0] == 0:
        raise ValueError("X must contain at least one sample.")
    return X


def _split_delta(X, cluster_assignments, split_index, epsilon):
    """Build a feature-wise split offset; robust for high-dimensional data."""
    n_features = X.shape[1]
    delta = np.zeros(n_features, dtype=float)
    members = X[cluster_assignments == split_index]

    if members.shape[0] > 1:
        axis = int(np.argmax(np.var(members, axis=0)))
    else:
        axis = 0

    delta[axis] = float(epsilon)
    return delta


def _run_reassignment_epochs(X, w, assignments, counts, max_iteration):
    for _ in range(max_iteration):
        changes = 0

        for i, x in enumerate(X):
            distances = cdist([x], w, "euclidean")[0]
            current_cluster_index = int(np.argmin(distances))
            previous_cluster_index = int(assignments[i])

            if previous_cluster_index == current_cluster_index:
                continue

            prev_count = int(counts[previous_cluster_index])
            # Cannot apply loser correction when the old cluster has only 1 sample.
            if prev_count <= 1:
                continue

            curr_count = int(counts[current_cluster_index])
            w[current_cluster_index] = w[current_cluster_index] + (
                (x - w[current_cluster_index]) / (curr_count + 1)
            )
            w[previous_cluster_index] = w[previous_cluster_index] - (
                (x - w[previous_cluster_index]) / (prev_count - 1)
            )

            counts[current_cluster_index] += 1
            counts[previous_cluster_index] -= 1
            assignments[i] = current_cluster_index
            changes += 1

        if changes == 0:
            return True

    return False


# Centroid Neural Networks
def centroid_neural_net(X, n_clusters, max_iteration=100, epsilon=0.05):
    X = _validate_inputs(X)
    if n_clusters < 1:
        raise ValueError("n_clusters must be >= 1.")

    n_samples, n_features = X.shape
    if n_clusters == 1:
        return np.mean(X, axis=0, keepdims=True), np.zeros(n_samples, dtype=int).tolist()

    centroid_X = np.mean(X, axis=0)
    eps = float(epsilon)
    w = np.vstack([centroid_X + eps, centroid_X - eps]).astype(float, copy=False)

    assignments = np.empty(n_samples, dtype=int)
    counts = np.zeros(2, dtype=int)

    # EPOCH 0
    for i, x in enumerate(X):
        distances = cdist([x], w, "euclidean")[0]
        index = int(np.argmin(distances))
        w[index] = w[index] + ((x - w[index]) / (counts[index] + 1))
        assignments[i] = index
        counts[index] += 1

    # EPOCH 1+ with dynamic split
    for _ in range(max_iteration):
        converged = _run_reassignment_epochs(X, w, assignments, counts, max_iteration=1)

        if not converged:
            continue

        if w.shape[0] >= n_clusters:
            break

        all_error = np.full(w.shape[0], -np.inf, dtype=float)
        for i in range(w.shape[0]):
            if counts[i] <= 0:
                continue
            members = X[assignments == i]
            dists = cdist([w[i]], members, "euclidean")[0]
            all_error[i] = np.sum(dists)

        split_index = int(np.argmax(all_error))
        if not np.isfinite(all_error[split_index]):
            break

        old_centroid = w[split_index].copy()
        delta = _split_delta(X, assignments, split_index, eps)

        # Make splitting symmetric around the original centroid.
        w[split_index] = old_centroid - delta
        w = np.vstack([w, old_centroid + delta])
        counts = np.append(counts, 0)

    return w, assignments.tolist()


def centroid_neural_net_original_with_history(
    X, n_clusters, max_iteration=100, epsilon=0.05
):
    """
    Original split policy:
    - Reassign until no changes.
    - Then split the highest-error cluster.

    Returns centroids, labels, and history with:
    - movement_per_epoch
    - split_epochs
    """
    X = _validate_inputs(X)
    if n_clusters < 1:
        raise ValueError("n_clusters must be >= 1.")

    n_samples, _ = X.shape
    if n_clusters == 1:
        centroids = np.mean(X, axis=0, keepdims=True)
        labels = np.zeros(n_samples, dtype=int).tolist()
        history = {"movement_per_epoch": [0.0], "split_epochs": []}
        return centroids, labels, history

    centroid_X = np.mean(X, axis=0)
    eps = float(epsilon)
    w = np.vstack([centroid_X + eps, centroid_X - eps]).astype(float, copy=False)

    assignments = np.empty(n_samples, dtype=int)
    counts = np.zeros(2, dtype=int)
    movement_per_epoch = []
    split_epochs = []

    # Epoch 0
    w_before = w.copy()
    for i, x in enumerate(X):
        distances = cdist([x], w, "euclidean")[0]
        winner = int(np.argmin(distances))
        w[winner] = w[winner] + ((x - w[winner]) / (counts[winner] + 1))
        assignments[i] = winner
        counts[winner] += 1
    movement_per_epoch.append(float(np.linalg.norm(w - w_before, axis=1).sum()))

    # Epoch 1+
    for epoch in range(max_iteration):
        w_before = w.copy()
        changes = 0

        for i, x in enumerate(X):
            distances = cdist([x], w, "euclidean")[0]
            current_idx = int(np.argmin(distances))
            prev_idx = int(assignments[i])

            if current_idx == prev_idx:
                continue

            prev_count = int(counts[prev_idx])
            if prev_count <= 1:
                continue

            curr_count = int(counts[current_idx])
            w[current_idx] = w[current_idx] + ((x - w[current_idx]) / (curr_count + 1))
            w[prev_idx] = w[prev_idx] - ((x - w[prev_idx]) / (prev_count - 1))
            counts[current_idx] += 1
            counts[prev_idx] -= 1
            assignments[i] = current_idx
            changes += 1

        movement_per_epoch.append(float(np.linalg.norm(w - w_before, axis=1).sum()))

        if changes != 0:
            continue

        if w.shape[0] >= n_clusters:
            break

        all_error = np.full(w.shape[0], -np.inf, dtype=float)
        for idx in range(w.shape[0]):
            if counts[idx] <= 0:
                continue
            members = X[assignments == idx]
            dists = cdist([w[idx]], members, "euclidean")[0]
            all_error[idx] = np.sum(dists)

        split_idx = int(np.argmax(all_error))
        if not np.isfinite(all_error[split_idx]):
            break

        old = w[split_idx].copy()
        delta = _split_delta(X, assignments, split_idx, eps)
        w[split_idx] = old - delta
        w = np.vstack([w, old + delta])
        counts = np.append(counts, 0)
        split_epochs.append(epoch + 1)

    history = {"movement_per_epoch": movement_per_epoch, "split_epochs": split_epochs}
    return w, assignments.tolist(), history


# FastCentNN
def centroid_neural_net_with_entropy(
    X,
    n_clusters,
    max_iteration=100,
    epsilon=0.05,
    movement_threshold=0.05,
    movement_patience=3,
    use_relative_threshold=True,
    min_epochs_before_split=1,
    return_history=False,
):
    """
    Centroid Neural Network with entropy-like split trigger.

    Entropy here is approximated by total centroid movement per epoch.
    A split can be triggered before full convergence when movement remains
    below a threshold for `movement_patience` consecutive epochs.

    When `return_history=True`, history includes:
    - movement_per_epoch
    - stage_baseline_per_epoch
    - threshold_per_epoch
    - stage_baseline_updates
    - split_epochs
    - split_reasons
    """
    X = _validate_inputs(X)
    if n_clusters < 1:
        raise ValueError("n_clusters must be >= 1.")
    if movement_patience < 1:
        raise ValueError("movement_patience must be >= 1.")
    if min_epochs_before_split < 0:
        raise ValueError("min_epochs_before_split must be >= 0.")

    n_samples, _ = X.shape
    if n_clusters == 1:
        centroids = np.mean(X, axis=0, keepdims=True)
        labels = np.zeros(n_samples, dtype=int).tolist()
        if return_history:
            history = {
                "movement_per_epoch": [0.0],
                "stage_baseline_per_epoch": [0.0],
                "threshold_per_epoch": [0.0],
                "stage_baseline_updates": [],
                "split_epochs": [],
                "split_reasons": [],
            }
            return centroids, labels, history
        return centroids, labels

    centroid_X = np.mean(X, axis=0)
    eps = float(epsilon)
    w = np.vstack([centroid_X + eps, centroid_X - eps]).astype(float, copy=False)

    assignments = np.empty(n_samples, dtype=int)
    counts = np.zeros(2, dtype=int)

    movement_per_epoch = []
    stage_baseline_per_epoch = []
    threshold_per_epoch = []
    stage_baseline_updates = []
    split_epochs = []
    split_reasons = []

    # EPOCH 0
    w_before = w.copy()
    for i, x in enumerate(X):
        distances = cdist([x], w, "euclidean")[0]
        index = int(np.argmin(distances))
        w[index] = w[index] + ((x - w[index]) / (counts[index] + 1))
        assignments[i] = index
        counts[index] += 1

    moved_epoch0 = float(np.linalg.norm(w - w_before, axis=1).sum())
    movement_per_epoch.append(moved_epoch0)

    # Keep a non-zero, data-aware baseline floor so relative thresholds remain usable
    # even after very small pre-split movement values.
    data_scale = float(np.linalg.norm(np.std(X, axis=0)))
    stage_baseline_floor = max(1e-12, eps * np.sqrt(X.shape[1]), 1e-3 * data_scale)
    stage_baseline = max(moved_epoch0, stage_baseline_floor)
    if use_relative_threshold:
        threshold_epoch0 = movement_threshold * stage_baseline
    else:
        threshold_epoch0 = movement_threshold
    stage_baseline_per_epoch.append(stage_baseline)
    threshold_per_epoch.append(threshold_epoch0)
    low_movement_streak = 0
    epochs_in_stage = 0

    # EPOCH 1+
    for epoch in range(max_iteration):
        w_before = w.copy()
        changes = 0

        for i, x in enumerate(X):
            distances = cdist([x], w, "euclidean")[0]
            current_cluster_index = int(np.argmin(distances))
            previous_cluster_index = int(assignments[i])

            if previous_cluster_index == current_cluster_index:
                continue

            prev_count = int(counts[previous_cluster_index])
            if prev_count <= 1:
                continue

            curr_count = int(counts[current_cluster_index])
            w[current_cluster_index] = w[current_cluster_index] + (
                (x - w[current_cluster_index]) / (curr_count + 1)
            )
            w[previous_cluster_index] = w[previous_cluster_index] - (
                (x - w[previous_cluster_index]) / (prev_count - 1)
            )

            counts[current_cluster_index] += 1
            counts[previous_cluster_index] -= 1
            assignments[i] = current_cluster_index
            changes += 1

        moved = float(np.linalg.norm(w - w_before, axis=1).sum())
        movement_per_epoch.append(moved)
        epochs_in_stage += 1

        if use_relative_threshold:
            threshold_value = movement_threshold * stage_baseline
        else:
            threshold_value = movement_threshold
        stage_baseline_per_epoch.append(stage_baseline)
        threshold_per_epoch.append(threshold_value)

        if moved <= threshold_value:
            low_movement_streak += 1
        else:
            low_movement_streak = 0

        should_split_early = (
            epochs_in_stage >= min_epochs_before_split
            and low_movement_streak >= movement_patience
        )
        should_split = (changes == 0) or should_split_early

        if w.shape[0] < n_clusters and should_split:
            all_error = np.full(w.shape[0], -np.inf, dtype=float)
            for i in range(w.shape[0]):
                if counts[i] <= 0:
                    continue
                members = X[assignments == i]
                dists = cdist([w[i]], members, "euclidean")[0]
                all_error[i] = np.sum(dists)

            split_index = int(np.argmax(all_error))
            if np.isfinite(all_error[split_index]):
                old_centroid = w[split_index].copy()
                delta = _split_delta(X, assignments, split_index, eps)
                w[split_index] = old_centroid - delta
                w = np.vstack([w, old_centroid + delta])
                counts = np.append(counts, 0)

                split_movement = float(2.0 * np.linalg.norm(delta))
                if should_split_early and changes == 0:
                    split_reason = "converged+early"
                elif should_split_early:
                    split_reason = "early"
                else:
                    split_reason = "converged"

                split_epochs.append(epoch + 1)
                split_reasons.append(split_reason)
                # Use post-split perturbation as new stage baseline.
                baseline_before = stage_baseline
                stage_baseline = max(split_movement, moved, stage_baseline_floor)
                stage_baseline_updates.append(
                    {
                        "epoch": int(epoch + 1),
                        "reason": split_reason,
                        "baseline_before": float(baseline_before),
                        "baseline_after": float(stage_baseline),
                        "split_movement": float(split_movement),
                    }
                )
                low_movement_streak = 0
                epochs_in_stage = 0
                continue

        if changes == 0 and w.shape[0] >= n_clusters:
            break

    if return_history:
        history = {
            "movement_per_epoch": movement_per_epoch,
            "stage_baseline_per_epoch": stage_baseline_per_epoch,
            "threshold_per_epoch": threshold_per_epoch,
            "stage_baseline_updates": stage_baseline_updates,
            "split_epochs": split_epochs,
            "split_reasons": split_reasons,
        }
        return w, assignments.tolist(), history

    return w, assignments.tolist()


# Centroid Neural Networks with Initialized Weights
def centroid_neural_net_init_weights(X, init_weights, max_iteration=100):
    X = _validate_inputs(X)
    w = np.asarray(init_weights, dtype=float).copy()

    if w.ndim != 2:
        raise ValueError("init_weights must be 2D: (n_clusters, n_features).")
    if w.shape[1] != X.shape[1]:
        raise ValueError("init_weights feature dimension must match X.")
    if w.shape[0] == 0:
        raise ValueError("init_weights must contain at least one centroid.")

    n_samples = X.shape[0]
    assignments = np.empty(n_samples, dtype=int)
    counts = np.zeros(w.shape[0], dtype=int)

    for i, x in enumerate(X):
        distances = cdist([x], w, "euclidean")[0]
        index = int(np.argmin(distances))
        w[index] = w[index] + ((x - w[index]) / (counts[index] + 1))
        assignments[i] = index
        counts[index] += 1

    _run_reassignment_epochs(X, w, assignments, counts, max_iteration=max_iteration)
    return w, assignments.tolist()