import numpy as np


def aggregate_window_evidence(window_similarity, log_energy):
    """Aggregate per-window similarity and log-energy into a fixed feature vector.

    Returns a float32 array of shape (10,) ordered as:
    [sim_mean, sim_max, sim_min, sim_std, top_two_mean,
     frac_ge_mean, longest_run_ge_mean,
     energy_mean, energy_max, energy_std].
    """
    sim = np.asarray(window_similarity, dtype=np.float64)
    energy = np.asarray(log_energy, dtype=np.float64)

    if sim.ndim != 1 or energy.ndim != 1:
        raise ValueError("window_similarity and log_energy must be 1-dimensional")
    if sim.size == 0 or energy.size == 0:
        raise ValueError("window_similarity and log_energy must be non-empty")
    if sim.size != energy.size:
        raise ValueError("window_similarity and log_energy must have equal length")
    if not (np.isfinite(sim).all() and np.isfinite(energy).all()):
        raise ValueError("window_similarity and log_energy must be finite")

    sim_mean = sim.mean()
    sim_max = sim.max()
    sim_min = sim.min()
    sim_std = sim.std()

    k = min(2, sim.size)
    top_two_mean = np.sort(sim)[-k:].mean()

    above = sim >= sim_mean
    frac_ge_mean = above.mean()

    if above.any():
        padded = np.concatenate(([False], above, [False]))
        diff = np.diff(padded.astype(np.int8))
        run_starts = np.flatnonzero(diff == 1)
        run_ends = np.flatnonzero(diff == -1)
        longest_run = int((run_ends - run_starts).max())
    else:
        longest_run = 0

    energy_mean = energy.mean()
    energy_max = energy.max()
    energy_std = energy.std()

    return np.array(
        [
            sim_mean,
            sim_max,
            sim_min,
            sim_std,
            top_two_mean,
            frac_ge_mean,
            longest_run,
            energy_mean,
            energy_max,
            energy_std,
        ],
        dtype=np.float32,
    )


def _validate_binary_calibration(labels, probabilities):
    """Validate shared preconditions for calibration metrics.

    Requires non-empty, 1-D, equal-length arrays of finite binary labels in
    {0, 1} containing both classes, with finite probabilities in [0, 1].
    """
    labels_arr = np.asarray(labels, dtype=np.float64)
    probs_arr = np.asarray(probabilities, dtype=np.float64)

    if labels_arr.ndim != 1 or probs_arr.ndim != 1:
        raise ValueError("labels and probabilities must be 1-dimensional")
    if labels_arr.size == 0 or probs_arr.size == 0:
        raise ValueError("labels and probabilities must be non-empty")
    if labels_arr.size != probs_arr.size:
        raise ValueError("labels and probabilities must have equal length")
    if not (np.isfinite(labels_arr).all() and np.isfinite(probs_arr).all()):
        raise ValueError("labels and probabilities must be finite")
    if not ((labels_arr == 0.0) | (labels_arr == 1.0)).all():
        raise ValueError("labels must be binary (0 or 1)")
    if not ((labels_arr == 0.0).any() and (labels_arr == 1.0).any()):
        raise ValueError("labels must contain both classes")
    if (probs_arr < 0.0).any() or (probs_arr > 1.0).any():
        raise ValueError("probabilities must be in [0, 1]")

    return labels_arr, probs_arr


def brier_score(labels, probabilities):
    """Mean squared probability error for binary labels."""
    labels_arr, probs_arr = _validate_binary_calibration(labels, probabilities)
    return float(np.mean((probs_arr - labels_arr) ** 2))


def expected_calibration_error(labels, probabilities, bins=10):
    """Expected calibration error with fixed equal-width bins over [0, 1].

    Probability 1 is included in the final bin. The score is the weighted
    mean absolute difference between bucket accuracy and bucket confidence.
    """
    labels_arr, probs_arr = _validate_binary_calibration(labels, probabilities)

    if isinstance(bins, bool) or not isinstance(bins, (int, np.integer)) or bins <= 0:
        raise ValueError("bins must be a positive integer")

    n_bins = int(bins)
    n = probs_arr.size
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.clip(np.digitize(probs_arr, edges, right=False) - 1, 0, n_bins - 1)

    ece = 0.0
    for b in range(n_bins):
        mask = bin_idx == b
        count = int(mask.sum())
        if count == 0:
            continue
        accuracy = labels_arr[mask].mean()
        confidence = probs_arr[mask].mean()
        ece += (count / n) * abs(accuracy - confidence)

    return float(ece)


def _validate_bucket_keys(bucket_keys, expected_length):
    """Validate a 1-D, equal-length, non-empty sequence of non-empty string keys."""
    keys_arr = np.asarray(bucket_keys, dtype=object)
    if keys_arr.ndim != 1:
        raise ValueError("bucket_keys must be 1-dimensional")
    if keys_arr.size == 0:
        raise ValueError("bucket_keys must be non-empty")
    if keys_arr.size != expected_length:
        raise ValueError("bucket_keys must have equal length to labels")
    for key in keys_arr.tolist():
        if not isinstance(key, str) or not key:
            raise ValueError("bucket_keys must be non-empty strings")
    return keys_arr


def presence_proxy_metrics(labels, probabilities, threshold):
    """Presence-proxy policy metrics for a single acceptance threshold.

    A probability >= threshold accepts. Returns false-reject rate, reject
    accuracy, false-accept rate, target-accept rate, and presence-proxy utility.
    """
    labels_arr, probs_arr = _validate_binary_calibration(labels, probabilities)
    threshold = float(threshold)
    if not np.isfinite(threshold) or threshold < 0.0 or threshold > 1.0:
        raise ValueError("threshold must be a finite value in [0, 1]")

    accept = probs_arr >= threshold
    positives = labels_arr == 1.0
    negatives = labels_arr == 0.0

    num_positives = int(positives.sum())
    num_negatives = int(negatives.sum())

    false_rejects = int((positives & ~accept).sum())
    false_accepts = int((negatives & accept).sum())
    target_accepts = int((positives & accept).sum())
    true_rejects = int((negatives & ~accept).sum())

    false_reject_rate = false_rejects / num_positives
    false_accept_rate = false_accepts / num_negatives
    target_accept_rate = target_accepts / num_positives
    reject_accuracy = true_rejects / num_negatives
    presence_proxy_utility = ((1.0 - false_reject_rate) + reject_accuracy) / 2.0

    return {
        "false_reject_rate": float(false_reject_rate),
        "reject_accuracy": float(reject_accuracy),
        "false_accept_rate": float(false_accept_rate),
        "target_accept_rate": float(target_accept_rate),
        "presence_proxy_utility": float(presence_proxy_utility),
    }


def bucketed_presence_proxy_metrics(labels, probabilities, bucket_keys, threshold):
    """Per-bucket presence-proxy metrics keyed by lexical bucket name."""
    labels_arr, probs_arr = _validate_binary_calibration(labels, probabilities)
    keys_arr = _validate_bucket_keys(bucket_keys, labels_arr.size)
    threshold = float(threshold)

    result = {}
    for name in sorted(set(keys_arr.tolist())):
        mask = keys_arr == name
        result[name] = presence_proxy_metrics(labels_arr[mask], probs_arr[mask], threshold)
    return result


def select_public_validation_threshold(
    labels, probabilities, bucket_keys, max_false_reject_rate=0.10, min_reject_accuracy=0.85
):
    """Select the best public-validation threshold subject to global constraints.

    Candidates are {0, 1} plus each unique probability. Eligibility uses global
    metrics; eligible candidates are ranked by greatest worst-bucket utility,
    then greatest global utility, then higher threshold.
    """
    labels_arr, probs_arr = _validate_binary_calibration(labels, probabilities)
    keys_arr = _validate_bucket_keys(bucket_keys, labels_arr.size)

    if (
        not np.isfinite(max_false_reject_rate)
        or max_false_reject_rate < 0.0
        or max_false_reject_rate > 1.0
    ):
        raise ValueError("max_false_reject_rate must be a finite value in [0, 1]")
    if (
        not np.isfinite(min_reject_accuracy)
        or min_reject_accuracy < 0.0
        or min_reject_accuracy > 1.0
    ):
        raise ValueError("min_reject_accuracy must be a finite value in [0, 1]")

    candidates = sorted({0.0, 1.0} | set(float(p) for p in probs_arr.tolist()))

    eligible = []
    for candidate in candidates:
        metrics = presence_proxy_metrics(labels_arr, probs_arr, candidate)
        if (
            metrics["false_reject_rate"] <= max_false_reject_rate
            and metrics["reject_accuracy"] >= min_reject_accuracy
        ):
            bucket_metrics = bucketed_presence_proxy_metrics(
                labels_arr, probs_arr, keys_arr, candidate
            )
            worst_bucket_utility = min(
                row["presence_proxy_utility"] for row in bucket_metrics.values()
            )
            eligible.append((candidate, metrics, bucket_metrics, worst_bucket_utility))

    if not eligible:
        raise ValueError("no eligible public validation threshold")

    eligible.sort(
        key=lambda entry: (
            entry[3],
            entry[1]["presence_proxy_utility"],
            entry[0],
        ),
        reverse=True,
    )
    threshold, metrics, bucket_metrics, worst_bucket_utility = eligible[0]

    return {
        "threshold": float(threshold),
        "metrics": metrics,
        "bucket_metrics": bucket_metrics,
        "worst_bucket_utility": float(worst_bucket_utility),
        "threshold_source": "public_validation",
    }


import torch


def build_public_proxy_features(
    enrollment, mixture, windows, log_energy
) -> tuple[np.ndarray, np.ndarray]:
    """Build global and per-frame public-proxy features from embeddings.

    Returns a float32 global vector of shape (10,) and a float32 frame array
    of shape (window_count, 2) whose column 0 is the enrollment-to-window
    cosine and column 1 is the supplied log-energy.
    """
    enrollment_arr = np.asarray(enrollment, dtype=np.float64)
    mixture_arr = np.asarray(mixture, dtype=np.float64)
    windows_arr = np.asarray(windows, dtype=np.float64)
    energy_arr = np.asarray(log_energy, dtype=np.float64)

    if enrollment_arr.ndim != 1:
        raise ValueError("enrollment must be 1-dimensional")
    if enrollment_arr.size == 0:
        raise ValueError("enrollment must be non-empty")
    if mixture_arr.ndim != 1:
        raise ValueError("mixture must be 1-dimensional")
    if mixture_arr.size == 0:
        raise ValueError("mixture must be non-empty")
    if mixture_arr.size != enrollment_arr.size:
        raise ValueError("enrollment and mixture must have equal length")
    if windows_arr.ndim != 2:
        raise ValueError("windows must be 2-dimensional")
    if windows_arr.shape[1] != enrollment_arr.size:
        raise ValueError("windows must share the enrollment embedding dimension")
    if energy_arr.ndim != 1:
        raise ValueError("log_energy must be 1-dimensional")
    if energy_arr.size != windows_arr.shape[0]:
        raise ValueError("log_energy must have one value per window")
    if not (
        np.isfinite(enrollment_arr).all()
        and np.isfinite(mixture_arr).all()
        and np.isfinite(windows_arr).all()
        and np.isfinite(energy_arr).all()
    ):
        raise ValueError("inputs must be finite")

    enrollment_norm = np.linalg.norm(enrollment_arr)
    mixture_norm = np.linalg.norm(mixture_arr)
    window_norms = np.linalg.norm(windows_arr, axis=1)
    if enrollment_norm == 0.0:
        raise ValueError("enrollment must have non-zero norm")
    if mixture_norm == 0.0:
        raise ValueError("mixture must have non-zero norm")
    if (window_norms == 0.0).any():
        raise ValueError("window embeddings must have non-zero norm")

    enrollment_unit = enrollment_arr / enrollment_norm
    mixture_unit = mixture_arr / mixture_norm
    windows_unit = windows_arr / window_norms[:, None]

    frame_cosine = windows_unit @ enrollment_unit
    enrollment_mixture_cosine = float(mixture_unit @ enrollment_unit)

    frame_features = np.column_stack((frame_cosine, energy_arr)).astype(np.float32)

    global_features = aggregate_window_evidence(frame_cosine, energy_arr).astype(np.float32)
    global_features[0] = np.float32(enrollment_mixture_cosine)

    return global_features, frame_features


class GlobalPresenceCalibrator(torch.nn.Module):
    """Minimal logits-only presence calibrator over a global feature vector."""

    def __init__(self, input_dim):
        super().__init__()
        if isinstance(input_dim, bool) or not isinstance(input_dim, (int, np.integer)):
            raise ValueError("input_dim must be an integer")
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        self.input_dim = int(input_dim)
        self.linear = torch.nn.Linear(self.input_dim, 1)

    def forward(self, features):
        return self.linear(features)
