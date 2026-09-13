"""Physics constraint checks for RUL predictions."""

import numpy as np
from typing import Dict


def monotonicity_violation_rate(
    predictions: np.ndarray,
    epsilon: float = 1.0,
) -> Dict:
    """
    Measure how often predicted RUL increases over time
    (violates the physical fact that engines do not heal).

    Answers RQ3.
    """
    preds = np.asarray(predictions, dtype=float)

    if preds.ndim == 1:
        diffs = np.diff(preds)
        violations = diffs > epsilon
        n_violations = int(np.sum(violations))
        total = len(diffs)
    elif preds.ndim == 2:
        all_diffs = []
        for row in preds:
            all_diffs.append(np.diff(row))
        diffs = np.concatenate(all_diffs)
        violations = diffs > epsilon
        n_violations = int(np.sum(violations))
        total = len(violations)
    else:
        raise ValueError("predictions must be 1D or 2D array")

    rate = n_violations / total if total > 0 else 0.0

    return {
        "violation_rate": float(rate),
        "n_violations": n_violations,
        "n_total_steps": total,
        "max_violation": float(np.max(diffs)) if total > 0 else 0.0,
        "mean_violation": float(np.mean(diffs[violations])) if n_violations > 0 else 0.0,
    }
