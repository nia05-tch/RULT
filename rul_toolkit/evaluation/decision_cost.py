"""Decision-aware evaluation for RUL prediction."""

import numpy as np
from typing import Dict
from dataclasses import dataclass


@dataclass
class CostAnalysisResult:
    total_cost: float
    avg_cost: float
    late_cost: float
    early_cost: float
    cost_ratio: float
    n_late_errors: int
    n_early_errors: int


def expected_decision_cost(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    cost_late: float = 50.0,
    cost_early: float = 1.0,
) -> CostAnalysisResult:
    """
    Calculate asymmetric maintenance decision cost.

    Predicting too high (late maintenance) is expensive.
    Predicting too low (early maintenance) wastes remaining life.
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have same length")

    errors = y_pred - y_true

    late_errors = np.maximum(errors, 0)
    early_errors = np.maximum(-errors, 0)

    n_late = int(np.sum(late_errors > 0))
    n_early = int(np.sum(early_errors > 0))

    total_late_cost = float(np.sum(late_errors * cost_late))
    total_early_cost = float(np.sum(early_errors * cost_early))
    total_cost = total_late_cost + total_early_cost

    return CostAnalysisResult(
        total_cost=total_cost,
        avg_cost=total_cost / len(y_true),
        late_cost=total_late_cost,
        early_cost=total_early_cost,
        cost_ratio=cost_late / cost_early,
        n_late_errors=n_late,
        n_early_errors=n_early,
    )


def ranking_stability(
    y_true: np.ndarray,
    predictions_dict: Dict[str, np.ndarray],
    cost_ratios: list = None,
) -> Dict[float, list]:
    """
    Check if model ranking changes across different cost ratios.
    This directly answers RQ1.
    """
    if cost_ratios is None:
        cost_ratios = [5, 10, 20, 50, 100]

    rankings = {}

    for ratio in cost_ratios:
        costs = {}
        for name, y_pred in predictions_dict.items():
            result = expected_decision_cost(
                y_true, y_pred, cost_late=ratio, cost_early=1.0
            )
            costs[name] = result.avg_cost

        ranked = sorted(costs.items(), key=lambda x: x[1])
        rankings[ratio] = [name for name, _ in ranked]

    return rankings
