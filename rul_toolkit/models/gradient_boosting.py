"""Gradient boosting models for RUL prediction."""

import numpy as np

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False


class LightGBMRUL:
    """LightGBM baseline – strong and fast on tabular data."""

    def __init__(
        self,
        n_estimators: int = 200,
        learning_rate: float = 0.05,
        max_depth: int = -1,
        num_leaves: int = 31,
        random_state: int = 42,
    ):
        if not HAS_LIGHTGBM:
            raise ImportError("lightgbm is not installed. Run: pip install lightgbm")

        self.model = lgb.LGBMRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            num_leaves=num_leaves,
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,
        )
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X)
        y = np.asarray(y).flatten()
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        X = np.asarray(X)
        return self.model.predict(X)

    def feature_importance(self):
        if not self.is_fitted:
            raise ValueError("Model not fitted.")
        return self.model.feature_importances_.tolist()
