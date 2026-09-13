"""Tree-based models for RUL prediction."""

import numpy as np
from sklearn.ensemble import RandomForestRegressor

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False


class RandomForestRUL:
    """Random Forest regression model for RUL prediction."""

    def __init__(
        self,
        n_estimators=100,
        max_depth=None,
        random_state=42,
    ):
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
        )
        self.is_fitted = False

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y).flatten()

        if len(X) != len(y):
            raise ValueError("X and y must have the same number of samples.")
        if len(y) == 0:
            raise ValueError("Cannot fit model on empty data.")

        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X):
        if not self.is_fitted:
            raise ValueError("Model not fitted.")

        return self.model.predict(np.asarray(X))


class LightGBMRUL:
    """LightGBM regression model for RUL prediction."""

    def __init__(
        self,
        n_estimators=100,
        learning_rate=0.05,
        max_depth=-1,
        random_state=42,
    ):
        if not LIGHTGBM_AVAILABLE:
            raise ImportError(
                "lightgbm is not installed. Run: pip install lightgbm"
            )

        self.model = lgb.LGBMRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state,
            verbosity=-1,
        )
        self.is_fitted = False

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y).flatten()

        if len(X) != len(y):
            raise ValueError("X and y must have the same number of samples.")
        if len(y) == 0:
            raise ValueError("Cannot fit model on empty data.")

        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X):
        if not self.is_fitted:
            raise ValueError("Model not fitted.")

        return self.model.predict(np.asarray(X))
