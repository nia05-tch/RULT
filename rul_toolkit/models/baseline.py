"""Simple baseline models for RUL prediction."""

import numpy as np
from sklearn.linear_model import LinearRegression


class SimpleRUL:
    """Linear regression baseline."""

    def __init__(self):
        self.model = LinearRegression()
        self.is_fitted = False

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y).flatten()
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X):
        if not self.is_fitted:
            raise ValueError("Model not fitted.")
        return self.model.predict(np.asarray(X))
