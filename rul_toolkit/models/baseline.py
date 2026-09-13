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

    def get_coefficients(self):
        if not self.is_fitted:
            raise ValueError("Model not fitted.")

        return self.model.coef_


class ConstantRUL:
    """Constant predictor baseline using the mean training RUL."""

    def __init__(self):
        self.mean_rul = None
        self.is_fitted = False

    def fit(self, X, y):
        y = np.asarray(y).flatten()

        if len(y) == 0:
            raise ValueError("Cannot fit model on empty data.")

        self.mean_rul = float(np.mean(y))
        self.is_fitted = True
        return self

    def predict(self, X):
        if not self.is_fitted:
            raise ValueError("Model not fitted.")

        X = np.asarray(X)

        return np.full(len(X), self.mean_rul)
