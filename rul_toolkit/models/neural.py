"""
Neural network-based RUL prediction models.
"""

import numpy as np
from typing import Tuple, Optional
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class DenseNeuralRUL:
    """Feed-forward neural network for RUL prediction."""
    
    def __init__(self, input_shape: int, hidden_units: list = None, dropout_rate: float = 0.3, learning_rate: float = 0.001):
        self.input_shape = input_shape
        self.hidden_units = hidden_units or [128, 64, 32]
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.model = None
        self.history = None
        self._build_model()
    
    def _build_model(self) -> None:
        model = keras.Sequential()
        model.add(layers.Input(shape=(self.input_shape,)))
        for units in self.hidden_units:
            model.add(layers.Dense(units, activation='relu'))
            model.add(layers.Dropout(self.dropout_rate))
        model.add(layers.Dense(1, activation='relu'))
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae']
        )
        self.model = model
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, epochs=100, batch_size=32, verbose=0):
        callbacks = []
        validation_data = None
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)
            callbacks.append(keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True))
        self.history = self.model.fit(X_train, y_train, validation_data=validation_data, epochs=epochs, batch_size=batch_size, callbacks=callbacks, verbose=verbose)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        predictions = self.model.predict(X, verbose=0)
        return predictions.flatten()
    
    def predict_with_uncertainty(self, X: np.ndarray, n_iterations: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        predictions_samples = []
        for _ in range(n_iterations):
            pred = self.model(X, training=True)
            predictions_samples.append(pred.numpy().flatten())
        predictions_samples = np.array(predictions_samples)
        mean_predictions = predictions_samples.mean(axis=0)
        std_predictions = predictions_samples.std(axis=0)
        return mean_predictions, std_predictions


class LSTMSequentialRUL:
    """LSTM-based RUL prediction using degradation sequences."""
    
    def __init__(self, input_shape: Tuple[int, int], lstm_units: int = 64, dropout_rate: float = 0.2, learning_rate: float = 0.001):
        self.sequence_length = input_shape[0]
        self.n_sensors = input_shape[1]
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.model = None
        self.history = None
        self._build_model()
    
    def _build_model(self) -> None:
        model = keras.Sequential([
            layers.Input(shape=(self.sequence_length, self.n_sensors)),
            layers.LSTM(self.lstm_units, activation='relu', return_sequences=False, dropout=self.dropout_rate, recurrent_dropout=self.dropout_rate),
            layers.Dense(32, activation='relu'),
            layers.Dropout(self.dropout_rate),
            layers.Dense(1, activation='relu')
        ])
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate), loss='mse', metrics=['mae'])
        self.model = model
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, epochs=100, batch_size=32, verbose=0):
        callbacks = []
        validation_data = None
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)
            callbacks.append(keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True))
        self.history = self.model.fit(X_train, y_train, validation_data=validation_data, epochs=epochs, batch_size=batch_size, callbacks=callbacks, verbose=verbose)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        predictions = self.model.predict(X, verbose=0)
        return predictions.flatten()
    
    def predict_with_uncertainty(self, X: np.ndarray, n_iterations: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        predictions_samples = []
        for _ in range(n_iterations):
            pred = self.model(X, training=True)
            predictions_samples.append(pred.numpy().flatten())
        predictions_samples = np.array(predictions_samples)
        mean_predictions = predictions_samples.mean(axis=0)
        std_predictions = predictions_samples.std(axis=0)
        return mean_predictions, std_predictions
    
    @staticmethod
    def create_sequences(X: np.ndarray, y: np.ndarray, sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
        sequences, targets = [], []
        for i in range(len(X) - sequence_length):
            sequences.append(X[i:i + sequence_length])
            targets.append(y[i + sequence_length])
        return np.array(sequences), np.array(targets)
