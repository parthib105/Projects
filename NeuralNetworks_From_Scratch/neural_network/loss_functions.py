import numpy as np
from typing import Literal

def cross_entropy(predictions: np.ndarray, labels: np.ndarray) -> float:
    n_samples = predictions.shape[0]
    logp = -np.log(predictions[range(n_samples), labels] + 1e-9)
    return float(np.sum(logp) / n_samples)

def grad_cross_entropy(predictions: np.ndarray, labels: np.ndarray) -> np.ndarray:
    n_samples = predictions.shape[0]
    grad = predictions.copy()
    grad[range(n_samples), labels] -= 1
    return grad / n_samples

def mse(predictions: np.ndarray, targets: np.ndarray) -> float:
    diff = predictions - targets
    return float(np.mean(diff * diff))

def grad_mse(predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
    return 2.0 * (predictions - targets) / predictions.shape[0]

def binary_cross_entropy(predictions: np.ndarray, targets: np.ndarray) -> float:
    eps = 1e-9
    predictions = np.clip(predictions, eps, 1 - eps)
    return float(-np.mean(targets * np.log(predictions) + (1 - targets) * np.log(1 - predictions)))

def grad_binary_cross_entropy(predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
    eps = 1e-9
    predictions = np.clip(predictions, eps, 1 - eps)
    return (predictions - targets) / (predictions * (1 - predictions) + eps) / predictions.shape[0]

LossName = Literal["cross_entropy", "mse", "bce"]

def get_loss(name: LossName):
    losses = {
        "cross_entropy": (cross_entropy, grad_cross_entropy),
        "mse": (mse, grad_mse),
        "bce": (binary_cross_entropy, grad_binary_cross_entropy),
    }
    return losses[name]