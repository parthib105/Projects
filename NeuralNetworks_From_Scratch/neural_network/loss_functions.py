import numpy as np

def cross_entropy(predictions, labels):
    n_samples = predictions.shape[0]
    logp = -np.log(predictions[range(n_samples), labels] + 1e-9)
    return np.sum(logp) / n_samples

def grad_cross_entropy(predictions, labels):
    n_samples = predictions.shape[0]
    grad = predictions.copy()
    grad[range(n_samples), labels] -= 1
    return grad / n_samples
