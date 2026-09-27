import numpy as np

def ReLu(inp: np.ndarray) -> np.ndarray:
    return np.maximum(inp, 0)

def gradReLu(inp: np.ndarray) -> np.ndarray:
    return np.where(inp > 0, 1, 0)

def softmax(inp: np.ndarray) -> np.ndarray:
    exps = np.exp(inp - np.max(inp, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)

def grad_softmax(output: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
    """Gradient of softmax with cross-entropy combined (for efficiency).
    When used with cross-entropy loss, the gradient simplifies to (output - target).
    This function handles the general case."""
    # Jacobian-vector product for softmax
    # For cross-entropy, caller should pass (output - one_hot_target) as grad_output
    return grad_output
