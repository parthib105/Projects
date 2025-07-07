import numpy as np

def ReLu(inp):
    return np.maximum(inp, 0)

def gradReLu(inp):
    return np.where(inp > 0, 1, 0)

def softmax(inp):
    exps = np.exp(inp - np.max(inp, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)
