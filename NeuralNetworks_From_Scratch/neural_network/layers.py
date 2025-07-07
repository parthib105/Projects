import numpy as np
from .activation_functions import ReLu, gradReLu

class DenseLayer:
    def __init__(self, input_dim, output_dim, activation='relu'):
        # Xavier/Glorot initialization
        self.W = np.random.randn(input_dim, output_dim) * np.sqrt(2.0 / input_dim)
        self.b = np.zeros((1, output_dim))
        self.input = None
        self.output = None
        self.activation = activation

    def forward(self, x):
        self.input = x
        self.output = x @ self.W + self.b
        
        if self.activation == 'relu':
            return ReLu(self.output)
        elif self.activation == 'linear':
            return self.output
        else:
            raise ValueError(f"Unsupported activation: {self.activation}")

    def backward(self, grad_output):
        # Apply activation gradient
        if self.activation == 'relu':
            grad_output = grad_output * gradReLu(self.output)
        
        grad_input = grad_output @ self.W.T
        grad_W = self.input.T @ grad_output
        grad_b = np.sum(grad_output, axis=0, keepdims=True)
        
        return grad_input, grad_W, grad_b