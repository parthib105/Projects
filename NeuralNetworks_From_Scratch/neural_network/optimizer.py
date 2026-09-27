from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Iterator
import numpy as np

from .layers import Parameter


class Optimizer(ABC):
    """Base optimizer class."""
    
    @abstractmethod
    def step(self, model: "Module") -> None:
        """Perform a single optimization step."""
        pass
    
    def zero_grad(self, model: "Module") -> None:
        model.zero_grad()


class SGD(Optimizer):
    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.0):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocity: dict[id, np.ndarray] = {}

    def step(self, model: "Module") -> None:
        for p in model.parameters():
            if p.grad is None or np.all(p.grad == 0):
                continue
            pid = id(p)
            if self.momentum > 0:
                if pid not in self.velocity:
                    self.velocity[pid] = np.zeros_like(p.data)
                self.velocity[pid] = self.momentum * self.velocity[pid] + p.grad
                p.data -= self.learning_rate * self.velocity[pid]
            else:
                p.data -= self.learning_rate * p.grad


class Adam(Optimizer):
    def __init__(
        self,
        learning_rate: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
    ):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m: dict[id, np.ndarray] = {}
        self.v: dict[id, np.ndarray] = {}
        self.t = 0

    def step(self, model: "Module") -> None:
        self.t += 1
        for p in model.parameters():
            if p.grad is None or np.all(p.grad == 0):
                continue
            pid = id(p)
            if pid not in self.m:
                self.m[pid] = np.zeros_like(p.data)
                self.v[pid] = np.zeros_like(p.data)
            
            # Update biased first moment estimate
            self.m[pid] = self.beta1 * self.m[pid] + (1 - self.beta1) * p.grad
            
            # Update biased second raw moment estimate
            self.v[pid] = self.beta2 * self.v[pid] + (1 - self.beta2) * (p.grad ** 2)
            
            # Compute bias-corrected estimates
            m_corrected = self.m[pid] / (1 - self.beta1 ** self.t)
            v_corrected = self.v[pid] / (1 - self.beta2 ** self.t)
            
            # Update parameters
            p.data -= self.learning_rate * m_corrected / (np.sqrt(v_corrected) + self.epsilon)


class RMSprop(Optimizer):
    def __init__(
        self,
        learning_rate: float = 0.001,
        alpha: float = 0.99,
        epsilon: float = 1e-8,
        momentum: float = 0.0,
    ):
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.epsilon = epsilon
        self.momentum = momentum
        self.square_avg: dict[id, np.ndarray] = {}
        self.velocity: dict[id, np.ndarray] = {}

    def step(self, model: "Module") -> None:
        for p in model.parameters():
            if p.grad is None or np.all(p.grad == 0):
                continue
            pid = id(p)
            if pid not in self.square_avg:
                self.square_avg[pid] = np.zeros_like(p.data)
                if self.momentum > 0:
                    self.velocity[pid] = np.zeros_like(p.data)
            
            self.square_avg[pid] = self.alpha * self.square_avg[pid] + (1 - self.alpha) * (p.grad ** 2)
            
            if self.momentum > 0:
                self.velocity[pid] = self.momentum * self.velocity[pid] + p.grad / (np.sqrt(self.square_avg[pid]) + self.epsilon)
                p.data -= self.learning_rate * self.velocity[pid]
            else:
                p.data -= self.learning_rate * p.grad / (np.sqrt(self.square_avg[pid]) + self.epsilon)