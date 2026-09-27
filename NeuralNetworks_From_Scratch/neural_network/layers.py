from __future__ import annotations
from typing import Dict, Iterator, List, Optional, Tuple
import numpy as np

from .activation_functions import ReLu, gradReLu, softmax, grad_softmax


class Parameter:
    """Container for a learnable parameter with its gradient."""
    def __init__(self, data: np.ndarray):
        self.data = data
        self.grad: np.ndarray = np.zeros_like(data)


class Module:
    """Base class for all neural network modules (PyTorch-style)."""
    def __init__(self):
        self._parameters: Dict[str, Parameter] = {}
        self._modules: Dict[str, Module] = {}
        self.training: bool = True

    def __setattr__(self, name: str, value: object) -> None:
        if isinstance(value, Parameter):
            self._parameters[name] = value
        elif isinstance(value, Module):
            self._modules[name] = value
        super().__setattr__(name, value)

    def parameters(self) -> Iterator[Parameter]:
        for p in self._parameters.values():
            yield p
        for m in self._modules.values():
            yield from m.parameters()

    def named_parameters(self) -> Iterator[Tuple[str, Parameter]]:
        for name, p in self._parameters.items():
            yield name, p
        for m_name, m in self._modules.items():
            for p_name, p in m.named_parameters():
                yield f"{m_name}.{p_name}", p

    def modules(self) -> Iterator[Module]:
        yield self
        for m in self._modules.values():
            yield from m.modules()

    def train(self) -> None:
        self.training = True
        for m in self._modules.values():
            m.train()

    def eval(self) -> None:
        self.training = False
        for m in self._modules.values():
            m.eval()

    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)

    def zero_grad(self) -> None:
        for p in self.parameters():
            p.grad.fill(0.0)


class Linear(Module):
    """Fully-connected layer with Xavier initialization."""
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(
            np.random.randn(in_features, out_features) * np.sqrt(2.0 / in_features)
        )
        if bias:
            self.bias = Parameter(np.zeros((1, out_features)))
        else:
            self.bias = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._input = x
        out = x @ self.weight.data
        if self.bias is not None:
            out += self.bias.data
        return out

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        self.weight.grad = self._input.T @ grad_output
        if self.bias is not None:
            self.bias.grad = np.sum(grad_output, axis=0, keepdims=True)
        return grad_output @ self.weight.data.T


class ReLU(Module):
    def forward(self, x: np.ndarray) -> np.ndarray:
        self._input = x
        return ReLu(x)

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        return grad_output * gradReLu(self._input)


class Softmax(Module):
    def forward(self, x: np.ndarray) -> np.ndarray:
        self._output = softmax(x)
        return self._output

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        return grad_softmax(self._output, grad_output)


class Sequential(Module):
    """Sequential container: runs modules in order."""
    def __init__(self, *modules: Module):
        super().__init__()
        for i, m in enumerate(modules):
            self._modules[str(i)] = m

    def forward(self, x: np.ndarray) -> np.ndarray:
        for m in self._modules.values():
            x = m(x)
        return x

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        for m in reversed(list(self._modules.values())):
            grad_output = m.backward(grad_output)
        return grad_output

    def append(self, module: Module) -> None:
        self._modules[str(len(self._modules))] = module


class Dropout(Module):
    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p
        self._mask: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        if not self.training or self.p == 0.0:
            return x
        self._mask = (np.random.rand(*x.shape) > self.p).astype(np.float32)
        return x * self._mask / (1.0 - self.p)

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self._mask is None:
            return grad_output
        return grad_output * self._mask / (1.0 - self.p)


class BatchNorm1d(Module):
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.gamma = Parameter(np.ones((1, num_features)))
        self.beta = Parameter(np.zeros((1, num_features)))
        self.running_mean = np.zeros((1, num_features))
        self.running_var = np.ones((1, num_features))

    def forward(self, x: np.ndarray) -> np.ndarray:
        if self.training:
            mean = x.mean(axis=0, keepdims=True)
            var = x.var(axis=0, keepdims=True)
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
        else:
            mean = self.running_mean
            var = self.running_var

        self._x_hat = (x - mean) / np.sqrt(var + self.eps)
        self._mean, self._var = mean, var
        return self.gamma.data * self._x_hat + self.beta.data

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        m = grad_output.shape[0]
        grad_gamma = np.sum(grad_output * self._x_hat, axis=0, keepdims=True)
        grad_beta = np.sum(grad_output, axis=0, keepdims=True)
        self.gamma.grad = grad_gamma
        self.beta.grad = grad_beta

        grad_x_hat = grad_output * self.gamma.data
        grad_var = np.sum(grad_x_hat * self._x_hat * -0.5 * (self._var + self.eps) ** -1.5, axis=0, keepdims=True)
        grad_mean = np.sum(grad_x_hat * -1 / np.sqrt(self._var + self.eps), axis=0, keepdims=True)
        grad_mean += grad_var * np.mean(-2 * self._x_hat, axis=0, keepdims=True)
        grad_input = grad_x_hat / np.sqrt(self._var + self.eps) + grad_var * 2 * self._x_hat / m + grad_mean / m
        return grad_input