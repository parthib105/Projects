from __future__ import annotations
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union
import numpy as np

from .layers import Module, Linear, ReLU, Softmax, Sequential, Parameter
from .loss_functions import cross_entropy, grad_cross_entropy, mse, grad_mse, get_loss, LossName
from .optimizer import Optimizer
from .utils import accuracy


class NeuralNetwork(Module):
    """General-purpose neural network built from Modules (PyTorch-style)."""
    
    def __init__(
        self,
        layers: List[Module],
        loss: LossName = "cross_entropy",
    ):
        super().__init__()
        self.model = Sequential(*layers)
        self.loss_name = loss
        self.loss_fn, self.grad_loss_fn = get_loss(loss)
        
        # Training history
        self.train_losses: List[float] = []
        self.train_accuracies: List[float] = []
        self.val_losses: List[float] = []
        self.val_accuracies: List[float] = []

    def forward(self, x: np.ndarray) -> np.ndarray:
        return self.model(x)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict class labels (for classification)."""
        self.eval()
        probs = self.forward(x)
        return np.argmax(probs, axis=1)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        self.eval()
        return self.forward(x)

    def evaluate(self, x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        self.eval()
        predictions = self.forward(x)
        loss = self.loss_fn(predictions, y)
        acc = accuracy(predictions, y)
        return loss, acc

    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 10,
        batch_size: int = 32,
        optimizer: Optional[Optimizer] = None,
        learning_rate: float = 0.01,
        verbose: bool = True,
    ) -> None:
        if optimizer is None:
            from .optimizer import SGD
            optimizer = SGD(learning_rate)

        n_samples = x_train.shape[0]
        n_batches = max(1, n_samples // batch_size)

        for epoch in range(epochs):
            # Shuffle
            perm = np.random.permutation(n_samples)
            x_shuffled = x_train[perm]
            y_shuffled = y_train[perm]

            epoch_loss = 0.0
            epoch_acc = 0.0

            self.train()
            for i in range(0, n_samples, batch_size):
                xb = x_shuffled[i:i+batch_size]
                yb = y_shuffled[i:i+batch_size]

                # Forward
                preds = self.forward(xb)
                loss = self.loss_fn(preds, yb)

                # Backward
                self.zero_grad()
                grad = self.grad_loss_fn(preds, yb)
                self.model.backward(grad)

                # Update
                optimizer.step(self)

                epoch_loss += loss
                epoch_acc += accuracy(preds, yb)

            epoch_loss /= n_batches
            epoch_acc /= n_batches
            self.train_losses.append(epoch_loss)
            self.train_accuracies.append(epoch_acc)

            # Validation
            if x_val is not None and y_val is not None:
                val_loss, val_acc = self.evaluate(x_val, y_val)
                self.val_losses.append(val_loss)
                self.val_accuracies.append(val_acc)
                if verbose:
                    print(f"Epoch {epoch+1}/{epochs}: "
                          f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}, "
                          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            elif verbose:
                print(f"Epoch {epoch+1}/{epochs}: "
                      f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}")

    # Backward-compatibility: state_dict / load_state_dict
    def state_dict(self) -> Dict[str, np.ndarray]:
        return {name: p.data.copy() for name, p in self.named_parameters()}

    def load_state_dict(self, state: Dict[str, np.ndarray]) -> None:
        for name, p in self.named_parameters():
            if name in state:
                p.data[...] = state[name]

    def save(self, path: str) -> None:
        np.savez_compressed(path, **self.state_dict())

    @classmethod
    def load(cls, path: str, layers: List[Module], loss: LossName = "cross_entropy") -> "NeuralNetwork":
        model = cls(layers, loss)
        data = np.load(path)
        model.load_state_dict({k: data[k] for k in data.files})
        return model


# Backward-compatible SimpleNN wrapper
class SimpleNN(NeuralNetwork):
    """Original 2-layer MLP interface for backward compatibility."""
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        loss: LossName = "cross_entropy",
    ):
        layers = [
            Linear(input_size, hidden_size),
            ReLU(),
            Linear(hidden_size, output_size),
            Softmax(),
        ]
        super().__init__(layers, loss)