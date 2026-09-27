"""Configurable arbitrary-depth Multi-Layer Perceptron (MLP) architecture."""

from __future__ import annotations

from typing import Dict, List, Type

import torch
import torch.nn as nn

# Registry of supported activation functions
ACTIVATION_REGISTRY: Dict[str, Type[nn.Module]] = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
    "leaky_relu": nn.LeakyReLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
}


class ConfigurableMLP(nn.Module):
    """Modular Multi-Layer Perceptron supporting arbitrary layer depth and configurations.

    Attributes:
        input_dim (int): Number of input features.
        hidden_dims (List[int]): List specifying neuron counts for each hidden layer.
        output_dim (int): Number of output classes or target values.
        activation (str): Non-linear activation name ('relu', 'gelu', 'silu', etc.).
        dropout (float): Dropout probability applied after each hidden layer (0.0 to 1.0).
        use_batch_norm (bool): If True, applies 1D batch normalization before activations.
        flatten_input (bool): If True, automatically flattens multi-dimensional inputs.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        activation: str = "relu",
        dropout: float = 0.0,
        use_batch_norm: bool = False,
        flatten_input: bool = True,
    ) -> None:
        super().__init__()

        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")
        if output_dim <= 0:
            raise ValueError(f"output_dim must be positive, got {output_dim}")
        if not (0.0 <= dropout < 1.0):
            raise ValueError(f"dropout must be in range [0.0, 1.0), got {dropout}")

        act_name = activation.lower().strip()
        if act_name not in ACTIVATION_REGISTRY:
            raise ValueError(
                f"Unsupported activation '{activation}'. "
                f"Supported: {list(ACTIVATION_REGISTRY.keys())}"
            )

        self.input_dim = input_dim
        self.hidden_dims = list(hidden_dims)
        self.output_dim = output_dim
        self.activation_name = act_name
        self.dropout_rate = dropout
        self.use_batch_norm = use_batch_norm
        self.flatten_input = flatten_input

        act_class = ACTIVATION_REGISTRY[act_name]

        # Build dynamic sequential layer stack
        layers: List[nn.Module] = []
        prev_dim = input_dim

        for hidden_dim in self.hidden_dims:
            if hidden_dim <= 0:
                raise ValueError(f"Hidden dimensions must be positive, got {hidden_dim}")

            layers.append(nn.Linear(prev_dim, hidden_dim))

            if self.use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))

            layers.append(act_class())

            if self.dropout_rate > 0.0:
                layers.append(nn.Dropout(p=self.dropout_rate))

            prev_dim = hidden_dim

        # Final projection layer (produces raw logits)
        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

        # Initialize network weights using Kaiming normal
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize linear layer weights using He / Kaiming normal initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the MLP.

        Args:
            x: Input tensor of shape (B, input_dim) or (B, C, H, W).

        Returns:
            Output logits tensor of shape (B, output_dim).
        """
        if self.flatten_input and x.dim() > 2:
            x = torch.flatten(x, start_dim=1)

        return self.network(x)

    def count_parameters(self) -> Dict[str, int]:
        """Compute the total and trainable parameter count of the model."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total_parameters": total, "trainable_parameters": trainable}


if __name__ == "__main__":
    import os
    import sys

    # Add project root to sys.path so direct execution works seamlessly
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

    from src.utils.device import get_device

    print("=" * 60)
    print(" Configurable Arbitrary-Depth MLP Verification")
    print("=" * 60)

    device = get_device("auto")

    # Example 1: 3-hidden-layer network for MNIST
    mlp_3layer = ConfigurableMLP(
        input_dim=784,
        hidden_dims=[512, 256, 128],
        output_dim=10,
        activation="relu",
        dropout=0.2,
        use_batch_norm=True,
    ).to(device)

    print("\n[1] 3-Hidden-Layer Network Architecture:")
    print(f"    Dimensions : 784 -> 512 -> 256 -> 128 -> 10")
    print(f"    Activation : {mlp_3layer.activation_name}")
    print(f"    Parameters : {mlp_3layer.count_parameters()}")

    # Test forward pass with multi-dimensional (B, 1, 28, 28) image tensor
    dummy_images = torch.randn(16, 1, 28, 28, device=device)
    out = mlp_3layer(dummy_images)
    print(f"    Input Shape: {tuple(dummy_images.shape)}")
    print(f"    Output Logits Shape: {tuple(out.shape)} (Expected: (16, 10))")

    # Example 2: Deep 5-hidden-layer network with GELU
    mlp_deep = ConfigurableMLP(
        input_dim=128,
        hidden_dims=[256, 256, 128, 64, 32],
        output_dim=2,
        activation="gelu",
        dropout=0.1,
    ).to(device)

    dummy_vec = torch.randn(8, 128, device=device)
    out_deep = mlp_deep(dummy_vec)
    print("\n[2] Deep 5-Hidden-Layer Network (GELU):")
    print(f"    Dimensions : 128 -> 256 -> 256 -> 128 -> 64 -> 32 -> 2")
    print(f"    Parameters : {mlp_deep.count_parameters()}")
    print(f"    Output Logits Shape: {tuple(out_deep.shape)} (Expected: (8, 2))")

    # Example 3: Gradient flow verification (Backpropagation)
    target = torch.randint(0, 10, (16,), device=device)
    criterion = nn.CrossEntropyLoss()
    loss = criterion(out, target)
    loss.backward()
    has_grad = all(p.grad is not None for p in mlp_3layer.parameters())
    print(f"\n[3] Backward Pass / Gradient Flow: {'PASS (All gradients computed)' if has_grad else 'FAIL'}")
    print("=" * 60)
