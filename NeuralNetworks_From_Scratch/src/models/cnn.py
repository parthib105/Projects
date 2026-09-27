"""Configurable Deep Convolutional Neural Network (CNN) architecture."""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional, Tuple, Union

# Ensure project root is in sys.path for direct execution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
import torch.nn as nn

from src.models.mlp import ACTIVATION_REGISTRY


class ConfigurableCNN(nn.Module):
    """Modular Deep Convolutional Neural Network supporting variable input resolutions.

    Combines configurable Conv2d blocks (with BatchNorm2d, Dropout2d, MaxPool2d)
    and an Adaptive Average Pooling neck that seamlessly supports variable input
    sizes (e.g. 28x28 MNIST, 32x32 CIFAR-10, or higher resolution images).

    Attributes:
        in_channels (int): Number of input image channels (e.g. 1 for Grayscale, 3 for RGB).
        conv_channels (List[int]): Filter counts for each convolutional stage.
        kernel_sizes (List[int]): Kernel size for each convolutional stage.
        fc_dims (List[int]): Dense hidden layer dimensions in the classification head.
        num_classes (int): Number of target output classes.
        activation (str): Non-linear activation name ('relu', 'gelu', 'silu', etc.).
        dropout (float): Dropout probability applied in conv and dense blocks (0.0 to 1.0).
        use_batch_norm (bool): If True, applies 2D batch normalization after convolutions.
        pool_size (Tuple[int, int]): Target spatial output from adaptive average pooling.
    """

    def __init__(
        self,
        in_channels: int = 1,
        conv_channels: Optional[List[int]] = None,
        kernel_sizes: Optional[Union[List[int], int]] = None,
        fc_dims: Optional[List[int]] = None,
        num_classes: int = 10,
        activation: str = "relu",
        dropout: float = 0.2,
        use_batch_norm: bool = True,
        pool_size: Tuple[int, int] = (1, 1),
    ) -> None:
        super().__init__()

        if in_channels <= 0:
            raise ValueError(f"in_channels must be positive, got {in_channels}")
        if num_classes <= 0:
            raise ValueError(f"num_classes must be positive, got {num_classes}")
        if not (0.0 <= dropout < 1.0):
            raise ValueError(f"dropout must be in range [0.0, 1.0), got {dropout}")

        act_name = activation.lower().strip()
        if act_name not in ACTIVATION_REGISTRY:
            raise ValueError(
                f"Unsupported activation '{activation}'. "
                f"Supported: {list(ACTIVATION_REGISTRY.keys())}"
            )

        self.in_channels = in_channels
        self.conv_channels = list(conv_channels) if conv_channels is not None else [32, 64, 128]
        self.fc_dims = list(fc_dims) if fc_dims is not None else [128]
        self.num_classes = num_classes
        self.activation_name = act_name
        self.dropout_rate = dropout
        self.use_batch_norm = use_batch_norm
        self.pool_size = pool_size

        act_class = ACTIVATION_REGISTRY[act_name]
        n_stages = len(self.conv_channels)

        # Standardize kernel sizes
        if kernel_sizes is None:
            self.kernel_sizes = [3] * n_stages
        elif isinstance(kernel_sizes, int):
            self.kernel_sizes = [kernel_sizes] * n_stages
        else:
            if len(kernel_sizes) != n_stages:
                raise ValueError(
                    f"Length of kernel_sizes ({len(kernel_sizes)}) must match "
                    f"conv_channels ({n_stages})"
                )
            self.kernel_sizes = list(kernel_sizes)

        # 1. Convolutional Feature Extractor
        conv_layers: List[nn.Module] = []
        prev_channels = self.in_channels

        for out_channels, k_size in zip(self.conv_channels, self.kernel_sizes):
            if out_channels <= 0:
                raise ValueError(f"Conv channel count must be positive, got {out_channels}")
            if k_size <= 0:
                raise ValueError(f"Kernel size must be positive, got {k_size}")

            # Padding preserves spatial dimensions before max pooling
            padding = k_size // 2

            conv_layers.append(
                nn.Conv2d(
                    prev_channels,
                    out_channels,
                    kernel_size=k_size,
                    padding=padding,
                    bias=not self.use_batch_norm,
                )
            )

            if self.use_batch_norm:
                conv_layers.append(nn.BatchNorm2d(out_channels))

            conv_layers.append(act_class())

            # Spatial downsampling
            conv_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

            if self.dropout_rate > 0.0:
                conv_layers.append(nn.Dropout2d(p=self.dropout_rate))

            prev_channels = out_channels

        self.features = nn.Sequential(*conv_layers)

        # 2. Adaptive Pooling Neck (enables variable resolution input)
        self.adaptive_pool = nn.AdaptiveAvgPool2d(self.pool_size)

        # 3. Dense Classification Head
        flattened_dim = prev_channels * self.pool_size[0] * self.pool_size[1]
        fc_layers: List[nn.Module] = []
        prev_fc = flattened_dim

        for fc_dim in self.fc_dims:
            if fc_dim <= 0:
                raise ValueError(f"FC dimensions must be positive, got {fc_dim}")

            fc_layers.append(nn.Linear(prev_fc, fc_dim))
            if self.use_batch_norm:
                fc_layers.append(nn.BatchNorm1d(fc_dim))
            fc_layers.append(act_class())
            if self.dropout_rate > 0.0:
                fc_layers.append(nn.Dropout(p=self.dropout_rate))
            prev_fc = fc_dim

        fc_layers.append(nn.Linear(prev_fc, num_classes))
        self.classifier = nn.Sequential(*fc_layers)

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Kaiming normal initialization for conv and linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract spatial convolutional feature maps before pooling.

        Useful for Grad-CAM heatmaps and intermediate activation visualizations.
        """
        return self.features(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the CNN.

        Args:
            x: Input image tensor of shape (B, C, H, W).

        Returns:
            Output logits tensor of shape (B, num_classes).
        """
        feat = self.features(x)
        pooled = self.adaptive_pool(feat)
        flat = torch.flatten(pooled, start_dim=1)
        return self.classifier(flat)

    def count_parameters(self) -> Dict[str, int]:
        """Compute total and trainable parameter counts."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total_parameters": total, "trainable_parameters": trainable}


if __name__ == "__main__":
    # Add project root to sys.path so direct execution works seamlessly
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

    from src.utils.device import get_device

    print("=" * 60)
    print(" Configurable Deep CNN Architecture Verification")
    print("=" * 60)

    device = get_device("auto")

    # Example 1: 1-channel 28x28 (MNIST / Fashion-MNIST)
    cnn_mnist = ConfigurableCNN(
        in_channels=1,
        conv_channels=[32, 64, 128],
        kernel_sizes=3,
        fc_dims=[128],
        num_classes=10,
        activation="relu",
        dropout=0.25,
        use_batch_norm=True,
    ).to(device)

    print("\n[1] MNIST / Fashion-MNIST Model (1 Channel, 28x28):")
    print(f"    Conv Channels : {cnn_mnist.conv_channels}")
    print(f"    Parameters    : {cnn_mnist.count_parameters()}")

    dummy_mnist = torch.randn(16, 1, 28, 28, device=device)
    out_mnist = cnn_mnist(dummy_mnist)
    print(f"    Input Shape   : {tuple(dummy_mnist.shape)}")
    print(f"    Output Shape  : {tuple(out_mnist.shape)} (Expected: (16, 10))")

    # Example 2: 3-channel 32x32 (CIFAR-10) with GELU activation
    cnn_cifar = ConfigurableCNN(
        in_channels=3,
        conv_channels=[64, 128, 256],
        kernel_sizes=[3, 3, 3],
        fc_dims=[256, 128],
        num_classes=10,
        activation="gelu",
        dropout=0.3,
        use_batch_norm=True,
    ).to(device)

    print("\n[2] CIFAR-10 Model (3 Channels, 32x32, GELU):")
    print(f"    Conv Channels : {cnn_cifar.conv_channels}")
    print(f"    Parameters    : {cnn_cifar.count_parameters()}")

    dummy_cifar = torch.randn(8, 3, 32, 32, device=device)
    out_cifar = cnn_cifar(dummy_cifar)
    print(f"    Input Shape   : {tuple(dummy_cifar.shape)}")
    print(f"    Output Shape  : {tuple(out_cifar.shape)} (Expected: (8, 10))")

    # Example 3: Feature extraction check
    features = cnn_mnist.extract_features(dummy_mnist)
    print(f"\n[3] Feature Extractor Output Shape: {tuple(features.shape)}")

    # Example 4: Backward pass & gradient flow on GPU
    target = torch.randint(0, 10, (16,), device=device)
    loss = nn.CrossEntropyLoss()(out_mnist, target)
    loss.backward()
    has_grad = all(p.grad is not None for p in cnn_mnist.parameters())
    print(f"[4] Backward Pass / Gradient Flow: {'PASS (All gradients computed)' if has_grad else 'FAIL'}")
    print("=" * 60)
