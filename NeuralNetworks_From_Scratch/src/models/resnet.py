"""Custom Residual Network (ResNet) architecture with modular residual blocks."""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional, Tuple, Type

# Ensure project root is in sys.path for direct execution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
import torch.nn as nn

from src.models.mlp import ACTIVATION_REGISTRY


def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding to preserve spatial dimensions when stride=1."""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution for projection shortcut / downsampling."""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=1,
        stride=stride,
        bias=False,
    )


class ResidualBlock(nn.Module):
    """Basic residual block featuring two 3x3 convs and a shortcut projection.

    y = Activation(F(x) + Shortcut(x))

    Attributes:
        in_channels (int): Input feature map channel count.
        out_channels (int): Output feature map channel count.
        stride (int): Stride for the first convolution (downsampling).
        downsample (Optional[nn.Module]): Projection shortcut if dimensions change.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        activation: str = "relu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        act_class = ACTIVATION_REGISTRY[activation]

        self.conv1 = conv3x3(in_channels, out_channels, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act1 = act_class()

        self.conv2 = conv3x3(out_channels, out_channels, stride=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act2 = act_class()

        self.downsample = downsample
        self.stride = stride
        self.dropout = nn.Dropout2d(p=dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.act2(out)

        return out


class CustomResNet(nn.Module):
    """Custom Residual Network with configurable depth, channels, and input modes.

    Supports both small-resolution vision tasks (28x28 MNIST, 32x32 CIFAR-10)
    and standard vision resolutions (64x64, 224x224).

    Attributes:
        in_channels (int): Input image channels (1 for grayscale, 3 for RGB).
        num_classes (int): Number of target classification classes.
        block_counts (List[int]): Number of residual blocks per stage (e.g. [2, 2, 2, 2] for ResNet-18).
        stage_channels (List[int]): Filter depth for each stage (e.g. [64, 128, 256, 512]).
        small_inputs (bool): If True, uses a 3x3 initial stem tailored for small inputs (MNIST/CIFAR)
                             avoiding aggressive spatial downsampling.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 10,
        block_counts: Optional[List[int]] = None,
        stage_channels: Optional[List[int]] = None,
        activation: str = "relu",
        dropout: float = 0.0,
        small_inputs: bool = True,
    ) -> None:
        super().__init__()

        if in_channels <= 0:
            raise ValueError(f"in_channels must be positive, got {in_channels}")
        if num_classes <= 0:
            raise ValueError(f"num_classes must be positive, got {num_classes}")

        act_name = activation.lower().strip()
        if act_name not in ACTIVATION_REGISTRY:
            raise ValueError(
                f"Unsupported activation '{activation}'. "
                f"Supported: {list(ACTIVATION_REGISTRY.keys())}"
            )

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.block_counts = list(block_counts) if block_counts is not None else [2, 2, 2, 2]
        self.stage_channels = (
            list(stage_channels) if stage_channels is not None else [64, 128, 256, 512]
        )
        self.activation_name = act_name
        self.dropout_rate = dropout
        self.small_inputs = small_inputs

        if len(self.block_counts) != len(self.stage_channels):
            raise ValueError(
                f"block_counts length ({len(self.block_counts)}) must match "
                f"stage_channels length ({len(self.stage_channels)})"
            )

        act_class = ACTIVATION_REGISTRY[act_name]
        init_channels = self.stage_channels[0]

        # Initial stem
        if small_inputs:
            # 3x3 conv preserving spatial dimensions for small images (28x28 or 32x32)
            self.stem = nn.Sequential(
                conv3x3(in_channels, init_channels, stride=1),
                nn.BatchNorm2d(init_channels),
                act_class(),
            )
        else:
            # Standard 7x7 conv with maxpool for large images (e.g. 224x224)
            self.stem = nn.Sequential(
                nn.Conv2d(in_channels, init_channels, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(init_channels),
                act_class(),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            )

        # Residual stages
        self.current_channels = init_channels
        self.stages = nn.ModuleList()

        for i, (num_blocks, out_ch) in enumerate(zip(self.block_counts, self.stage_channels)):
            # First stage has stride=1, subsequent stages downsample with stride=2
            stride = 1 if i == 0 else 2
            stage = self._make_stage(out_ch, num_blocks, stride=stride)
            self.stages.append(stage)

        # Global average pooling and linear classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.stage_channels[-1], num_classes)

        # Weight initialization
        self._init_weights()

    def _make_stage(self, out_channels: int, num_blocks: int, stride: int = 1) -> nn.Sequential:
        """Create a residual stage containing multiple ResidualBlocks."""
        downsample = None

        if stride != 1 or self.current_channels != out_channels:
            downsample = nn.Sequential(
                conv1x1(self.current_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels),
            )

        layers: List[nn.Module] = []
        layers.append(
            ResidualBlock(
                self.current_channels,
                out_channels,
                stride=stride,
                downsample=downsample,
                activation=self.activation_name,
                dropout=self.dropout_rate,
            )
        )
        self.current_channels = out_channels

        for _ in range(1, num_blocks):
            layers.append(
                ResidualBlock(
                    out_channels,
                    out_channels,
                    stride=1,
                    downsample=None,
                    activation=self.activation_name,
                    dropout=self.dropout_rate,
                )
            )

        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        """Kaiming normal initialization for conv layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract spatial convolutional feature maps prior to pooling."""
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the residual network."""
        x = self.extract_features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def count_parameters(self) -> Dict[str, int]:
        """Compute total and trainable parameter counts."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total_parameters": total, "trainable_parameters": trainable}


if __name__ == "__main__":
    from src.utils.device import get_device

    print("=" * 60)
    print(" Custom ResNet Architecture Verification")
    print("=" * 60)

    device = get_device("auto")

    # Example 1: ResNet-10 (4 stages, 1 block each: [1, 1, 1, 1]) for MNIST (1 channel, 28x28)
    resnet_mnist = CustomResNet(
        in_channels=1,
        num_classes=10,
        block_counts=[1, 1, 1, 1],
        stage_channels=[32, 64, 128, 256],
        small_inputs=True,
    ).to(device)

    print("\n[1] ResNet-10 for MNIST (1 Channel, 28x28):")
    print(f"    Block Counts   : {resnet_mnist.block_counts}")
    print(f"    Stage Channels : {resnet_mnist.stage_channels}")
    print(f"    Parameters     : {resnet_mnist.count_parameters()}")

    dummy_mnist = torch.randn(8, 1, 28, 28, device=device)
    out_mnist = resnet_mnist(dummy_mnist)
    print(f"    Input Shape    : {tuple(dummy_mnist.shape)}")
    print(f"    Output Shape   : {tuple(out_mnist.shape)} (Expected: (8, 10))")

    # Example 2: ResNet-18 ([2, 2, 2, 2]) for CIFAR-10 (3 channels, 32x32)
    resnet_cifar = CustomResNet(
        in_channels=3,
        num_classes=10,
        block_counts=[2, 2, 2, 2],
        stage_channels=[64, 128, 256, 512],
        small_inputs=True,
    ).to(device)

    print("\n[2] ResNet-18 for CIFAR-10 (3 Channels, 32x32):")
    print(f"    Parameters     : {resnet_cifar.count_parameters()}")
    dummy_cifar = torch.randn(4, 3, 32, 32, device=device)
    out_cifar = resnet_cifar(dummy_cifar)
    print(f"    Output Shape   : {tuple(out_cifar.shape)} (Expected: (4, 10))")

    # Example 3: Feature extraction check
    features = resnet_mnist.extract_features(dummy_mnist)
    print(f"\n[3] Extracted Feature Maps Shape: {tuple(features.shape)}")

    # Example 4: Gradient flow backprop verification across all residual stages
    target = torch.randint(0, 10, (8,), device=device)
    loss = nn.CrossEntropyLoss()(out_mnist, target)
    loss.backward()
    has_grad = all(p.grad is not None for p in resnet_mnist.parameters())
    print(f"[4] Backward Pass / Gradient Flow: {'PASS (All gradients computed across stages)' if has_grad else 'FAIL'}")
    print("=" * 60)
