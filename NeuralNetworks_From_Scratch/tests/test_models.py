"""Unit tests for neural network model architectures."""

import pytest
import torch
import torch.nn as nn

from src.models.cnn import ConfigurableCNN
from src.models.mlp import ACTIVATION_REGISTRY, ConfigurableMLP
from src.utils.device import get_device


class TestConfigurableMLP:
    """Test suite for ConfigurableMLP architecture."""

    def test_default_initialization(self):
        """Test default single hidden layer MLP creation."""
        model = ConfigurableMLP(input_dim=784, hidden_dims=[128], output_dim=10)
        assert model.input_dim == 784
        assert model.hidden_dims == [128]
        assert model.output_dim == 10
        assert model.activation_name == "relu"
        assert model.dropout_rate == 0.0
        assert not model.use_batch_norm

    @pytest.mark.parametrize(
        "hidden_dims",
        [
            [],  # Direct linear classifier: 784 -> 10
            [128],  # 1 hidden layer
            [256, 128],  # 2 hidden layers
            [512, 256, 128, 64],  # 4 hidden layers
            [1024, 512, 256, 128, 64, 32],  # 6 hidden layers
        ],
    )
    def test_arbitrary_depth_forward_pass(self, hidden_dims):
        """Verify forward pass output shape matches (B, output_dim) for arbitrary depths."""
        batch_size = 16
        input_dim = 64
        output_dim = 5
        model = ConfigurableMLP(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
        )

        dummy_x = torch.randn(batch_size, input_dim)
        out = model(dummy_x)

        assert out.shape == (batch_size, output_dim)
        assert not torch.isnan(out).any()

    def test_multi_dimensional_input_flattening(self):
        """Verify automatic flattening of 4D image inputs (B, C, H, W) -> (B, output_dim)."""
        model = ConfigurableMLP(
            input_dim=784,
            hidden_dims=[256, 128],
            output_dim=10,
            flatten_input=True,
        )
        dummy_images = torch.randn(8, 1, 28, 28)
        out = model(dummy_images)
        assert out.shape == (8, 10)

    @pytest.mark.parametrize("activation", list(ACTIVATION_REGISTRY.keys()))
    def test_all_supported_activations(self, activation):
        """Verify all registered activation functions work properly."""
        model = ConfigurableMLP(
            input_dim=32,
            hidden_dims=[64, 32],
            output_dim=4,
            activation=activation,
        )
        dummy_x = torch.randn(4, 32)
        out = model(dummy_x)
        assert out.shape == (4, 4)

    def test_batch_norm_and_dropout(self):
        """Verify model runs with both BatchNorm and Dropout active."""
        model = ConfigurableMLP(
            input_dim=32,
            hidden_dims=[64, 32],
            output_dim=2,
            dropout=0.3,
            use_batch_norm=True,
        )
        dummy_x = torch.randn(16, 32)

        # Training mode
        model.train()
        train_out = model(dummy_x)
        assert train_out.shape == (16, 2)

        # Eval mode
        model.eval()
        eval_out = model(dummy_x)
        assert eval_out.shape == (16, 2)

    def test_backward_pass_and_gradient_flow(self):
        """Verify loss backpropagation computes valid gradients for all parameters."""
        model = ConfigurableMLP(
            input_dim=64,
            hidden_dims=[128, 64],
            output_dim=10,
            use_batch_norm=True,
        )
        dummy_x = torch.randn(8, 64)
        targets = torch.randint(0, 10, (8,))

        criterion = nn.CrossEntropyLoss()
        logits = model(dummy_x)
        loss = criterion(logits, targets)
        loss.backward()

        for name, param in model.named_parameters():
            assert param.grad is not None, f"Gradient missing for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"

    def test_cuda_execution(self):
        """Verify model execution on CUDA if available."""
        device = get_device("auto")
        model = ConfigurableMLP(
            input_dim=128,
            hidden_dims=[64, 32],
            output_dim=5,
        ).to(device)

        dummy_x = torch.randn(4, 128, device=device)
        out = model(dummy_x)

        assert out.device.type == device.type
        assert out.shape == (4, 5)

    def test_parameter_counting(self):
        """Verify total and trainable parameter counts."""
        # Simple: (4*8 + 8) + (8*2 + 2) = 40 + 18 = 58
        model = ConfigurableMLP(
            input_dim=4,
            hidden_dims=[8],
            output_dim=2,
            use_batch_norm=False,
        )
        counts = model.count_parameters()
        assert counts["total_parameters"] == 58
        assert counts["trainable_parameters"] == 58

    def test_invalid_parameters_raise_errors(self):
        """Verify invalid configuration arguments raise appropriate ValueError."""
        with pytest.raises(ValueError, match="input_dim must be positive"):
            ConfigurableMLP(input_dim=-1, hidden_dims=[10], output_dim=2)

        with pytest.raises(ValueError, match="output_dim must be positive"):
            ConfigurableMLP(input_dim=10, hidden_dims=[10], output_dim=0)

        with pytest.raises(ValueError, match="dropout must be in range"):
            ConfigurableMLP(input_dim=10, hidden_dims=[10], output_dim=2, dropout=1.5)

        with pytest.raises(ValueError, match="Unsupported activation"):
            ConfigurableMLP(input_dim=10, hidden_dims=[10], output_dim=2, activation="unknown")


class TestConfigurableCNN:
    """Test suite for ConfigurableCNN architecture."""

    def test_default_initialization(self):
        """Test default CNN creation with standard defaults."""
        model = ConfigurableCNN(in_channels=1, num_classes=10)
        assert model.in_channels == 1
        assert model.conv_channels == [32, 64, 128]
        assert model.num_classes == 10
        assert model.activation_name == "relu"

    @pytest.mark.parametrize(
        "in_channels,spatial_size,num_classes",
        [
            (1, (28, 28), 10),  # MNIST / Fashion-MNIST
            (3, (32, 32), 10),  # CIFAR-10
            (3, (64, 64), 100),  # Higher resolution / more classes
        ],
    )
    def test_variable_input_resolutions(self, in_channels, spatial_size, num_classes):
        """Verify CNN accepts variable spatial dimensions via adaptive pooling."""
        batch_size = 4
        model = ConfigurableCNN(
            in_channels=in_channels,
            conv_channels=[16, 32],
            num_classes=num_classes,
        )
        dummy_x = torch.randn(batch_size, in_channels, *spatial_size)
        out = model(dummy_x)

        assert out.shape == (batch_size, num_classes)
        assert not torch.isnan(out).any()

    def test_feature_map_extraction(self):
        """Verify feature map extractor returns pre-pooling spatial features."""
        model = ConfigurableCNN(
            in_channels=1,
            conv_channels=[32, 64],
            kernel_sizes=3,
        )
        dummy_x = torch.randn(2, 1, 28, 28)
        features = model.extract_features(dummy_x)

        # After 2 max-pool stages: 28 -> 14 -> 7
        assert features.shape == (2, 64, 7, 7)

    @pytest.mark.parametrize("activation", ["relu", "gelu", "silu", "leaky_relu"])
    def test_activations_in_cnn(self, activation):
        """Verify different activations operate cleanly in conv blocks."""
        model = ConfigurableCNN(
            in_channels=3,
            conv_channels=[16, 32],
            activation=activation,
            num_classes=5,
        )
        dummy_x = torch.randn(2, 3, 32, 32)
        out = model(dummy_x)
        assert out.shape == (2, 5)

    def test_backward_pass_and_gradient_flow(self):
        """Verify loss backpropagation computes valid gradients for conv and linear layers."""
        model = ConfigurableCNN(
            in_channels=1,
            conv_channels=[16, 32],
            fc_dims=[64],
            num_classes=10,
            use_batch_norm=True,
        )
        dummy_x = torch.randn(4, 1, 28, 28)
        targets = torch.randint(0, 10, (4,))

        criterion = nn.CrossEntropyLoss()
        logits = model(dummy_x)
        loss = criterion(logits, targets)
        loss.backward()

        for name, param in model.named_parameters():
            assert param.grad is not None, f"Gradient missing for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"

    def test_cuda_execution(self):
        """Verify CNN runs on CUDA device if available."""
        device = get_device("auto")
        model = ConfigurableCNN(
            in_channels=3,
            conv_channels=[32, 64],
            num_classes=10,
        ).to(device)

        dummy_x = torch.randn(4, 3, 32, 32, device=device)
        out = model(dummy_x)

        assert out.device.type == device.type
        assert out.shape == (4, 10)

    def test_invalid_parameters_raise_errors(self):
        """Verify invalid arguments raise appropriate ValueError."""
        with pytest.raises(ValueError, match="in_channels must be positive"):
            ConfigurableCNN(in_channels=0)

        with pytest.raises(ValueError, match="num_classes must be positive"):
            ConfigurableCNN(num_classes=-5)

        with pytest.raises(ValueError, match="Length of kernel_sizes"):
            ConfigurableCNN(conv_channels=[32, 64], kernel_sizes=[3, 3, 3])
