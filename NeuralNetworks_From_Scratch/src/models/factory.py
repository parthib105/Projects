"""Unified model factory for instant neural network instantiation from config."""

from __future__ import annotations

import copy
import logging
import os
import sys
from typing import Any, Dict, Type

# Ensure project root is in sys.path for direct execution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch.nn as nn

from src.models.cnn import ConfigurableCNN
from src.models.mlp import ConfigurableMLP
from src.models.resnet import CustomResNet

logger = logging.getLogger(__name__)

# Registry of supported model architectures
MODEL_REGISTRY: Dict[str, Type[nn.Module]] = {
    "mlp": ConfigurableMLP,
    "cnn": ConfigurableCNN,
    "resnet": CustomResNet,
}


def register_model(name: str, model_cls: Type[nn.Module]) -> None:
    """Register a new custom model architecture into the model factory.

    Args:
        name: Unique string identifier for the model (e.g. 'vision_transformer').
        model_cls: nn.Module class.
    """
    cleaned_name = name.lower().strip()
    if cleaned_name in MODEL_REGISTRY:
        logger.warning(f"Overwriting existing model registration for '{cleaned_name}'.")
    MODEL_REGISTRY[cleaned_name] = model_cls


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Calculate the total and trainable parameter counts of any PyTorch module.

    Args:
        model: PyTorch nn.Module instance.

    Returns:
        Dictionary with 'total_parameters' and 'trainable_parameters'.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total_parameters": total, "trainable_parameters": trainable}


def build_model(config: Dict[str, Any]) -> nn.Module:
    """Instantiate a neural network model from a configuration dictionary.

    Args:
        config: Dictionary containing model specification. Can either be a flat
                dictionary with a 'type' key (e.g. {'type': 'cnn', 'in_channels': 1, ...})
                or a nested dictionary containing a 'model' section.

    Returns:
        Instantiated nn.Module ready for training or evaluation.

    Raises:
        ValueError: If model type is missing or unsupported.
    """
    cfg = copy.deepcopy(config)

    # Support nested {"model": {"type": ...}} or flat {"type": ...}
    if "model" in cfg and isinstance(cfg["model"], dict):
        cfg = cfg["model"]

    # Support 'type' or 'name' as key
    model_type = cfg.pop("type", cfg.pop("name", None))
    if not model_type:
        raise ValueError(
            "Model configuration must specify 'type' or 'name'. "
            f"Supported types: {list(MODEL_REGISTRY.keys())}"
        )

    model_key = str(model_type).lower().strip()
    if model_key not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model type '{model_type}'. "
            f"Available architectures: {list(MODEL_REGISTRY.keys())}"
        )

    model_class = MODEL_REGISTRY[model_key]

    try:
        model = model_class(**cfg)
    except TypeError as e:
        raise TypeError(
            f"Failed to instantiate model '{model_key}' with kwargs {cfg}: {e}"
        ) from e

    params = count_parameters(model)
    logger.info(
        f"Built model '{model_key}' with {params['trainable_parameters']:,} "
        f"trainable parameters (total: {params['total_parameters']:,})."
    )

    return model


if __name__ == "__main__":
    import torch
    from src.utils.device import get_device

    print("=" * 60)
    print(" Unified Model Factory Verification")
    print("=" * 60)

    device = get_device("auto")

    # 1. Build MLP from dictionary config
    mlp_config = {
        "type": "mlp",
        "input_dim": 784,
        "hidden_dims": [256, 128],
        "output_dim": 10,
        "activation": "gelu",
        "dropout": 0.2,
    }
    mlp = build_model(mlp_config).to(device)
    print(f"\n[1] Built MLP from config: {mlp.__class__.__name__}")
    print(f"    Params: {count_parameters(mlp)}")
    out_mlp = mlp(torch.randn(4, 784, device=device))
    print(f"    Forward pass output shape: {tuple(out_mlp.shape)}")

    # 2. Build CNN from dictionary config
    cnn_config = {
        "type": "cnn",
        "in_channels": 1,
        "conv_channels": [32, 64],
        "fc_dims": [64],
        "num_classes": 10,
    }
    cnn = build_model(cnn_config).to(device)
    print(f"\n[2] Built CNN from config: {cnn.__class__.__name__}")
    print(f"    Params: {count_parameters(cnn)}")
    out_cnn = cnn(torch.randn(4, 1, 28, 28, device=device))
    print(f"    Forward pass output shape: {tuple(out_cnn.shape)}")

    # 3. Build ResNet from dictionary config
    resnet_config = {
        "type": "resnet",
        "in_channels": 3,
        "num_classes": 10,
        "block_counts": [1, 1],
        "stage_channels": [32, 64],
        "small_inputs": True,
    }
    resnet = build_model(resnet_config).to(device)
    print(f"\n[3] Built ResNet from config: {resnet.__class__.__name__}")
    print(f"    Params: {count_parameters(resnet)}")
    out_resnet = resnet(torch.randn(4, 3, 32, 32, device=device))
    print(f"    Forward pass output shape: {tuple(out_resnet.shape)}")

    print("\n[4] All factory builds and forward passes: PASS")
    print("=" * 60)
