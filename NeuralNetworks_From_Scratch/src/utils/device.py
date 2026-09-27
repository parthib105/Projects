"""GPU and compute device manager with VRAM safety monitoring.

Tailored for consumer/laptop GPUs (such as NVIDIA GeForce RTX 3050 4GB)
to ensure automatic mixed precision compatibility, cuDNN optimization,
and Out-Of-Memory (OOM) prevention.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import torch

logger = logging.getLogger(__name__)


def get_device(preference: str = "auto") -> torch.device:
    """Resolve and return the appropriate torch.device.

    Args:
        preference: "auto", "cuda", "cuda:N", or "cpu".

    Returns:
        torch.device instance configured with optimal backend flags.
    """
    preference = preference.lower().strip()

    if preference == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
    elif preference.startswith("cuda"):
        if not torch.cuda.is_available():
            logger.warning("CUDA requested but not available. Falling back to CPU.")
            device = torch.device("cpu")
        else:
            device = torch.device(preference)
    elif preference == "cpu":
        device = torch.device("cpu")
    else:
        raise ValueError(
            f"Invalid device preference '{preference}'. Use 'auto', 'cuda', 'cuda:N', or 'cpu'."
        )

    # Enable optimal cuDNN benchmarks when running on CUDA
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        logger.info(
            f"Active compute device: {torch.cuda.get_device_name(device)} ({device})"
        )
    else:
        logger.info("Active compute device: CPU")

    return device


def get_device_info(device: Optional[torch.device] = None) -> Dict[str, Any]:
    """Retrieve hardware and runtime metadata for the target device.

    Args:
        device: Target device. If None, resolves via get_device("auto").

    Returns:
        Dictionary containing device metadata (name, compute capability, VRAM).
    """
    if device is None:
        device = get_device("auto")

    if device.type != "cuda":
        return {
            "device_type": "cpu",
            "device_name": "Host CPU",
            "cuda_available": False,
        }

    device_idx = device.index if device.index is not None else 0
    props = torch.cuda.get_device_properties(device_idx)
    total_memory_gb = props.total_memory / (1024**3)

    # Safely retrieve CUDA version string without throwing exceptions
    cuda_ver = getattr(getattr(torch, "version", None), "cuda", None)

    return {
        "device_type": "cuda",
        "device_index": device_idx,
        "device_name": props.name,
        "cuda_version": cuda_ver or "Unknown",
        "major_capability": props.major,
        "minor_capability": props.minor,
        "multi_processor_count": props.multi_processor_count,
        "total_memory_gb": round(total_memory_gb, 2),
        "total_memory_mb": round(props.total_memory / (1024**2), 1),
    }


def get_memory_summary(device: Optional[torch.device] = None) -> Dict[str, float]:
    """Return memory metrics (allocated, reserved, free) in megabytes (MB).

    Args:
        device: Target device. If None or CPU, returns zeros.

    Returns:
        Dictionary of allocated_mb, reserved_mb, total_mb, and free_mb.
    """
    if device is None:
        device = get_device("auto")

    if device.type != "cuda":
        return {
            "allocated_mb": 0.0,
            "reserved_mb": 0.0,
            "total_mb": 0.0,
            "free_mb": 0.0,
        }

    device_idx = device.index if device.index is not None else 0
    props = torch.cuda.get_device_properties(device_idx)
    total_bytes = props.total_memory
    allocated_bytes = torch.cuda.memory_allocated(device_idx)
    reserved_bytes = torch.cuda.memory_reserved(device_idx)
    free_bytes = total_bytes - reserved_bytes

    return {
        "allocated_mb": round(allocated_bytes / (1024**2), 2),
        "reserved_mb": round(reserved_bytes / (1024**2), 2),
        "total_mb": round(total_bytes / (1024**2), 2),
        "free_mb": round(free_bytes / (1024**2), 2),
    }


def check_memory_safety(
    threshold_ratio: float = 0.85,
    device: Optional[torch.device] = None,
) -> bool:
    """Check if GPU memory utilization is within safe operational limits.

    Warns if reserved memory exceeds threshold_ratio of total VRAM (e.g., 3.4 GB on 4 GB).

    Args:
        threshold_ratio: Upper limit ratio (0.0 to 1.0) before triggering a warning.
        device: Target device.

    Returns:
        True if memory is within safe limit, False if approaching OOM danger.
    """
    summary = get_memory_summary(device)
    if summary["total_mb"] == 0:
        return True

    usage_ratio = summary["reserved_mb"] / summary["total_mb"]
    if usage_ratio >= threshold_ratio:
        logger.warning(
            f"VRAM warning: Reserved {summary['reserved_mb']:.1f} MB / "
            f"{summary['total_mb']:.1f} MB ({usage_ratio * 100:.1f}%). "
            f"Approaching safe threshold of {threshold_ratio * 100:.0f}%."
        )
        return False
    return True


def clear_gpu_cache() -> None:
    """Release unallocated cached memory held by PyTorch back to the OS."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        logger.debug("GPU cache flushed successfully.")


if __name__ == "__main__":
    import sys

    print("=" * 60)
    print(" PyTorch Device & GPU Verification Utility")
    print("=" * 60)

    # 1. PyTorch Runtime info
    print(f"[*] Python Version   : {sys.version.split()[0]}")
    print(f"[*] PyTorch Version  : {torch.__version__}")
    print(f"[*] CUDA Available   : {torch.cuda.is_available()}")

    cudnn_ver = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else "N/A"
    print(f"[*] cuDNN Version    : {cudnn_ver}")

    # 2. Resolve target device
    device = get_device("auto")
    print(f"[*] Resolved Device  : {device}")

    # 3. Hardware metadata
    print("\n--- Device Information ---")
    info = get_device_info(device)
    for key, val in info.items():
        print(f"  {key:<22}: {val}")

    # 4. Memory breakdown
    print("\n--- Memory Status (Before Allocation) ---")
    mem_before = get_memory_summary(device)
    for key, val in mem_before.items():
        print(f"  {key:<22}: {val} MB")

    # 5. Live Tensor Allocation Test on GPU
    if device.type == "cuda":
        print("\n--- Live GPU Tensor Test ---")
        print("  Allocating a test tensor of shape (2048, 2048) on CUDA...")
        test_tensor = torch.randn(2048, 2048, device=device)
        result = (test_tensor @ test_tensor).sum().item()
        print(f"  Matrix multiplication result checksum: {result:.4f}")

        mem_during = get_memory_summary(device)
        print(f"  Allocated VRAM: {mem_during['allocated_mb']} MB")
        print(f"  Reserved VRAM : {mem_during['reserved_mb']} MB")

        # Cleanup
        del test_tensor
        clear_gpu_cache()
        mem_after = get_memory_summary(device)
        print(f"  Allocated after cache clear: {mem_after['allocated_mb']} MB")

    # 6. Safety Guard Check
    print("\n--- Memory Safety Verification ---")
    is_safe = check_memory_safety(0.85, device)
    print(f"  Memory Safety Guard (<= 85% VRAM): {'PASS (Safe)' if is_safe else 'ALERT (Near Limit)'}")
    print("=" * 60)
