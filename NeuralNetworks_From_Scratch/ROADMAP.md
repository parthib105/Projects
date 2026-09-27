# Roadmap: Config-Driven PyTorch Deep Learning Framework

> **For coding agents:** This is an ordered, actionable backlog. Work top-to-bottom within each phase unless told otherwise. Each task lists goal, files to touch, approach, and acceptance criteria. Do not skip Phase 0 verification.

**Current state:** Python 3.12 virtual environment (`.nn-venv`) configured with PyTorch 2.6 (`cu124`), CUDA GPU acceleration verified on NVIDIA RTX 3050 Laptop GPU (4 GB VRAM), and core production dependencies (`pyyaml`, `tensorboard`, `gradio`, `onnx`, `onnxruntime`, `scikit-learn`, `pytest`, `ruff`) installed.

### Target Directory Structure
```text
NeuralNetworks_From_Scratch/
├── configs/
│   ├── mnist_cnn.yaml              # Config for MNIST training (CNN / MLP)
│   ├── fashion_mnist.yaml          # Config for Fashion-MNIST
│   └── cifar10_custom.yaml         # Config for CIFAR-10 / Custom datasets
├── src/
│   ├── __init__.py
│   ├── cli.py                      # CLI entry point: train, evaluate, predict, export, demo
│   ├── config.py                   # YAML schema validation & parsing
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py              # PyTorch Dataset & DataLoader pipelines (MNIST, Fashion-MNIST, CIFAR-10, custom folders)
│   │   └── transforms.py           # Data augmentations & normalization
│   ├── models/
│   │   ├── __init__.py
│   │   ├── mlp.py                  # Configurable arbitrary-depth MLP
│   │   ├── cnn.py                  # Modern Deep CNN with BatchNorm & Dropout
│   │   ├── resnet.py               # Custom Residual Networks (ResNet blocks)
│   │   └── factory.py              # Model factory instantiation from config
│   ├── engine/
│   │   ├── __init__.py
│   │   ├── trainer.py              # Production Trainer: AMP (FP16 mixed precision for RTX 3050), Checkpointing, Early Stopping
│   │   ├── evaluator.py            # Evaluation loop & metrics (Accuracy, Top-k, Confusion Matrix)
│   │   └── lr_scheduler.py         # Schedulers (CosineAnnealingLR, OneCycleLR, StepLR)
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── device.py               # GPU memory manager & CUDA device handler for 4GB VRAM
│   │   ├── logger.py               # TensorBoard & structured logging
│   │   └── export.py               # ONNX & TorchScript model export
│   └── demo/
│       └── app.py                  # Interactive Gradio Web App (draw digits/upload images, view predictions & activation heatmaps)
├── tests/                          # Automated Pytest suite
│   ├── test_models.py
│   ├── test_data.py
│   └── test_trainer.py
├── ROADMAP.md                      # Updated real-world roadmap
├── README.md                       # Comprehensive documentation with CLI commands & architecture
```

---

## Phase 0 — Environment & GPU Baseline (verify before building)
- [x] Install Python 3.12 virtual environment via `uv`.
- [x] Install PyTorch with CUDA 12.4 (`torch`, `torchvision`, `torchaudio`).
- [x] Verify CUDA device detection, RTX 3050 Laptop GPU properties, and memory allocation in `.nn-venv`.
- [x] Install all production requirements (`pyyaml`, `tensorboard`, `gradio`, `onnx`, `onnxruntime`, `pytest`, `ruff`).
- [x] Create GPU device manager utility (`src/utils/device.py`) with VRAM monitoring and benchmark flags.

---

## Phase 1 — Core Model Architectures & Model Factory (highest priority)
### 1.1 Configurable Arbitrary-Depth MLP `[Completed]`
- **Goal:** Modular fully connected network supporting arbitrary depth, configurable layer dimensions, activations, dropout, and normalization.
- **Files:** `src/models/mlp.py`, `src/models/__init__.py`, `tests/test_models.py`
- **Approach:** Implement `ConfigurableMLP(nn.Module)` that accepts `input_dim`, `hidden_dims: list[int]`, `output_dim`, `activation='relu'|'gelu'|'silu'|'leaky_relu'`, `dropout: float`, and `use_batch_norm: bool`. Build dynamic `nn.Sequential` stack using PyTorch linear layers.
- **Acceptance:** Can construct and run forward pass for any depth (e.g. `[784, 512, 256, 128, 10]`) without code changes; shapes match `(B, output_dim)`. Tests: 18/18 pytest passing.

### 1.2 Configurable Deep CNN `[Completed]`
- **Goal:** Multi-stage convolutional network with Conv2d, BatchNorm2d, Dropout2d, MaxPool2d, and AdaptiveAvgPool2d for variable input resolutions.
- **Files:** `src/models/cnn.py`, `src/models/__init__.py`, `tests/test_models.py`
- **Approach:** Implement `ConfigurableCNN(nn.Module)` with configurable `in_channels`, `conv_channels: list[int]`, `kernel_sizes: list[int]`, `fc_dims: list[int]`, `num_classes`, and spatial pooling.
- **Acceptance:** Accepts both 1-channel 28x28 (MNIST/Fashion-MNIST) and 3-channel 32x32 (CIFAR-10) inputs; output tensor shape equals `(B, num_classes)`. Tests: 30/30 pytest passing.

### 1.3 Custom ResNet / Residual Blocks
- **Goal:** Deep residual network architecture with skip connections (`ResidualBlock`), bottleneck options, and projection downsampling.
- **Files:** `src/models/resnet.py`, `src/models/__init__.py`
- **Approach:** Implement `ResidualBlock(nn.Module)` with $3\times3$ convolutions, BatchNorm, and optional $1\times1$ projection shortcut for dimension changes. Implement `CustomResNet(nn.Module)` stacking multiple residual stages.
- **Acceptance:** Clean gradient flow through 18+ layers without vanishing gradients; forward pass verified on GPU.

### 1.4 Unified Model Factory
- **Goal:** Instant model instantiation, architecture inspection, and parameter counting via configuration dictionary or YAML string.
- **Files:** `src/models/factory.py`, `src/models/__init__.py`
- **Approach:** Implement `build_model(config: dict) -> nn.Module` that parses model type (`mlp`, `cnn`, `resnet`) and keyword arguments. Add helper `count_parameters(model) -> dict[str, int]` reporting total and trainable parameters.
- **Acceptance:** `build_model({'type': 'cnn', ...})` creates correct instance and logs parameter count accurately.

---

## Phase 2 — Real-World Training Engineering & Pipelines
### 2.1 Config-Driven Runs (YAML & CLI)
- **Goal:** No hardcoded hyperparameters in code; model architecture, dataset, optimizer, learning rate, batch size, epochs, and precision loaded from YAML config files.
- **Files:** `configs/mnist_mlp.yaml`, `configs/mnist_cnn.yaml`, `configs/fashion_mnist.yaml`, `configs/cifar10_resnet.yaml`, `src/config.py`
- **Approach:** Implement `load_config(path: str) -> dict` with Pydantic/dataclass schema validation, default fallback values, and CLI argument override support.
- **Acceptance:** `python -m src.cli train --config configs/mnist_cnn.yaml --epochs 15` runs successfully with CLI overrides applied.

### 2.2 GPU Memory & Mixed Precision Training (AMP)
- **Goal:** Automatic Mixed Precision (`torch.cuda.amp.autocast`) and gradient scaling (`torch.cuda.amp.GradScaler`) for 2x speedup and memory efficiency on RTX 3050 (4GB VRAM).
- **Files:** `src/engine/trainer.py`, `src/utils/device.py`
- **Approach:** Wrap forward pass and loss computation inside `torch.amp.autocast('cuda')`, step optimizer with `GradScaler()`, and implement dynamic batch sizing/gradient accumulation.
- **Acceptance:** GPU VRAM utilization stays under 3.2 GB during full batch training; AMP enabled/disabled via config flag.

### 2.3 Checkpointing + Early Stopping
- **Goal:** Save best validation model checkpoint (`checkpoints/best_model.pt`) and last state; halt training when validation metric plateaus.
- **Files:** `src/engine/trainer.py`, `src/utils/checkpoint.py`
- **Format:** PyTorch dictionary saving `epoch`, `model_state_dict`, `optimizer_state_dict`, `scaler_state_dict`, `val_loss`, `val_acc`, and the input `config`.
- **Acceptance:** Interrupted or completed training can be resumed via `--resume checkpoints/last.pt`; early stopping triggers after `patience` epochs without improvement.

### 2.4 Logging & Experiment Tracking (TensorBoard)
- **Goal:** Comprehensive real-time experiment tracking with TensorBoard summary writer and console progress bars (`tqdm`).
- **Files:** `src/engine/trainer.py`, `src/utils/logger.py`
- **Approach:** Log scalar train/val loss, train/val accuracy, learning rate per step/epoch, and parameter gradient histograms to `runs/<experiment_name>`.
- **Acceptance:** Running `tensorboard --logdir runs/` displays synchronized training and validation curves.

### 2.5 Multi-Dataset Support & Augmentations
- **Goal:** Unified PyTorch `Dataset` & `DataLoader` pipeline supporting MNIST, Fashion-MNIST, CIFAR-10, and generic ImageFolder directories.
- **Files:** `src/data/dataset.py`, `src/data/transforms.py`, `src/data/__init__.py`
- **Approach:** Build `get_dataloaders(dataset_name, batch_size, augment=True, ...)` with train/val/test splits, `torchvision.transforms.v2` (RandomCrop, RandomRotation, ColorJitter, Normalization), `pin_memory=True`, and `num_workers=2`.
- **Acceptance:** `--dataset mnist|fashion_mnist|cifar10|image_folder` works seamlessly; data streams without GPU starvation.

### 2.6 Learning Rate Schedules & Modern Optimizers
- **Goal:** AdamW, Adam, SGD with Nesterov momentum, RMSprop, combined with CosineAnnealingLR, OneCycleLR, and StepLR.
- **Files:** `src/engine/lr_scheduler.py`, `src/engine/optimizer.py`
- **Acceptance:** All optimizers and schedulers instantiated via config name; dynamic learning rate decay logged accurately per step.

---

## Phase 3 — Quality Gates (tests, lint, CI)
### 3.1 Unit Tests for Models & Layers (pytest)
- **Goal:** Pytest unit test suite covering MLP, CNN, ResNet architectures, forward pass output dimensions, and parameter counting.
- **Files:** `tests/test_models.py`, `tests/conftest.py`
- **Acceptance:** `pytest tests/test_models.py` runs green on both CPU and CUDA tensors.

### 3.2 Integration & Training Loop Tests
- **Goal:** End-to-end integration tests verifying data loader batching, AMP training steps, checkpoint save/load round-trips, and evaluator metric accuracy.
- **Files:** `tests/test_data.py`, `tests/test_trainer.py`
- **Acceptance:** Synthetic 2-epoch micro-training test executes and passes without assertions.

### 3.3 Lint / Format / Types
- **Goal:** Strict code formatting and linting via `ruff`, full type hinting across public APIs.
- **Files:** `pyproject.toml`, `src/**/*.py`
- **Acceptance:** `ruff check . && ruff format --check .` passes with zero errors.

### 3.4 Packaging
- **Goal:** Clean installable Python package (`pip install -e .`) with `pyproject.toml` configuration and console script entry point.
- **Files:** `pyproject.toml`, `requirements.txt`
- **Acceptance:** Fresh venv: `pip install -e .` makes `torch-framework` / `python -m src.cli` globally executable.

### 3.5 CI (GitHub Actions)
- **Goal:** Automated GitHub Actions pipeline: install dependencies → run ruff lint → execute pytest suite on push/PR.
- **Files:** `.github/workflows/ci.yml`
- **Acceptance:** Green checkmark on repository pull requests.

### 3.6 Docker (optional)
- **Goal:** NVIDIA CUDA-enabled Docker container for reproducible GPU training and headless evaluation.
- **Files:** `Dockerfile`, `.dockerignore`
- **Acceptance:** `docker build` + `docker run --gpus all` executes training pipeline inside container.

---

## Phase 4 — Demo, Deployment & Portfolio Layer
### 4.1 CLI Tool
- **Goal:** Production command-line interface supporting subcommands: `train`, `eval`, `predict`, `export`, and `demo`.
- **Files:** `src/cli.py`
- **Acceptance:** `python -m src.cli train --config configs/mnist_cnn.yaml`, `python -m src.cli predict --image sample.png --checkpoint checkpoints/best.pt`, and `python -m src.cli export --checkpoint checkpoints/best.pt --format onnx` all work.

### 4.2 Interactive Web Demo (Gradio)
- **Goal:** Interactive browser application featuring:
  - Interactive digit/sketch canvas for live handwritten classification.
  - Image file upload for Fashion-MNIST / CIFAR-10 classification.
  - Top-K probability confidence bar chart.
  - Feature map / activation visualization heatmap.
- **Files:** `src/demo/app.py`
- **Acceptance:** `python -m src.cli demo --checkpoint checkpoints/best_model.pt` opens local browser UI with instant inference.

### 4.3 Visualizations & Evaluation Report
- **Goal:** Generate publication-quality visualization plots: confusion matrix, precision-recall curves, ROC curves, misclassified sample grids, and training loss/accuracy dynamics.
- **Files:** `src/utils/visualizer.py`
- **Acceptance:** Running evaluation exports all plots as high-resolution PNGs in `outputs/plots/`.

### 4.4 ONNX & TorchScript Export & Benchmarks / RESULTS.md
- **Goal:** Export trained PyTorch weights to optimized ONNX and TorchScript formats; verify numerical parity with `onnxruntime`; generate benchmark comparison table in `RESULTS.md`.
- **Files:** `src/utils/export.py`, `RESULTS.md`, `benchmarks.py`
- **Acceptance:** ONNX model outputs match PyTorch outputs within `1e-5` relative tolerance; `RESULTS.md` documents latency, accuracy, and memory benchmarks across architectures.

---

## Phase 5 — Docs Polish
- [ ] README: Professional project overview, architecture diagram, benchmark accuracy table, CLI reference, and live GIF demo of Gradio canvas.
- [ ] Jupyter notebook walkthrough: `notebooks/framework_walkthrough.ipynb` demonstrating config customization, AMP training on RTX 3050, and ONNX export.
- [ ] Keep this ROADMAP updated: check off items and document completion dates as milestones land.

---

## Conventions for Agents
1. **Never violate GPU safety:** always auto-detect device (`cuda` if available else `cpu`), handle memory management, and use `pin_memory=True` appropriately.
2. **Strict configuration integrity:** never hardcode dataset paths or hyperparameters; all settings must flow through YAML configs or CLI overrides.
3. **Reproducibility first:** always seed random number generators (`torch.manual_seed`, `np.random.seed`, `torch.cuda.manual_seed_all`).
4. **TDD & Quality Gates:** write unit tests alongside new architectures, verify outputs with `pytest`, and run `ruff check .` before marking tasks complete.
5. **Clean public APIs:** keep module signatures modular, typed, and well-documented with docstrings.
6. **Commit style:** `feat:`, `fix:`, `test:`, `docs:`, `chore:` prefixes; small, focused commits per phase.

---

## High-Demand Task Suggestions (pick these first for impact)
1. **Configurable Model Suite + Factory** (Phase 1) — builds the core extensible neural network engine.
2. **Config-Driven AMP Training Loop + TensorBoard** (Phase 2 & 3) — production training pipeline leveraging RTX 3050 CUDA cores.
3. **Automated Test Suite + Ruff Linting** (Phase 3) — ensures bulletproof code quality and continuous integration.
4. **ONNX Export & Latency Benchmarks** (Phase 4.4) — industry-standard deployment and optimization proof of work.
5. **Interactive Gradio Drawing Canvas Web App** (Phase 4.2) — visual, interactive demo ready for portfolio and real-world testing.
