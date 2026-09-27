import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from neural_network.model import SimpleNN
from neural_network.optimizer import SGD, Adam
from neural_network.utils import accuracy
from datasets import load_dataset

def load_mnist(data_dir: str = './data', cache_name: str = 'mnist.npz') -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load MNIST from HuggingFace datasets, cached as .npz in data_dir."""
    cache_path = os.path.join(data_dir, cache_name)

    if os.path.exists(cache_path):
        print("Loading MNIST from local cache...")
        data = np.load(cache_path)
        return data['X_train'], data['y_train'], data['X_test'], data['y_test']

    print("Downloading MNIST from HuggingFace...")
    ds = load_dataset("ylecun/mnist")

    def to_xy(split) -> tuple[np.ndarray, np.ndarray]:
        images = np.array(
            [np.array(img).reshape(-1) for img in split["image"]],
            dtype=np.float32
        ) / 255.0
        labels = np.array(split["label"], dtype=np.int64)
        return images, labels

    X_train, y_train = to_xy(ds["train"])
    X_test, y_test = to_xy(ds["test"])

    np.savez_compressed(cache_path, X_train=X_train, y_train=y_train,
                        X_test=X_test, y_test=y_test)
    print(f"Cached dataset to {cache_path}")
    return X_train, y_train, X_test, y_test


def plot_training_history(model: SimpleNN, save_path: str | None = None) -> None:
    """Plot training history"""
    params: tuple[Figure, list[Axes]] = plt.subplots(1, 2, figsize=(12, 4))
    
    ax: list[Axes] = params[1]
    
    # Plot loss
    ax[0].plot(model.train_losses, label='Train Loss')
    if model.val_losses:
        ax[0].plot(model.val_losses, label='Val Loss')
    ax[0].set_title('Training Loss')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Loss')
    ax[0].legend()
    ax[0].grid(True)
    
    # Plot accuracy
    ax[1].plot(model.train_accuracies, label='Train Accuracy')
    if model.val_accuracies:
        ax[1].plot(model.val_accuracies, label='Val Accuracy')
    ax[1].set_title('Training Accuracy')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Accuracy')
    ax[1].legend()
    ax[1].grid(True)
    
    plt.tight_layout()
    
    save_fig: str = "./path/training_hist"
    if save_path:
        save_fig = os.path.join(save_path, "training_hist")

    plt.savefig(save_fig)
    

def main() -> None:
    # Load MNIST data from HuggingFace (with local caching)
    print("Loading MNIST data...")
    X_train, y_train, X_test, y_test = load_mnist()

    # Create validation set from training data
    val_size = 10000
    X_val = X_train[:val_size]
    y_val = y_train[:val_size]
    X_train = X_train[val_size:]
    y_train = y_train[val_size:]

    print(f"Training set size: {X_train.shape[0]}")
    print(f"Validation set size: {X_val.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")

    # Initialize model
    model = SimpleNN(input_size=784, hidden_size=128, output_size=10)
    
    # Choose optimizer
    optimizer = Adam(learning_rate=0.001)  # Try Adam optimizer
    # optimizer = SGD(learning_rate=0.01)   # Or use SGD
    
    # Training parameters
    epochs = 10
    batch_size = 128

    print(f"\nTraining with {optimizer.__class__.__name__} optimizer...")
    print(f"Epochs: {epochs}, Batch size: {batch_size}")
    
    # Train the model
    model.fit(
        X_train, y_train, 
        X_val, y_val,
        epochs=epochs,
        batch_size=batch_size,
        optimizer=optimizer,
        verbose=True
    )

    # Final evaluation on test set
    print("\nEvaluating on test set...")
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

    # Plot training history
    plot_training_history(model, "./plots")

    # Example predictions
    print("\nSample predictions:")
    sample_indices = np.random.choice(len(X_test), 5)
    sample_X = X_test[sample_indices]
    sample_y = y_test[sample_indices]
    predictions = model.predict(sample_X)
    
    for i, (true_label, pred_label) in enumerate(zip(sample_y, predictions)):
        print(f"Sample {i+1}: True={true_label}, Predicted={pred_label}")

if __name__ == "__main__":
    main()