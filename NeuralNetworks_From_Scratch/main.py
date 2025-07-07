import numpy as np
import matplotlib.pyplot as plt
from neural_network.model import SimpleNN
from neural_network.optimizer import SGD, Adam
from neural_network.utils import accuracy
import torchvision.datasets as datasets

def plot_training_history(model, save_path=None):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot loss
    ax1.plot(model.train_losses, label='Train Loss')
    if model.val_losses:
        ax1.plot(model.val_losses, label='Val Loss')
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracy
    ax2.plot(model.train_accuracies, label='Train Accuracy')
    if model.val_accuracies:
        ax2.plot(model.val_accuracies, label='Val Accuracy')
    ax2.set_title('Training Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def main():
    # Load MNIST data
    print("Loading MNIST data...")
    mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=None)
    mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=None)

    X_train = mnist_trainset.data.numpy().reshape(-1, 28*28) / 255.0
    y_train = mnist_trainset.targets.numpy()

    X_test = mnist_testset.data.numpy().reshape(-1, 28*28) / 255.0
    y_test = mnist_testset.targets.numpy()

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
    plot_training_history(model)

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