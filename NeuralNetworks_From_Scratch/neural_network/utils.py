import numpy as np

def accuracy(predictions, labels):
    """
    Calculate accuracy given predictions and true labels.
    
    Args:
        predictions: Softmax probabilities (batch_size, num_classes)
        labels: True labels (batch_size,)
    
    Returns:
        Accuracy as a float between 0 and 1
    """
    predicted_labels = np.argmax(predictions, axis=1)
    return np.mean(predicted_labels == labels)

def one_hot_encode(labels, num_classes):
    """
    Convert labels to one-hot encoding.
    
    Args:
        labels: Integer labels (batch_size,)
        num_classes: Number of classes
    
    Returns:
        One-hot encoded labels (batch_size, num_classes)
    """
    one_hot = np.zeros((len(labels), num_classes))
    one_hot[np.arange(len(labels)), labels] = 1
    return one_hot

def shuffle_data(X, y):
    """
    Shuffle data while maintaining correspondence between X and y.
    """
    permutation = np.random.permutation(X.shape[0])
    return X[permutation], y[permutation]

def create_mini_batches(X, y, batch_size):
    """
    Create mini-batches from the data.
    """
    batches = []
    for i in range(0, X.shape[0], batch_size):
        X_batch = X[i:i+batch_size]
        y_batch = y[i:i+batch_size]
        batches.append((X_batch, y_batch))
    return batches