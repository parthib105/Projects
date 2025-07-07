import numpy as np
from .activation_functions import ReLu, gradReLu, softmax
from .loss_functions import cross_entropy, grad_cross_entropy
from .utils import accuracy

class SimpleNN:
    def __init__(self, input_size, hidden_size, output_size):
        # Xavier/Glorot initialization
        self.params = {
            'W1': np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size),
            'b1': np.zeros((1, hidden_size)),
            'W2': np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size),
            'b2': np.zeros((1, output_size)),
        }
        
        # For tracking training history
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []

    def forward(self, X):
        self.cache = {}
        self.cache['Z1'] = X @ self.params['W1'] + self.params['b1']
        self.cache['A1'] = ReLu(self.cache['Z1'])
        self.cache['Z2'] = self.cache['A1'] @ self.params['W2'] + self.params['b2']
        self.cache['A2'] = softmax(self.cache['Z2'])
        return self.cache['A2']

    def backward(self, X, y):
        m = X.shape[0]
        dZ2 = grad_cross_entropy(self.cache['A2'], y)
        dW2 = self.cache['A1'].T @ dZ2
        db2 = np.sum(dZ2, axis=0, keepdims=True)

        dA1 = dZ2 @ self.params['W2'].T
        dZ1 = dA1 * gradReLu(self.cache['Z1'])
        dW1 = X.T @ dZ1
        db1 = np.sum(dZ1, axis=0, keepdims=True)

        grads = {'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2}
        return grads

    def update(self, grads, lr):
        for key in self.params:
            self.params[key] -= lr * grads['d' + key]
    
    def predict(self, X):
        """Make predictions on new data"""
        predictions = self.forward(X)
        return np.argmax(predictions, axis=1)
    
    def evaluate(self, X, y):
        """Evaluate model on test/validation data"""
        predictions = self.forward(X)
        loss = cross_entropy(predictions, y)
        acc = accuracy(predictions, y)
        return loss, acc

    def fit(self, X_train, y_train, X_val=None, y_val=None, epochs=10, 
            batch_size=32, learning_rate=0.01, optimizer=None, verbose=True):
        """
        Train the model with optional validation data.
        """
        if optimizer is None:
            from .optimizer import SGD
            optimizer = SGD(learning_rate)
        
        n_samples = X_train.shape[0]
        n_batches = n_samples // batch_size
        
        for epoch in range(epochs):
            # Shuffle training data
            permutation = np.random.permutation(n_samples)
            X_train_shuffled = X_train[permutation]
            y_train_shuffled = y_train[permutation]
            
            epoch_loss = 0
            epoch_acc = 0
            
            # Mini-batch training
            for i in range(0, n_samples, batch_size):
                X_batch = X_train_shuffled[i:i+batch_size]
                y_batch = y_train_shuffled[i:i+batch_size]
                
                # Forward pass
                predictions = self.forward(X_batch)
                loss = cross_entropy(predictions, y_batch)
                
                # Backward pass
                grads = self.backward(X_batch, y_batch)
                
                # Update parameters
                if hasattr(optimizer, 'update'):
                    optimizer.update(self.params, grads)
                else:
                    self.update(grads, learning_rate)
                
                epoch_loss += loss
                epoch_acc += accuracy(predictions, y_batch)
            
            # Average metrics for the epoch
            epoch_loss /= n_batches
            epoch_acc /= n_batches
            
            self.train_losses.append(epoch_loss)
            self.train_accuracies.append(epoch_acc)
            
            # Validation evaluation
            if X_val is not None and y_val is not None:
                val_loss, val_acc = self.evaluate(X_val, y_val)
                self.val_losses.append(val_loss)
                self.val_accuracies.append(val_acc)
                
                if verbose:
                    print(f"Epoch {epoch+1}/{epochs}: "
                          f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}, "
                          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            else:
                if verbose:
                    print(f"Epoch {epoch+1}/{epochs}: "
                          f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}")
