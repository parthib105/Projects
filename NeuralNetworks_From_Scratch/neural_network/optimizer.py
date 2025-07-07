import numpy as np

class SGD:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
    
    def update(self, params, grads):
        for key in params:
            params[key] -= self.learning_rate * grads['d' + key]

class Adam:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}
        self.v = {}
        self.t = 0
    
    def update(self, params, grads):
        self.t += 1
        
        # Initialize momentum and velocity if first iteration
        if not self.m:
            for key in params:
                self.m['d' + key] = np.zeros_like(params[key])
                self.v['d' + key] = np.zeros_like(params[key])
        
        for key in params:
            grad_key = 'd' + key
            
            # Update biased first moment estimate
            self.m[grad_key] = self.beta1 * self.m[grad_key] + (1 - self.beta1) * grads[grad_key]
            
            # Update biased second raw moment estimate
            self.v[grad_key] = self.beta2 * self.v[grad_key] + (1 - self.beta2) * (grads[grad_key] ** 2)
            
            # Compute bias-corrected first moment estimate
            m_corrected = self.m[grad_key] / (1 - self.beta1 ** self.t)
            
            # Compute bias-corrected second raw moment estimate
            v_corrected = self.v[grad_key] / (1 - self.beta2 ** self.t)
            
            # Update parameters
            params[key] -= self.learning_rate * m_corrected / (np.sqrt(v_corrected) + self.epsilon)
