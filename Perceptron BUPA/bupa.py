import numpy as np

class BatchPerceptron:
    def __init__(self, learning_rate=0.01, epochs=50):
        self.eta = learning_rate
        self.epochs = epochs
    
    def train(self, X, y):
        # Initialize weights (one weight per feature)
        self.w_ = np.zeros(X.shape[1])
        self.errors_ = []
        
        for epoch in range(self.epochs):
            batch_updates = np.zeros(X.shape[1])
            errors = 0
            
            # Collect updates from all misclassified examples
            for xi, target in zip(X, y):
                prediction = self.predict(xi)
                error = target - prediction
                
                # If misclassified
                if error != 0:
                    batch_updates += error * xi  # Accumulate updates
                    errors += 1
            
            # Apply batch update to weights
            self.w_ += self.eta * batch_updates
            self.errors_.append(errors)
            
            # Stop if no errors
            if errors == 0:
                break
                
        return self
    
    def net_input(self, X):
        return np.dot(X, self.w_)
    
    def predict(self, X):
        return np.where(self.net_input(X) > 0.0, 1, 0)