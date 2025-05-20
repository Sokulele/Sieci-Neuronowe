import numpy as np

# Funkcja aktywacji - funkcja skoku jednostkowego
def activation_function1(x: float) -> float:
    return 1.0 if x > 0 else 0.0

# Radialna funkcja bazowa (RBF) - funkcja Gaussa
def rbf(x, center, sigma=1.0):
    return np.exp(-np.linalg.norm(x - center)**2 / (2 * sigma**2))

# Transformacja danych przy użyciu RBF
def transform_with_rbf(X, centers, sigma=1.0):
    X_transformed = []
    for x in X:
        x_without_bias = np.array(x[1:])
        rbf_values = [rbf(x_without_bias, center, sigma) for center in centers]
        # Dodanie biasu [1.0] do przekształconych danych
        transformed = [1.0] + rbf_values
        X_transformed.append(transformed)
    return np.array(X_transformed)

# Perceptron z algorytmem BUPA
class PerceptronBUPA:
    def __init__(self):
        self.activation_function = activation_function1
        self.weights = None

    def process_input(self, x):
        return self.activation_function(np.dot(x, self.weights))

    def train_bupa(self, X_train, y_expected, epochs, learning_rate):
        X_train = np.array(X_train)
        y_expected = np.array(y_expected)
        self.weights = np.random.randn(X_train.shape[1])
        self.y_expected = y_expected

        print("Początkowe wagi:", self.weights)
        for epoch in range(epochs):
            print(f"\nEpoka {epoch + 1}:")
            y_pred = np.array([self.process_input(x) for x in X_train])
            errors = y_expected - y_pred
            errors_count = np.sum(errors != 0)

            total_correction = np.zeros_like(self.weights)
            for i, (x, error) in enumerate(zip(X_train, errors)):
                correction = learning_rate * error * x
                total_correction += correction
                print(f"  Przykład {i+1}: x={x.tolist()}, y (oczekiwana) = {y_expected[i]}, "
                      f"f(v)={int(y_pred[i])}, error={error}")

            self.weights = self.weights + total_correction
            print(f"  Nowe wagi po epoce: {self.weights.tolist()}")
            print(f"  Liczba błędów: {errors_count}")

            if errors_count == 0:
                print(f"Uczenie zakończone po {epoch + 1} epokach.")
                break

    def solve_xor_with_rbf(self, X_train, y_expected, centers, sigma, epochs, learning_rate):
        self.X_original = np.array(X_train)
        self.centers = centers
        self.sigma = sigma
        X_transformed = transform_with_rbf(X_train, centers, sigma)

        print("Dane po transformacji RBF:")
        for i, (x_orig, x_trans) in enumerate(zip(X_train, X_transformed)):
            print(f"Oryginalne: {x_orig}, Po transformacji: {x_trans}")

        self.train_bupa(X_transformed, y_expected, epochs, learning_rate)

    def predict(self, X, centers, sigma):
        X_transformed = transform_with_rbf(X, centers, sigma)
        return [self.process_input(x) for x in X_transformed]

# Dane treningowe (x1=bias=1, x2, x3)
x_train = [[1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]]
y_train = [0, 1, 1, 0]

# Centra RBF (bez biasu)
centers = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

# Trening perceptronu
perceptron_bupa = PerceptronBUPA()
perceptron_bupa.solve_xor_with_rbf(
    X_train=x_train, 
    y_expected=y_train, 
    centers=centers, 
    sigma=0.5, 
    epochs=100, 
    learning_rate=1.0
)
