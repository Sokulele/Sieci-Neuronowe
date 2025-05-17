import numpy as np

# Funkcja aktywacji
def activation_function1(x: float) -> float:
    return 1.0 if x > 0 else 0.0

# Perceptron z logowaniem wag
class PerceptronVerbose:
    def __init__(self):
        self.activation_function = activation_function1
        self.weights = np.array([0.5, 0, 1])  # w0 (bias jako x1), w1, w2

    def process_input(self, x: list[float]) -> float:
        return self.activation_function(np.dot(x, self.weights))

    def train(self, X_train: list[list[float]], y_expected: list[float], epochs: int, learning_rate: float) -> None:
        for epoch in range(epochs):
            errors = 0
            print(f"\nEpoka {epoch + 1}:")
            for i, x in enumerate(X_train):
                x = np.array(x)
                y_pred = self.process_input(x)
                error = y_expected[i] - y_pred
                correction = learning_rate * error
                self.weights = self.weights + correction * x
                if error != 0:
                    errors += 1
                print(f"  Przykład {i+1}: x={x.tolist()}, y={y_expected[i]}, y = f(v)={int(y_pred)}, error(d-y) ={error}, wagi={self.weights.tolist()}")
            print(f"Liczba błędów: {errors}")
            if errors == 0:
                print(f"Uczenie zakończone po {epoch + 1} epokach.")
                break

    def predict(self, X: list[list[float]]) -> list[float]:
        return [self.process_input(x) for x in X]


# Dane treningowe (x1=bias=1, x2, x3)
x_train = [[1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]]

y_train = [x_train[0][1] and x_train[0][2],  # AND x2 AND x3
            x_train[1][1] and x_train[1][2],  # AND x2 AND x3
            x_train[2][1] and x_train[2][2],  # AND x2 AND x3
            x_train[3][1] and x_train[3][2]   # AND x2 AND x3 
         ]  

# Trening perceptronu
perceptron_verbose = PerceptronVerbose()
perceptron_verbose.train(x_train, y_train, epochs=10, learning_rate=1.0)

