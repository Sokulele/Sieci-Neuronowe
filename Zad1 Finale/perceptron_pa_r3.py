import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D

# Funkcja aktywacji
def activation_function1(x: float) -> float:
    return 1.0 if x > 0 else 0.0

# Perceptron z logowaniem wag i wizualizacją granicy decyzyjnej w 3D
class PerceptronVerbose:
    def __init__(self):
        self.activation_function = activation_function1
        self.weights = np.array([0.5, 0, 1])  # w0 (bias), w1, w2
        self.fig = None
        self.ax = None
        self.setup_plot()

    def setup_plot(self):
        """Inicjalizacja wykresu 3D"""
        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_xlim([-0.5, 1.5])
        self.ax.set_ylim([-0.5, 1.5])
        self.ax.set_zlim([-0.5, 1.5])
        self.ax.set_xlabel('x1 (bias)')
        self.ax.set_ylabel('x2')
        self.ax.set_zlabel('x3')
        self.ax.grid(True)

    def plot_decision_boundary(self, epoch, example=None):
        """Rysowanie płaszczyzny decyzyjnej w R³"""
        self.ax.clear()
        self.ax.set_xlim([-0.5, 1.5])
        self.ax.set_ylim([-0.5, 1.5])
        self.ax.set_zlim([-0.5, 1.5])
        self.ax.set_xlabel('x1 (bias)')
        self.ax.set_ylabel('x2')
        self.ax.set_zlabel('x3')
        self.ax.grid(True)

        w = self.weights
        print(f"  Wzór granicy decyzyjnej: {w[0]}*x1 + {w[1]}*x2 + {w[2]}*x3 = 0")

        # Rysowanie płaszczyzny decyzyjnej
        if w[0] != 0:
            x2_vals, x3_vals = np.meshgrid(np.linspace(-0.5, 1.5, 20), np.linspace(-0.5, 1.5, 20))
            x1_vals = -(w[1] * x2_vals + w[2] * x3_vals) / w[0]
            self.ax.plot_surface(x1_vals, x2_vals, x3_vals, alpha=0.3, color='red')
        else:
            print("  Nie można narysować płaszczyzny: w0 = 0")

        # Rysowanie punktów danych
        labels_drawn = set()
        for i, x in enumerate(self.X_train):
            label = f'Klasa {int(self.y_expected[i])}'
            color = 'blue' if self.y_expected[i] == 0 else 'red'
            marker = 'o' if self.y_expected[i] == 0 else '^'
            if label not in labels_drawn:
                self.ax.scatter(x[0], x[1], x[2], c=color, marker=marker, s=100, label=label)
                labels_drawn.add(label)
            else:
                self.ax.scatter(x[0], x[1], x[2], c=color, marker=marker, s=100)

        # Aktualny przykład
        if example is not None:
            x = self.X_train[example]
            self.ax.scatter(x[0], x[1], x[2], color='green', marker='*', s=200, label='Aktualny przykład')

        # Dodanie legendy z elementem dla płaszczyzny
        handles, labels = self.ax.get_legend_handles_labels()
        legend_elements = [Line2D([0], [0], marker='s', color='w', label='Granica decyzyjna',
                                  markerfacecolor='red', markersize=15, alpha=0.3)]
        handles.extend(legend_elements)
        self.ax.legend(handles=handles, loc='best')

        # Tytuł
        title = f'Epoka {epoch+1}'
        if example is not None:
            title += f', Przykład {example+1}'
        title += f'\nWagi: w0={w[0]:.2f}, w1={w[1]:.2f}, w2={w[2]:.2f}'
        self.ax.set_title(title)

        plt.tight_layout()
        plt.draw()
        plt.pause(0.5)

    def process_input(self, x: list[float]) -> float:
        return self.activation_function(np.dot(x, self.weights))

    def train(self, X_train: list[list[float]], y_expected: list[float], epochs: int, learning_rate: float) -> None:
        self.X_train = X_train
        self.y_expected = y_expected

        for epoch in range(epochs):
            errors = 0
            print(f"\nEpoka {epoch + 1}:")
            self.plot_decision_boundary(epoch)

            for i, x in enumerate(X_train):
                x = np.array(x)
                y_pred = self.process_input(x)
                error = y_expected[i] - y_pred
                correction = learning_rate * error
                self.weights = self.weights + correction * x
                if error != 0:
                    errors += 1
                print(f"  Przykład {i+1}: x={x.tolist()}, y (oczek.) = {y_expected[i]}, f(v)={int(y_pred)}, error={error}, wagi={self.weights.tolist()}")
                self.plot_decision_boundary(epoch, example=i)

            print(f"Liczba błędów: {errors}")
            if errors == 0:
                print(f"Uczenie zakończone po {epoch + 1} epokach.")
                self.plot_decision_boundary(epoch)
                plt.savefig('final_decision_boundary_3D.png')
                break

    def predict(self, X: list[list[float]]) -> list[float]:
        return [self.process_input(x) for x in X]

# Dane treningowe (x1=bias=1, x2, x3)
x_train = [[1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]]
y_train = [0, 0, 0, 1]  # Operacja AND: x2 AND x3

# Trening perceptronu
perceptron_verbose = PerceptronVerbose()
perceptron_verbose.train(x_train, y_train, epochs=10, learning_rate=1.0)

# Zakończenie
plt.ioff()
plt.show()
