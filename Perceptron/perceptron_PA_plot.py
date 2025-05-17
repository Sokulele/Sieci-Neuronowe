import numpy as np
import matplotlib.pyplot as plt

# Funkcja aktywacji
def activation_function1(x: float) -> float:
    return 1.0 if x > 0 else 0.0

# Perceptron z logowaniem wag i wizualizacją granicy decyzyjnej
class PerceptronVerbose:
    def __init__(self):
        self.activation_function = activation_function1
        self.weights = np.array([0.5, 0, 1])  # w0 (bias jako x1), w1, w2
        self.fig = None
        self.ax = None
        self.setup_plot()
        
    def setup_plot(self):
        """Inicjalizacja wykresu"""
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.ax.set_xlim([-0.5, 1.5])
        self.ax.set_ylim([-0.5, 1.5])
        self.ax.set_xlabel('x2')
        self.ax.set_ylabel('x3')
        self.ax.grid(True)
    
    def plot_decision_boundary(self, epoch, example=None):
        """Rysowanie granicy decyzyjnej"""
        # Czyszczenie wykresu
        self.ax.clear()
        
        # Ustawienie limitu osi
        self.ax.set_xlim([-0.5, 1.5])
        self.ax.set_ylim([-0.5, 1.5])
        self.ax.set_xlabel('x2')
        self.ax.set_ylabel('x3')
        self.ax.grid(True)
        
        # Równanie granicy decyzyjnej: w0 + w1*x2 + w2*x3 = 0
        # Przekształcamy to do postaci x3 = -(w0 + w1*x2)/w2
        if self.weights[2] != 0:  # Sprawdzamy, czy w2 nie jest zerem
            x2_values = np.linspace(-0.5, 1.5, 100)
            x3_values = -(self.weights[0] + self.weights[1] * x2_values) / self.weights[2]
            self.ax.plot(x2_values, x3_values, 'r-', label='Granica decyzyjna')
        else:
            # W przypadku gdy w2=0, granica jest pionowa linia
            x2_boundary = -self.weights[0] / self.weights[1] if self.weights[1] != 0 else None
            if x2_boundary is not None:
                self.ax.axvline(x=x2_boundary, color='r', label='Granica decyzyjna')
        
        # Rysowanie punktów danych treningowych
        colors = ['blue', 'red']
        markers = ['o', 'x']
        
        for i, x in enumerate(self.X_train):
            color_idx = int(self.y_expected[i])
            self.ax.scatter(x[1], x[2], color=colors[color_idx], marker=markers[color_idx], 
                          s=100, label=f'Klasa {int(self.y_expected[i])}')
        
        # Wyróżnienie aktualnego przykładu, jeśli podano
        if example is not None:
            x = self.X_train[example]
            self.ax.scatter(x[1], x[2], color='green', marker='*', s=200, 
                          label=f'Aktualna iteracja {example+1}')
        
        # Dodanie legendy bez duplikatów
        handles, labels = self.ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        self.ax.legend(by_label.values(), by_label.keys(), loc='best')
        
        # Tytuł wykresu
        title = f'Epoka {epoch+1}'
        if example is not None:
            title += f', Iteracja {example+1}'
        title += f'\nWagi: w0={self.weights[0]:.2f}, w1={self.weights[1]:.2f}, w2={self.weights[2]:.2f}'
        self.ax.set_title(title)
        
        # Wyświetlenie wykresu
        plt.tight_layout()
        plt.draw()
        plt.pause(0.5)  # Pauza aby zobaczyć zmiany
    
    def process_input(self, x: list[float]) -> float:
        return self.activation_function(np.dot(x, self.weights))

    def train(self, X_train: list[list[float]], y_expected: list[float], epochs: int, learning_rate: float) -> None:
        self.X_train = X_train
        self.y_expected = y_expected
        
        for epoch in range(epochs):
            errors = 0
            print(f"\nEpoka {epoch + 1}:")
            
            # Rysowanie granicy decyzyjnej na początku epoki
            self.plot_decision_boundary(epoch)
            
            for i, x in enumerate(X_train):
                x = np.array(x)
                y_pred = self.process_input(x)
                error = y_expected[i] - y_pred
                correction = learning_rate * error
                self.weights = self.weights + correction * x
                if error != 0:
                    errors += 1
                print(f"  Iteracja {i+1}: x={x.tolist()}, y={y_expected[i]}, y = f(v)={int(y_pred)}, error(d-y) ={error}, wagi={self.weights.tolist()}")
                
                # Rysowanie granicy decyzyjnej po każdej aktualizacji wag
                self.plot_decision_boundary(epoch, example=i)
            
            print(f"Liczba błędów: {errors}")
            if errors == 0:
                print(f"Uczenie zakończone po {epoch + 1} epokach.")
                # Rysowanie finalnej granicy decyzyjnej
                self.plot_decision_boundary(epoch)
                plt.savefig('final_decision_boundary.png')
                break

    def predict(self, X: list[list[float]]) -> list[float]:
        return [self.process_input(x) for x in X]


# Dane treningowe (x1=bias=1, x2, x3)
x_train = [[1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]]

# Operacja logiczna AND
y_train = [0, 0, 0, 1]  # AND: x2 AND x3

# Trening perceptronu
perceptron_verbose = PerceptronVerbose()
perceptron_verbose.train(x_train, y_train, epochs=10, learning_rate=1.0)

# Zachowanie wykresu na końcu
plt.ioff()  # Wyłączenie trybu interaktywnego
plt.show()  # Pokazanie ostatecznego wykresu