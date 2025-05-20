import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Funkcja aktywacji
def activation_function(x: float) -> float:
    return 1.0 if x > 0 else 0.0

# Perceptron z logowaniem wag i wizualizacją granicy decyzyjnej w 3D
class Perceptron3D:
    def __init__(self):
        self.activation_function = activation_function
        # w0 (dla biasu), w1, w2, w3
        self.weights = np.array([0.5, 0, 0, 1])
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
        self.ax.set_xlabel('x1')
        self.ax.set_ylabel('x2')
        self.ax.set_zlabel('x3')
        self.ax.grid(True)
    
    def plot_decision_boundary(self, epoch, example=None):
        """Rysowanie granicy decyzyjnej w 3D (płaszczyzna)"""
        # Czyszczenie wykresu
        self.ax.clear()
        
        # Ustawienie limitu osi
        self.ax.set_xlim([-0.5, 1.5])
        self.ax.set_ylim([-0.5, 1.5])
        self.ax.set_zlim([-0.5, 1.5])
        self.ax.set_xlabel('x1')
        self.ax.set_ylabel('x2')
        self.ax.set_zlabel('x3')
        self.ax.grid(True)
        
        # Równanie płaszczyzny decyzyjnej: w0 + w1*x1 + w2*x2 + w3*x3 = 0
        # Przekształcamy do postaci: x3 = -(w0 + w1*x1 + w2*x2) / w3
        
        if self.weights[3] != 0:  # Sprawdzamy, czy w3 nie jest zerem
            # Tworzymy siatkę punktów dla x1 i x2
            x1_values = np.linspace(-0.5, 1.5, 10)
            x2_values = np.linspace(-0.5, 1.5, 10)
            x1_grid, x2_grid = np.meshgrid(x1_values, x2_values)
            
            # Obliczamy odpowiadające wartości x3
            x3_grid = -(self.weights[0] + self.weights[1] * x1_grid + self.weights[2] * x2_grid) / self.weights[3]
            
            # Rysowanie płaszczyzny decyzyjnej
            surface = self.ax.plot_surface(x1_grid, x2_grid, x3_grid, alpha=0.5, 
                                        color='lightblue', label='Granica decyzyjna')
            
            formula = f"x3 = -({self.weights[0]:.2f} + {self.weights[1]:.2f}*x1 + {self.weights[2]:.2f}*x2) / {self.weights[3]:.2f}"
            print(f"  Wzór granicy decyzyjnej: {formula}")
        else:
            # W przypadku gdy w3=0, granica jest płaszczyzną równoległą do osi Z
            print("  Uwaga: w3=0, granica decyzyjna jest płaszczyzną równoległą do osi x3")
            # Nie rysujemy powierzchni w tym przypadku, ale możemy oznaczyć to w inny sposób
        
        # Rysowanie punktów danych treningowych
        colors = ['blue', 'red']
        markers = ['o', 'x']
        
        for i, x in enumerate(self.X_train):
            color_idx = int(self.y_expected[i])
            self.ax.scatter(x[1], x[2], x[3], color=colors[color_idx], marker=markers[color_idx], 
                          s=100, label=f'Klasa {int(self.y_expected[i])}')
        
        # Wyróżnienie aktualnego przykładu, jeśli podano
        if example is not None:
            x = self.X_train[example]
            self.ax.scatter(x[1], x[2], x[3], color='green', marker='*', s=200, 
                          label=f'Aktualny przykład {example+1}')
        
        # Dodanie legendy bez duplikatów
        handles, labels = [], []
        # W matplotlib 3D nie możemy bezpośrednio uzyskać handles jak w 2D
        # Zamiast tego dodajemy tylko etykiety klas
        if len(self.X_train) > 0:
            # Dodajemy ręcznie etykiety dla klas
            custom_handles = [
                plt.Line2D([0], [0], marker='o', color='blue', linestyle='None', markersize=10),
                plt.Line2D([0], [0], marker='x', color='red', linestyle='None', markersize=10)
            ]
            custom_labels = ['Klasa 0', 'Klasa 1']
            
            if example is not None:
                custom_handles.append(plt.Line2D([0], [0], marker='*', color='green', linestyle='None', markersize=10))
                custom_labels.append(f'Aktualny przykład {example+1}')
                
            self.ax.legend(custom_handles, custom_labels, loc='upper right')
        
        # Tytuł wykresu
        title = f'Epoka {epoch+1}'
        if example is not None:
            title += f', Przykład {example+1}'
        title += f'\nWagi: w0={self.weights[0]:.2f}, w1={self.weights[1]:.2f}, w2={self.weights[2]:.2f}, w3={self.weights[3]:.2f}'
        self.ax.set_title(title)
        
        # Wyświetlenie wykresu
        plt.tight_layout()
        plt.draw()
        plt.pause(0.5)  # Pauza aby zobaczyć zmiany
    
    def process_input(self, x: list[float]) -> float:
        """Oblicza wartość wyjściową perceptronu dla podanego wektora wejściowego
            Iloczyn sklarany wag i danych wejsciowych / dot product """
        return self.activation_function(np.dot(x, self.weights))

    def train(self, X_train: list[list[float]], y_expected: list[float], epochs: int, learning_rate: float) -> None:
        """Trenuje perceptron na danych treningowych
        Args:
            X_train: Lista wektorów wejściowych. Pierwszy element każdego wektora to bias.
            y_expected: Lista oczekiwanych wyjść dla każdego wektora wejściowego
            epochs: Maksymalna liczba epok uczenia
            learning_rate: Współczynnik uczenia (szybkość uczenia)
        """
        
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
                print(f"  Przykład {i+1}: x={x.tolist()}, y (wartość oczekiwana) = {y_expected[i]}, "
                      f"f(v)={int(y_pred)}, error(d-y) ={error}, wagi={self.weights.tolist()}")
                
                # Rysowanie granicy decyzyjnej po każdej aktualizacji wag
                self.plot_decision_boundary(epoch, example=i)
            
            print(f"Liczba błędów: {errors}")
            if errors == 0:
                print(f"Uczenie zakończone po {epoch + 1} epokach.")
                # Rysowanie finalnej granicy decyzyjnej
                self.plot_decision_boundary(epoch)
                plt.savefig('final_decision_boundary_3d.png')
                break

    def predict(self, X: list[list[float]]) -> list[float]:
        return [self.process_input(x) for x in X]


# Przykład użycia - klasyfikacja operacji logicznej w przestrzeni 3D
# Dane treningowe (x0=bias=1, x1, x2, x3)
x_train = [
    [1, 0, 0, 0],  # (0,0,0)
    [1, 0, 0, 1],  # (0,0,1)
    [1, 0, 1, 0],  # (0,1,0)
    [1, 0, 1, 1],  # (0,1,1)
    [1, 1, 0, 0],  # (1,0,0)
    [1, 1, 0, 1],  # (1,0,1)
    [1, 1, 1, 0],  # (1,1,0)
    [1, 1, 1, 1]   # (1,1,1)
]

# Operacja logiczna AND dla trzech zmiennych (x1 AND x2 AND x3)
y_train = [0, 0, 0, 0, 0, 0, 0, 1]  # tylko dla (1,1,1) wynik jest 1

# Trening perceptronu
perceptron = Perceptron3D()
perceptron.train(x_train, y_train, epochs=10, learning_rate=1.0)

# Zachowanie wykresu na końcu
plt.ioff()  # Wyłączenie trybu interaktywnego
plt.show()  # Pokazanie ostatecznego wykresu