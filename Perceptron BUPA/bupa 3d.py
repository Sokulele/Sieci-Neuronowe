import numpy as np
import matplotlib.pyplot as plt

# Funkcja aktywacji
def activation_function1(x: float) -> float:
    return 1.0 if x > 0 else 0.0

# Batch Update Perceptron Algorithm (BUPA) z logowaniem wag i wizualizacją granicy decyzyjnej w przestrzeni 3D
class BUPAPerceptronVerbose3D:
    def __init__(self):
        self.activation_function = activation_function1
        self.weights = np.array([0.5, 0, 1])  # w0 (dla biasu), w1, w2
        self.fig = None
        self.ax = None
        self.setup_plot()
        
    def setup_plot(self):
        """Inicjalizacja wykresu 3D"""
        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_xlim([0, 1])  # bias (x1) zawsze 1, ale pokazujemy dla wizualizacji
        self.ax.set_ylim([-0.5, 1.5])  # x2
        self.ax.set_zlim([-0.5, 1.5])  # x3
        self.ax.set_xlabel('x1 (bias)')
        self.ax.set_ylabel('x2')
        self.ax.set_zlabel('x3')
        self.ax.grid(True)
    
    def plot_decision_boundary_3d(self, epoch, example=None):
        """Rysowanie granicy decyzyjnej w przestrzeni 3D"""
        # Czyszczenie wykresu
        self.ax.clear()
        
        # Ustawienie limitu osi
        self.ax.set_xlim([0, 1])  # Ograniczamy oś x1 (bias) od 0 do 1
        self.ax.set_ylim([-0.5, 1.5])  # x2
        self.ax.set_zlim([-0.5, 1.5])  # x3
        self.ax.set_xlabel('x1 (bias)')
        self.ax.set_ylabel('x2')
        self.ax.set_zlabel('x3')
        self.ax.grid(True)
        
        # Rysowanie płaszczyzny decyzyjnej: w0*x1 + w1*x2 + w2*x3 = 0
        # Przekształcamy do postaci: x3 = -(w0*x1 + w1*x2)/w2
        
        if self.weights[2] != 0:  # Upewniamy się, że w2 nie jest zerem
            # Tworzenie siatki punktów dla x1 (bias) i x2
            x1_points = np.array([0, 1])  # Pokazujemy tylko x1=0 i x1=1 dla lepszej wizualizacji
            x2_points = np.array([-0.5, 1.5])
            x1_grid, x2_grid = np.meshgrid(x1_points, x2_points)
            
            # Obliczanie odpowiadających wartości x3 dla płaszczyzny decyzyjnej
            x3_grid = -(self.weights[0] * x1_grid + self.weights[1] * x2_grid) / self.weights[2]
            
            # Rysowanie płaszczyzny decyzyjnej
            self.ax.plot_surface(x1_grid, x2_grid, x3_grid, alpha=0.5, color='red')
            print(f"  Równanie płaszczyzny decyzyjnej: {self.weights[0]}*x1 + {self.weights[1]}*x2 + {self.weights[2]}*x3 = 0")
            print(f"  Przekształcone: x3 = -({self.weights[0]}*x1 + {self.weights[1]}*x2) / {self.weights[2]}")
        else:
            # W przypadku gdy w2=0, płaszczyzna jest prostopadła do osi x2
            print(f"  Płaszczyzna prostopadła do osi x2, równanie: {self.weights[0]}*x1 + {self.weights[1]}*x2 = 0")
        
        # Rysowanie punktów danych treningowych w 3D
        colors = ['blue', 'red']
        markers = ['o', 'x']
        
        for i, x in enumerate(self.X_train):
            color_idx = int(self.y_expected[i])
            # x[0] to bias (zawsze 1), x[1] to x2, x[2] to x3
            self.ax.scatter(x[0], x[1], x[2], color=colors[color_idx], marker=markers[color_idx], 
                           s=100, label=f'Klasa {int(self.y_expected[i])}')
        
        # Wyróżnienie aktualnego przykładu, jeśli podano
        if example is not None:
            x = self.X_train[example]
            self.ax.scatter(x[0], x[1], x[2], color='green', marker='*', s=200, 
                           label=f'Aktualny przykład {example+1}')
        
        # Dodanie legendy bez duplikatów
        handles, labels = self.ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        self.ax.legend(by_label.values(), by_label.keys(), loc='best')
        
        # Tytuł wykresu
        title = f'Epoka {epoch+1}'
        if example is not None:
            title += f', Przykład {example+1}'
        title += f'\nWagi: w0={self.weights[0]:.2f}, w1={self.weights[1]:.2f}, w2={self.weights[2]:.2f}'
        self.ax.set_title(title)
        
        # Wyświetlenie wykresu
        plt.tight_layout()
        plt.draw()
        plt.pause(0.5)  # Pauza aby zobaczyć zmiany
    
    def process_input(self, x: list[float]) -> float:
        """Oblicza wartość wyjściową perceptronu dla podanego wektora wejściowego"""
        return self.activation_function(np.dot(x, self.weights))

    def train(self, X_train: list[list[float]], y_expected: list[float], epochs: int, learning_rate: float) -> None:
        """Trenuje perceptron metodą BUPA (Batch Update Perceptron Algorithm) na danych treningowych
        
        Args:
            X_train: Lista wektorów wejściowych. Pierwszy element każdego wektora to bias.
            y_expected: Lista oczekiwanych wyjść dla każdego wektora wejściowego
            epochs: Maksymalna liczba epok uczenia
            learning_rate: Współczynnik uczenia (szybkość uczenia)
        """
        
        self.X_train = np.array(X_train)
        self.y_expected = np.array(y_expected)
        
        for epoch in range(epochs):
            errors = 0
            print(f"\nEpoka {epoch + 1} (BUPA 3D):")
            
            # Rysowanie granicy decyzyjnej na początku epoki
            self.plot_decision_boundary_3d(epoch)
            
            # Inicjalizacja zbiorczej korekty wag
            batch_corrections = np.zeros(len(self.weights))
            
            # Przegląd wszystkich przykładów treningowych
            for i, x in enumerate(X_train):
                x = np.array(x)
                y_pred = self.process_input(x)
                error = y_expected[i] - y_pred
                
                # Zbieranie błędów dla całej partii
                if error != 0:
                    batch_corrections += error * x
                    errors += 1
                
                print(f"  Przykład {i+1}: x={x.tolist()}, y (wartość oczekiwana) = {y_expected[i]}, f(v)={int(y_pred)}, error(d-y)={error}")
                
                # Rysowanie granicy decyzyjnej z aktualnym przykładem (ale bez aktualizacji wag)
                self.plot_decision_boundary_3d(epoch, example=i)
            
            # Aktualizacja wag po przetworzeniu całej partii
            self.weights = self.weights + learning_rate * batch_corrections
            print(f"  Aktualizacja wag po epoce: wagi={self.weights.tolist()}")
            
            # Rysowanie granicy decyzyjnej po aktualizacji wag na koniec epoki
            self.plot_decision_boundary_3d(epoch)
            
            print(f"Liczba błędów w epoce: {errors}")
            if errors == 0:
                print(f"Uczenie zakończone po {epoch + 1} epokach.")
                # Rysowanie finalnej granicy decyzyjnej
                self.plot_decision_boundary_3d(epoch)
                plt.savefig('bupa_final_decision_boundary_3d.png')
                break

    def predict(self, X: list[list[float]]) -> list[float]:
        """Predykcja dla zestawu danych wejściowych"""
        return [self.process_input(x) for x in X]


# Dane treningowe (x1=bias=1, x2, x3)
x_train = [[1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]]

# Operacja logiczna AND
y_train = [0, 0, 0, 1]  # AND: x2 AND x3

# Trening perceptronu metodą BUPA z wizualizacją 3D
perceptron_bupa_3d = BUPAPerceptronVerbose3D()
perceptron_bupa_3d.train(x_train, y_train, epochs=10, learning_rate=1.0)

# Zachowanie wykresu na końcu
plt.ioff()  # Wyłączenie trybu interaktywnego
plt.show()  # Pokazanie ostatecznego wykresu