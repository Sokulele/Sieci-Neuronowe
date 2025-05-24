import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# Funkcja aktywacji - funkcja skoku jednostkowego
def activation_function1(x: float) -> float:
    return 1.0 if x > 0 else 0.0

# Radialna funkcja bazowa (RBF) - funkcja Gaussa
def rbf(x, center, sigma=1.0):
    """
    Radialna funkcja bazowa (Gaussa)
    Args:
        x: punkt wejściowy
        center: punkt centralny funkcji RBF
        sigma: parametr szerokości (dyspersji)
    Returns:
        Wartość funkcji RBF dla punktu x
    """
    return np.exp(-np.linalg.norm(x - center)**2 / (2 * sigma**2))

# Transformacja danych przy użyciu RBF
def transform_with_rbf(X, centers, sigma=1.0):
    """
    Transformuje dane wejściowe używając radialnych funkcji bazowych
    Args:
        X: dane wejściowe (lista wektorów)
        centers: centra RBF
        sigma: parametr szerokości RBF
    Returns:
        Przekształcone dane w wyższym wymiarze
    """
    X_transformed = []
    for x in X:
        # Usuwamy bias, ponieważ będziemy go dodawać ponownie po transformacji
        x_without_bias = np.array(x[1:])
        # Obliczamy wartości RBF dla każdego centrum
        rbf_values = [rbf(x_without_bias, center, sigma) for center in centers]
        # Dodajemy bias (1.0) z powrotem jako pierwszy element
        transformed = [1.0] + rbf_values
        X_transformed.append(transformed)
    
    return np.array(X_transformed)

# Perceptron z algorytmem BUPA (Batch Updating Perceptron Algorithm)
class PerceptronBUPA:
    def __init__(self):
        self.activation_function = activation_function1
        self.weights = None
        self.fig = None
        self.ax = None
        self.ax_3d = None
        
    def setup_plot(self, is_3d=False):
        """Inicjalizacja wykresu"""
        if is_3d:
            self.fig = plt.figure(figsize=(12, 10))
            self.ax_3d = self.fig.add_subplot(111, projection='3d')
            self.ax_3d.set_xlabel('x2')
            self.ax_3d.set_ylabel('x3')
            self.ax_3d.set_zlabel('RBF')
        else:
            self.fig, self.ax = plt.subplots(figsize=(8, 6))
            self.ax.set_xlim([-0.5, 1.5])
            self.ax.set_ylim([-0.5, 1.5])
            self.ax.set_xlabel('x2')
            self.ax.set_ylabel('x3')
            self.ax.grid(True)       
    
    def plot_decision_surface(self, X_original, centers, sigma, epoch=None):
        """Rysowanie powierzchni decyzyjnej w oryginalnej przestrzeni"""
        # Generujemy siatkę punktów
        x2_range = np.linspace(-0.5, 1.5, 100)
        x3_range = np.linspace(-0.5, 1.5, 100)
        x2_grid, x3_grid = np.meshgrid(x2_range, x3_range)
        
        # Przygotowujemy punkty do predykcji
        grid_points = []
        for i in range(len(x2_range)):
            for j in range(len(x3_range)):
                grid_points.append([1.0, x2_grid[j, i], x3_grid[j, i]])
        
        # Transformujemy punkty siatki przy użyciu RBF
        grid_points_array = np.array(grid_points)
        grid_transformed = transform_with_rbf(grid_points_array, centers, sigma)
        
        # Dokonujemy predykcji
        predictions = [self.process_input(x) for x in grid_transformed]
        
        # Przekształcamy wyniki do siatki
        decision_surface = np.array(predictions).reshape(len(x3_range), len(x2_range))
        
        # Rysujemy powierzchnię decyzyjną
        if self.ax is None:
            self.setup_plot()
        else:
            self.ax.clear()
            self.ax.set_xlim([-0.5, 1.5])
            self.ax.set_ylim([-0.5, 1.5])
            self.ax.set_xlabel('x2')
            self.ax.set_ylabel('x3')
            self.ax.grid(True)
        
        contour = self.ax.contourf(x2_grid, x3_grid, decision_surface, levels=1, 
                                 colors=['lightblue', 'lightsalmon'], alpha=0.5)
        
        # Rysujemy punkty danych treningowych
        colors = ['blue', 'red']
        markers = ['o', 'x']
        
        for i, x in enumerate(X_original):
            color_idx = int(self.y_expected[i])
            self.ax.scatter(x[1], x[2], color=colors[color_idx], marker=markers[color_idx], 
                          s=100, label=f'Klasa {int(self.y_expected[i])}')
        
        # Dodanie legendy bez duplikatów
        handles, labels = self.ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        self.ax.legend(by_label.values(), by_label.keys(), loc='best')
        
        # Tytuł wykresu
        title = 'Powierzchnia decyzyjna XOR po transformacji RBF'
        if epoch is not None:
            title += f' (Epoka {epoch+1})'
        title += f'\nWagi: {", ".join([f"w{i}={w:.2f}" for i, w in enumerate(self.weights)])}'
        self.ax.set_title(title)
        
        # Wyświetlenie wykresu
        plt.tight_layout()
        plt.draw()
        plt.pause(0.5)
    
    def process_input(self, x):
        """Oblicza wartość wyjściową perceptronu dla podanego wektora wejściowego"""
        return self.activation_function(np.dot(x, self.weights))

    def train_bupa(self, X_train, y_expected, epochs, learning_rate):
        """
        Implementacja algorytmu BUPA (Batch Updating Perceptron Algorithm)
        
        Args:
            X_train: Lista wektorów wejściowych (po transformacji RBF)
            y_expected: Lista oczekiwanych wyjść
            epochs: Maksymalna liczba epok uczenia
            learning_rate: Współczynnik uczenia
        """
        X_train = np.array(X_train)
        y_expected = np.array(y_expected)
        
        # Inicjalizacja wag (losowe wartości)
        self.weights = np.random.randn(X_train.shape[1])
        self.y_expected = y_expected
        
        print("Początkowe wagi:", self.weights)
        
        for epoch in range(epochs):
            print(f"\nEpoka {epoch + 1}:")
            
            # Obliczamy przewidywania dla wszystkich przykładów
            y_pred = np.array([self.process_input(x) for x in X_train])
            
            # Obliczamy błędy
            errors = y_expected - y_pred
            errors_count = np.sum(errors != 0)
            
            # W algorytmie BUPA aktualizujemy wagi raz na epokę, używając sumy korekt
            total_correction = np.zeros_like(self.weights)
            for i, (x, error) in enumerate(zip(X_train, errors)):
                correction = learning_rate * error * x
                total_correction += correction
                print(f"  Przykład {i+1}: x={x.tolist()}, y (oczekiwana) = {y_expected[i]}, "
                      f"f(v)={int(y_pred[i])}, error={error}")
            
            # Aktualizacja wag po przejrzeniu wszystkich przykładów
            self.weights = self.weights + total_correction
            print(f"  Nowe wagi po epoce: {self.weights.tolist()}")
            print(f"  Liczba błędów: {errors_count}")
            
            # Wizualizacja aktualnej powierzchni decyzyjnej
            self.plot_decision_surface(self.X_original, self.centers, self.sigma, epoch)
            
            # Sprawdzenie warunku zatrzymania
            if errors_count == 0:
                print(f"Uczenie zakończone po {epoch + 1} epokach.")
                break
                
    def solve_xor_with_rbf(self, X_train, y_expected, centers, sigma, epochs, learning_rate):
        """
        Rozwiązanie problemu XOR przy użyciu RBF i algorytmu BUPA
        Args:
            X_train: Dane wejściowe (oryginalne)
            y_expected: Oczekiwane wyjścia
            centers: Centra funkcji RBF
            sigma: Parametr szerokości RBF
            epochs: Liczba epok
            learning_rate: Współczynnik uczenia
        """
        # Zapisujemy oryginalne dane do wizualizacji
        self.X_original = np.array(X_train)
        self.centers = centers
        self.sigma = sigma
        
        # Transformacja danych przy użyciu RBF
        X_transformed = transform_with_rbf(X_train, centers, sigma)
        
        print("Dane po transformacji RBF:")
        for i, (x_orig, x_trans) in enumerate(zip(X_train, X_transformed)):
            print(f"Oryginalne: {x_orig}, Po transformacji: {x_trans}")
                
        # Trenowanie perceptronu na danych po transformacji
        self.train_bupa(X_transformed, y_expected, epochs, learning_rate)
        
        # Rysowanie ostatecznej powierzchni decyzyjnej
        self.plot_decision_surface(X_train, centers, sigma)
        plt.savefig('xor_rbf_decision_surface.png')
        
    def predict(self, X, centers, sigma):
        """Predykcja dla nowych danych, po ich transformacji RBF"""
        X_transformed = transform_with_rbf(X, centers, sigma)
        return [self.process_input(x) for x in X_transformed]

    
    def plot_decision_surface_3d(self, X_original, centers, sigma):
        """Rysowanie 3D powierzchni decyzyjnej perceptronu po RBF"""
        from mpl_toolkits.mplot3d import Axes3D

        # Generujemy siatkę punktów
        x2_range = np.linspace(-0.2, 1.2, 100)
        x3_range = np.linspace(-0.2, 1.2, 100)
        x2_grid, x3_grid = np.meshgrid(x2_range, x3_range)

        z_grid = np.zeros_like(x2_grid)

        # Obliczamy wartość net input (bez aktywacji)
        for i in range(x2_grid.shape[0]):
            for j in range(x2_grid.shape[1]):
                point = np.array([x2_grid[i, j], x3_grid[i, j]])
                phi = [1.0] + [rbf(point, c, sigma) for c in centers]  # φ(x)
                net_input = np.dot(self.weights, phi)
                z_grid[i, j] = net_input

        # Tworzymy wykres 3D
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(x2_grid, x3_grid, z_grid, cmap='coolwarm', alpha=0.8, edgecolor='none')
        ax.contour(x2_grid, x3_grid, z_grid, levels=[0], colors='black', offset=0)

        # Dodajemy dane treningowe
        for i, point in enumerate(X_original):
            x, y = point[1], point[2]
            z = 1 if self.y_expected[i] else -1
            ax.scatter(x, y, z, color='red' if self.y_expected[i] else 'blue', s=100)

        ax.set_xlabel('x2')
        ax.set_ylabel('x3')
        ax.set_zlabel('net input (w^T * φ(x))')
        ax.set_title('3D powierzchnia decyzyjna perceptronu (XOR po RBF)')
        plt.tight_layout()
        plt.show()
    

# Dane treningowe (x1=bias=1, x2, x3)
x_train = [[1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]]

# Operacja logiczna XOR
y_train = [0, 1, 1, 0]  # XOR: x2 XOR x3

# Centra RBF - wybieramy punkty danych jako centra
# Dla XOR dobrze działają centra umieszczone w punktach treningowych bez biasu
centers = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

# Inicjalizacja i trening perceptronu
perceptron_bupa = PerceptronBUPA()
perceptron_bupa.solve_xor_with_rbf(
    X_train=x_train, 
    y_expected=y_train, 
    centers=centers, 
    sigma=0.5,  # Parametr szerokości RBF
    epochs=100, 
    learning_rate=0.1
)

perceptron_bupa.plot_decision_surface_3d(x_train, centers, 0.5)

# Wyświetlanie ostatecznego wykresu
plt.ioff()  # Wyłączenie trybu interaktywnego
plt.show()

