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
    
    def plot_transformed_data(self, X_original, X_transformed, y_expected):
        """Wizualizacja danych w przestrzeni transformowanej"""
        # Tworzymy wykres 3D
        self.setup_plot(is_3d=True)
        
        # Kolory dla klas
        colors = ['blue', 'red']
        
        # Rysujemy punkty w 3D (oryginalne x2, x3 + pierwsza wartość RBF)
        # for i, (x_orig, x_trans) in enumerate(zip(X_original, X_transformed)):
        #     color_idx = int(y_expected[i])
        #     self.ax_3d.scatter(x_orig[1], x_orig[2], x_trans[1], 
        #                      color=colors[color_idx], marker='o', s=100)
            
        # self.ax_3d.set_title('Dane po transformacji RBF')
        # plt.tight_layout()
        # plt.draw()
        # plt.pause(0.5)
    
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
        
        # Wizualizacja transformowanych danych
        self.plot_transformed_data(X_train, X_transformed, y_expected)
        
        # Trenowanie perceptronu na danych po transformacji
        self.train_bupa(X_transformed, y_expected, epochs, learning_rate)
        
        # Rysowanie ostatecznej powierzchni decyzyjnej
        self.plot_decision_surface(X_train, centers, sigma)
        plt.savefig('xor_rbf_decision_surface.png')
        
    def predict(self, X, centers, sigma):
        """Predykcja dla nowych danych, po ich transformacji RBF"""
        X_transformed = transform_with_rbf(X, centers, sigma)
        return [self.process_input(x) for x in X_transformed]


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

# Wyświetlanie ostatecznego wykresu
plt.ioff()  # Wyłączenie trybu interaktywnego
plt.show()

# # Test działania na danych treningowych
# print("\nTest na danych treningowych:")
# predictions = perceptron_bupa.predict(x_train, centers, sigma=0.5)
# for x, y_true, y_pred in zip(x_train, y_train, predictions):
#     print(f"Wejście: {x[1:]}, Oczekiwane: {y_true}, Predykcja: {int(y_pred)}")

# # Wizualizacja 3D przestrzeni cech po transformacji RBF
# def plot_rbf_space():
#     fig = plt.figure(figsize=(10, 8))
#     ax = fig.add_subplot(111, projection='3d')
    
#     # Generowanie gęstej siatki punktów w przestrzeni wejściowej
#     x2_range = np.linspace(-0.5, 1.5, 30)
#     x3_range = np.linspace(-0.5, 1.5, 30)
#     x2_grid, x3_grid = np.meshgrid(x2_range, x3_range)
    
#     # Dla każdego punktu w siatce, obliczamy wartości RBF dla pierwszego i drugiego centrum
#     grid_rbf1 = np.zeros_like(x2_grid)
#     grid_rbf2 = np.zeros_like(x2_grid)
    
#     for i in range(len(x2_range)):
#         for j in range(len(x3_range)):
#             point = np.array([x2_grid[j, i], x3_grid[j, i]])
#             grid_rbf1[j, i] = rbf(point, centers[0], sigma=0.5)
#             grid_rbf2[j, i] = rbf(point, centers[1], sigma=0.5)
    
#     # Rysowanie powierzchni RBF
#     ax.plot_surface(x2_grid, x3_grid, grid_rbf1, cmap=cm.coolwarm, alpha=0.5, label='RBF1')
#     ax.plot_surface(x2_grid, x3_grid, grid_rbf2, cmap=cm.viridis, alpha=0.5, label='RBF2')
    
#     # Dodanie punktów danych
#     colors = ['blue', 'red']
#     for i, x in enumerate(x_train):
#         color_idx = int(y_train[i])
#         ax.scatter(x[1], x[2], 0, color=colors[color_idx], marker='o', s=100)
    
#     ax.set_xlabel('x2')
#     ax.set_ylabel('x3')
#     ax.set_zlabel('RBF Value')
#     ax.set_title('Przestrzeń cech po transformacji RBF')
    
#     plt.tight_layout()
#     plt.savefig('rbf_feature_space.png')
#     plt.show()

# plot_rbf_space()