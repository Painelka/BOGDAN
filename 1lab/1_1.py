import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os

def read_data(filename):
    """Чтение данных из CSV файла с проверкой"""
    try:
        data = pd.read_csv(filename)
        if len(data.columns) < 2:
            raise ValueError("Файл должен содержать как минимум 2 столбца")
        return data
    except Exception as e:
        print(f"\nОшибка при чтении файла: {e}")
        print(f"Текущая директория: {os.getcwd()}")
        exit()

def show_statistics(data, x_col, y_col):
    """Вывод статистической информации о данных"""
    stats = {
        'Количество': data[[x_col, y_col]].count(),
        'Минимум': data[[x_col, y_col]].min(),
        'Максимум': data[[x_col, y_col]].max(),
        'Среднее': data[[x_col, y_col]].mean()
    }
    stats_df = pd.DataFrame(stats)
    print(f"\nСтатистика по столбцам '{x_col}' и '{y_col}':")
    print(stats_df)

def plot_data(data, x_col, y_col):
    """Визуализация исходных данных"""
    plt.figure(figsize=(15, 5))
    
    # Исходные точки
    plt.subplot(1, 3, 1)
    plt.scatter(data[x_col], data[y_col], color='blue', label='Исходные данные')
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title('1. Исходные данные')
    plt.legend()
    plt.grid(True)

    return plt

def linear_regression(data, x_col, y_col):
    """Реализация линейной регрессии методом наименьших квадратов"""
    x = data[x_col].values
    y = data[y_col].values
    
    n = len(x)
    k = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / (n * np.sum(x**2) - np.sum(x)**2)
    b = (np.sum(y) - k * np.sum(x)) / n
    
    return k, b

def plot_regression_line(plt, data, x_col, y_col, k, b):
    """Добавление регрессионной прямой на график"""
    plt.subplot(1, 3, 2)
    plt.scatter(data[x_col], data[y_col], color='blue', label='Исходные данные')
    
    x_values = np.linspace(data[x_col].min(), data[x_col].max(), 100)
    y_values = k * x_values + b
    plt.plot(x_values, y_values, color='red', label=f'y = {k:.2f}x + {b:.2f}')
    
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title('2. Линейная регрессия')
    plt.legend()
    plt.grid(True)

def plot_error_squares(plt, data, x_col, y_col, k, b):
    """Отрисовка квадратов ошибок"""
    plt.subplot(1, 3, 3)
    plt.scatter(data[x_col], data[y_col], color='blue', label='Исходные данные')
    
    x_values = np.linspace(data[x_col].min(), data[x_col].max(), 100)
    y_values = k * x_values + b
    plt.plot(x_values, y_values, color='red', label=f'y = {k:.2f}x + {b:.2f}')
    
    # Отрисовка квадратов ошибок (исправлена ширина прямоугольников)
    for xi, yi in zip(data[x_col], data[y_col]):
        y_pred = k * xi + b
        error = yi - y_pred
        if error != 0:
            rect = Rectangle((xi, min(yi, y_pred)), 0.05 * (data[x_col].max() - data[x_col].min()), abs(error),
                           linewidth=1, edgecolor='green', facecolor='green', alpha=0.2)
            plt.gca().add_patch(rect)
    
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title('3. Квадраты ошибок')
    plt.legend()
    plt.grid(True)

def main():
    
    # Ввод имени файла
    filename = input("\nВведите имя CSV файла (например: data.csv): ").strip()
    data = read_data(filename)
    
    # Выбор столбцов
    print("\nДоступные столбцы:", list(data.columns))
    x_col = input("Выберите столбец для X: ").strip()
    y_col = input("Выберите столбец для Y: ").strip()
    
    # Проверка выбранных столбцов
    if x_col not in data.columns or y_col not in data.columns:
        print(f"\nОшибка: столбцы '{x_col}' или '{y_col}' не найдены в файле!")
        exit()
    
    # Анализ данных
    show_statistics(data, x_col, y_col)
    k, b = linear_regression(data, x_col, y_col)
    print(f"\nУравнение регрессии: y = {k:.4f}x + {b:.4f}")
    
    # Визуализация
    plt = plot_data(data, x_col, y_col)
    plot_regression_line(plt, data, x_col, y_col, k, b)
    plot_error_squares(plt, data, x_col, y_col, k, b)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
