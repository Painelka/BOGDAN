import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Загрузка данных
diabetes = datasets.load_diabetes()
df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
df['target'] = diabetes.target

# Исследование данных
print("Первые 5 строк данных:")
print(df.head())
print("\nОписание данных:")
print(df.describe())
print("\nКорреляция с целевой переменной:")
print(df.corr()['target'].sort_values(ascending=False))

# Выбор признака с наибольшей корреляцией (кроме target)
selected_feature = df.corr()['target'].abs().sort_values(ascending=False).index[1]
print(f"\nВыбранный признак для регрессии: {selected_feature}")

# Подготовка данных
X = df[[selected_feature]].values
y = df['target'].values

# Разделение на тренировочную и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 1. Линейная регрессия с использованием Scikit-Learn
sklearn_model = LinearRegression()
sklearn_model.fit(X_train, y_train)
y_pred_sklearn = sklearn_model.predict(X_test)

# 2. Собственная реализация линейной регрессии
def custom_linear_regression(X, y):
    X_mean = np.mean(X)
    y_mean = np.mean(y)
    
    numerator = np.sum((X - X_mean) * (y - y_mean))
    denominator = np.sum((X - X_mean) ** 2)
    
    beta = numerator / denominator
    alpha = y_mean - beta * X_mean
    
    return alpha, beta

alpha, beta = custom_linear_regression(X_train.flatten(), y_train)
y_pred_custom = alpha + beta * X_test.flatten()

# Вывод коэффициентов
print("\nКоэффициенты:")
print(f"Scikit-Learn: intercept = {sklearn_model.intercept_:.2f}, coef = {sklearn_model.coef_[0]:.2f}")
print(f"Собственный алгоритм: alpha = {alpha:.2f}, beta = {beta:.2f}")

# Оценка качества моделей
print("\nОценка качества:")
print(f"Scikit-Learn R²: {r2_score(y_test, y_pred_sklearn):.3f}")
print(f"Собственный R²: {r2_score(y_test, y_pred_custom):.3f}")

# Визуализация
plt.figure(figsize=(15, 5))

# Исходные данные
plt.subplot(1, 2, 1)
plt.scatter(X_train, y_train, color='blue', label='Тренировочные данные')
plt.scatter(X_test, y_test, color='green', label='Тестовые данные')
plt.xlabel(selected_feature)
plt.ylabel('Target')
plt.title('Исходные данные')
plt.legend()

# Регрессионные прямые
plt.subplot(1, 2, 2)
plt.scatter(X_test, y_test, color='green', label='Тестовые данные')
plt.plot(X_test, y_pred_sklearn, color='red', linewidth=2, 
         label=f'Scikit-Learn: y = {sklearn_model.coef_[0]:.2f}x + {sklearn_model.intercept_:.2f}')
plt.plot(X_test, y_pred_custom, color='purple', linestyle='--', linewidth=2,
         label=f'Custom: y = {beta:.2f}x + {alpha:.2f}')
plt.xlabel(selected_feature)
plt.ylabel('Target')
plt.title('Регрессионные прямые')
plt.legend()

plt.tight_layout()
plt.show()

# Таблица с результатами предсказаний
results = pd.DataFrame({
    'Actual': y_test,
    'Scikit-Learn Predicted': y_pred_sklearn,
    'Custom Predicted': y_pred_custom,
    'Scikit-Learn Error': y_test - y_pred_sklearn,
    'Custom Error': y_test - y_pred_custom
})

print("\nТаблица с результатами предсказаний (первые 10 строк):")
print(results.head(10))