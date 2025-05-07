import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# 1. Загрузка данных
diabetes = datasets.load_diabetes()
X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
y = diabetes.target

# 2. Выбор признака с наибольшей корреляцией
correlations = X.corrwith(pd.Series(y)).abs()
selected_feature = correlations.idxmax()
X = X[[selected_feature]].values

# 3. Разделение данных на train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Обучение модели (Scikit-Learn)
sklearn_model = LinearRegression()
sklearn_model.fit(X_train, y_train)
y_pred_sklearn = sklearn_model.predict(X_test)

# 5. Собственная реализация линейной регрессии
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

# 6. Метрики качества
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

metrics = {
    'R²': r2_score(y_test, y_pred_sklearn),
    'MAE': mean_absolute_error(y_test, y_pred_sklearn),
    'MAPE': mean_absolute_percentage_error(y_test, y_pred_sklearn)
}

print("\nМетрики для Scikit-Learn модели:")
for name, value in metrics.items():
    print(f"{name}: {value:.3f}" + ("%" if name == "MAPE" else ""))

# 7. Визуализация
plt.figure(figsize=(12, 5))
plt.scatter(X_test, y_test, color='black', label='Реальные значения')
plt.plot(X_test, y_pred_sklearn, color='blue', linewidth=2, label='Scikit-Learn')
plt.plot(X_test, y_pred_custom, 'r--', label='Собственная реализация')
plt.xlabel(selected_feature)
plt.ylabel('Target')
plt.title('Сравнение моделей линейной регрессии')
plt.legend()
plt.show()