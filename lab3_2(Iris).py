# Iris Logistic Regression.ipynb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

# Загрузка данных
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)[['petal length (cm)', 'petal width (cm)']]
y = iris.target

# Масштабирование данных
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Обучение модели
model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
model.fit(X_scaled, y)

# Визуализация
plt.figure(figsize=(10, 6))
xx, yy = np.meshgrid(np.linspace(-2, 2, 100), np.linspace(-2, 2, 100))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.Paired)
for i, color in zip(range(3), 'ryb'):
    idx = np.where(y == i)
    plt.scatter(X_scaled[idx, 0], X_scaled[idx, 1], c=color, 
                label=iris.target_names[i], edgecolor='k')

plt.xlabel('Standardized petal length')
plt.ylabel('Standardized petal width')
plt.title('Многоклассовая логистическая регрессия (Iris)')
plt.legend()
plt.show()

# Отчет о классификации
print(classification_report(y, model.predict(X_scaled), target_names=iris.target_names))

# Вывод коэффициентов
print("\nКоэффициенты модели:")
for i, class_name in enumerate(iris.target_names):
    print(f"\n{class_name}:")
    for j, feature in enumerate(['petal length', 'petal width']):
        print(f"{feature}: {model.coef_[i,j]:.4f}")
    print(f"intercept: {model.intercept_[i]:.4f}")