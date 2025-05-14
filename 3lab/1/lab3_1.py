import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris, make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

## 1. Загрузка и подготовка данных
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target
df['species'] = df['target'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

print("Первые 5 строк датасета:")
print(df.head())
print("\nНазвания классов:", iris.target_names)

## 2. Визуализация данных (задание 1)
plt.figure(figsize=(12, 5))

# График sepal length vs sepal width
plt.subplot(1, 2, 1)
for species in iris.target_names:
    subset = df[df['species'] == species]
    plt.scatter(subset['sepal length (cm)'], subset['sepal width (cm)'], 
                label=species, alpha=0.7)
plt.title('Sepal Length vs Sepal Width')
plt.xlabel('sepal length (cm)')
plt.ylabel('sepal width (cm)')
plt.legend()

# График petal length vs petal width
plt.subplot(1, 2, 2)
for species in iris.target_names:
    subset = df[df['species'] == species]
    plt.scatter(subset['petal length (cm)'], subset['petal width (cm)'], 
                label=species, alpha=0.7)
plt.title('Petal Length vs Petal Width')
plt.xlabel('petal length (cm)')
plt.ylabel('petal width (cm)')
plt.legend()

plt.tight_layout()
plt.show()

## 3. Pairplot (задание 2)
print("\nPairplot всего датасета:")
sns.pairplot(df, hue='species', palette='viridis')
plt.show()

## 4. Подготовка датасетов (задание 3)
# Датасет 1: setosa и versicolor
df1 = df[df['target'].isin([0, 1])].copy()
X1 = df1[iris.feature_names].values
y1 = df1['target'].values

# Датасет 2: versicolor и virginica
df2 = df[df['target'].isin([1, 2])].copy()
X2 = df2[iris.feature_names].values
y2 = df2['target'].values

## 5-8. Обучение и оценка моделей (задания 4-8)
def train_and_evaluate(X, y, dataset_name):
    # Разделение на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, 
                                                        random_state=42)
    
    # Создание и обучение модели
    clf = LogisticRegression(random_state=0)
    clf.fit(X_train, y_train)
    
    # Предсказание и оценка
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nРезультаты для {dataset_name}:")
    print(f"Точность модели: {accuracy:.4f}")
    print(f"Коэффициенты: {clf.coef_}")
    print(f"Интерсепт: {clf.intercept_}")
    
    return clf

# Обучение для первого датасета (setosa vs versicolor)
model1 = train_and_evaluate(X1, y1, "setosa vs versicolor")

# Обучение для второго датасета (versicolor vs virginica)
model2 = train_and_evaluate(X2, y2, "versicolor vs virginica")

## 9. Генерация и классификация синтетического датасета (задание 9)
# Генерация данных
X_synth, y_synth = make_classification(n_samples=1000, n_features=2, n_redundant=0,
                                     n_informative=2, random_state=1,
                                     n_clusters_per_class=1)

# Визуализация
plt.figure(figsize=(8, 6))
plt.scatter(X_synth[:, 0], X_synth[:, 1], c=y_synth, cmap='viridis', alpha=0.6)
plt.title('Сгенерированный датасет для бинарной классификации')
plt.xlabel('Признак 1')
plt.ylabel('Признак 2')
plt.colorbar()
plt.show()

# Обучение модели
X_train, X_test, y_train, y_test = train_test_split(X_synth, y_synth, 
                                                    test_size=0.3, random_state=42)
clf_synth = LogisticRegression(random_state=0)
clf_synth.fit(X_train, y_train)
y_pred = clf_synth.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\nРезультаты для синтетического датасета:")
print(f"Точность модели: {accuracy:.4f}")
print(f"Коэффициенты: {clf_synth.coef_}")
print(f"Интерсепт: {clf_synth.intercept_}")

# Визуализация разделяющей границы
def plot_decision_boundary(X, y, model):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.4, cmap='viridis')
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', alpha=0.8)
    plt.title('Разделяющая граница логистической регрессии')
    plt.xlabel('Признак 1')
    plt.ylabel('Признак 2')
    plt.show()

plot_decision_boundary(X_synth, y_synth, clf_synth)
