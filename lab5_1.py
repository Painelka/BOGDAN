import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, confusion_matrix, precision_recall_curve,
                           roc_curve, auc, classification_report)
import graphviz
from IPython.display import Image

# Загрузка данных
def load_data():
    try:
        df = pd.read_csv('diabetes.csv')
        print("Данные успешно загружены")
        return df
    except FileNotFoundError:
        print("Ошибка: Файл diabetes.csv не найден")
        return None

# Предобработка данных
def preprocess_data(df):
    # Проверка на пропущенные значения
    if df.isnull().any().any():
        print("Обнаружены пропущенные значения")
        df = df.dropna()
    
    # Разделение на признаки и целевую переменную
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    
    return X, y

# Часть 1: Сравнение моделей
def compare_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Логистическая регрессия
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    
    # Решающее дерево
    tree = DecisionTreeClassifier(random_state=42)
    tree.fit(X_train, y_train)
    y_pred_tree = tree.predict(X_test)
    
    # Вывод метрик
    print("\n=== Логистическая регрессия ===")
    print(classification_report(y_test, y_pred_lr, target_names=['Здоров', 'Диабет']))
    
    print("\n=== Решающее дерево ===")
    print(classification_report(y_test, y_pred_tree, target_names=['Здоров', 'Диабет']))
    
    # Матрицы ошибок
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    sns.heatmap(confusion_matrix(y_test, y_pred_lr), annot=True, fmt='d', cmap='Blues',
               xticklabels=['Здоров', 'Диабет'], yticklabels=['Здоров', 'Диабет'])
    plt.title('Матрица ошибок\n(Логистическая регрессия)')
    
    plt.subplot(1, 2, 2)
    sns.heatmap(confusion_matrix(y_test, y_pred_tree), annot=True, fmt='d', cmap='Blues',
               xticklabels=['Здоров', 'Диабет'], yticklabels=['Здоров', 'Диабет'])
    plt.title('Матрица ошибок\n(Решающее дерево)')
    
    plt.tight_layout()
    plt.show()

# Часть 2: Оптимизация глубины дерева
def optimize_tree_depth(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    depths = range(1, 21)
    f1_scores = []
    
    for depth in depths:
        tree = DecisionTreeClassifier(max_depth=depth, random_state=42)
        tree.fit(X_train, y_train)
        y_pred = tree.predict(X_test)
        f1_scores.append(f1_score(y_test, y_pred))
    
    # Построение графика
    plt.figure(figsize=(10, 6))
    plt.plot(depths, f1_scores, marker='o')
    plt.xlabel('Глубина дерева')
    plt.ylabel('F1-мера')
    plt.title('Зависимость F1-меры от глубины дерева')
    plt.grid()
    plt.show()
    
    # Нахождение оптимальной глубины
    optimal_depth = depths[np.argmax(f1_scores)]
    print(f"\nОптимальная глубина дерева: {optimal_depth} (F1 = {max(f1_scores):.3f})")
    
    return optimal_depth

# Часть 3: Визуализация оптимального дерева
# Замените функцию visualize_tree на этот вариант:

def visualize_tree(X, y, max_depth):
    tree = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    tree.fit(X, y)
    
    # Важность признаков
    plt.figure(figsize=(10, 6))
    sns.barplot(x=tree.feature_importances_, y=X.columns, palette='viridis')
    plt.title('Важность признаков')
    plt.xlabel('Важность')
    plt.ylabel('Признак')
    plt.show()
    
    # Текстовое представление дерева (альтернатива графику)
    from sklearn.tree import export_text
    tree_rules = export_text(tree, feature_names=list(X.columns))
    print("Правила дерева решений:\n", tree_rules)
    
    # ROC и PR кривые
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    y_proba = tree.predict_proba(X_test)[:, 1]
    
    # ROC кривая
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('Доля ложноположительных')
    plt.ylabel('Доля истинноположительных')
    plt.title('ROC-кривая')
    plt.legend()
    
    # PR кривая
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    pr_auc = auc(recall, precision)
    
    plt.subplot(1, 2, 2)
    plt.plot(recall, precision, label=f'AUC = {pr_auc:.2f}')
    plt.xlabel('Полнота')
    plt.ylabel('Точность')
    plt.title('PR-кривая')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# Опциональная часть: Исследование других параметров
def explore_parameters(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Исследование max_features
    features_range = range(1, X.shape[1]+1)
    f1_scores = []
    
    for n_features in features_range:
        tree = DecisionTreeClassifier(max_features=n_features, random_state=42)
        tree.fit(X_train, y_train)
        y_pred = tree.predict(X_test)
        f1_scores.append(f1_score(y_test, y_pred))
    
    plt.figure(figsize=(10, 6))
    plt.plot(features_range, f1_scores, marker='o')
    plt.xlabel('max_features')
    plt.ylabel('F1-мера')
    plt.title('Зависимость F1-меры от max_features')
    plt.grid()
    plt.show()

# Основной блок
if __name__ == "__main__":
    # Загрузка и предобработка данных
    df = load_data()
    if df is not None:
        X, y = preprocess_data(df)
        
        print("\n=== Часть 1: Сравнение моделей ===")
        compare_models(X, y)
        
        print("\n=== Часть 2: Оптимизация глубины дерева ===")
        optimal_depth = optimize_tree_depth(X, y)
        
        print("\n=== Часть 3: Визуализация оптимального дерева ===")
        visualize_tree(X, y, optimal_depth)
        
        print("\n=== Опционально: Исследование других параметров ===")
        explore_parameters(X, y)