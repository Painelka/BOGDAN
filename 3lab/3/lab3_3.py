import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, confusion_matrix, precision_recall_curve,
                            roc_curve, auc, RocCurveDisplay, PrecisionRecallDisplay)

# Загрузка данных
def load_data():
    try:
        df = pd.read_csv('Titanic.csv')
    except FileNotFoundError:
        url = "https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv"
        df = pd.read_csv(url)
    return df

# Предобработка данных
def preprocess_data(df):
    df = df.drop(['Name', 'Ticket', 'Cabin', 'PassengerId'], axis=1)
    df = df.dropna()
    df['Sex'] = LabelEncoder().fit_transform(df['Sex'])
    df['Embarked'] = LabelEncoder().fit_transform(df['Embarked'])
    return df

# Масштабирование данных
def scale_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

# Функция для вычисления метрик
def evaluate_model(model, X, y, model_name):
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]
    
    print(f"\nМетрики для модели {model_name}:")
    print(f"Точность (Accuracy): {accuracy_score(y, y_pred):.4f}")
    print(f"Точность (Precision): {precision_score(y, y_pred):.4f}")
    print(f"Полнота (Recall): {recall_score(y, y_pred):.4f}")
    print(f"F1-мера: {f1_score(y, y_pred):.4f}")
    
    # Матрица ошибок
    cm = confusion_matrix(y, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Не выжил', 'Выжил'],
                yticklabels=['Не выжил', 'Выжил'])
    plt.title(f'Матрица ошибок ({model_name})')
    plt.xlabel('Предсказанный класс')
    plt.ylabel('Истинный класс')
    plt.show()
    
    # Кривая PR
    precision, recall, _ = precision_recall_curve(y, y_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f'{model_name} (AUC = {auc(recall, precision):.2f})')
    plt.xlabel('Полнота (Recall)')
    plt.ylabel('Точность (Precision)')
    plt.title(f'Кривая Precision-Recall ({model_name})')
    plt.legend()
    plt.grid()
    plt.show()
    
    # Кривая ROC
    fpr, tpr, _ = roc_curve(y, y_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('Доля ложноположительных (FPR)')
    plt.ylabel('Доля истинноположительных (TPR)')
    plt.title(f'ROC-кривая ({model_name})')
    plt.legend()
    plt.grid()
    plt.show()

# Часть 1: Анализ модели логистической регрессии
def part1():
    print("=== Часть 1: Анализ модели логистической регрессии ===")
    df = load_data()
    df = preprocess_data(df)
    
    X = df.drop('Survived', axis=1)
    y = df['Survived']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    X_train_scaled, X_test_scaled = scale_data(X_train, X_test)
    
    # Логистическая регрессия
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train_scaled, y_train)
    
    evaluate_model(lr_model, X_test_scaled, y_test, "Логистическая регрессия")

# Часть 2: Сравнение с другими моделями
def part2():
    print("\n=== Часть 2: Сравнение моделей ===")
    df = load_data()
    df = preprocess_data(df)
    
    X = df.drop('Survived', axis=1)
    y = df['Survived']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    X_train_scaled, X_test_scaled = scale_data(X_train, X_test)
    
    # SVM модель
    svm_model = SVC(probability=True, random_state=42)
    svm_model.fit(X_train_scaled, y_train)
    
    evaluate_model(svm_model, X_test_scaled, y_test, "Метод опорных векторов (SVM)")
    
    # KNN модель
    knn_model = KNeighborsClassifier()
    knn_model.fit(X_train_scaled, y_train)
    
    evaluate_model(knn_model, X_test_scaled, y_test, "Метод ближайших соседей (KNN)")

if __name__ == "__main__":
    part1()
    part2()
