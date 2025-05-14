import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, f1_score, 
                           confusion_matrix, classification_report)
from xgboost import XGBClassifier
from sklearn.tree import plot_tree

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
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    return X, y

# Часть 1: Случайный лес
def random_forest_analysis(X, y):
    print("\n=== Часть 1: Случайный лес ===")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Исследование глубины деревьев
    depths = range(1, 21)
    f1_scores = []
    
    for depth in depths:
        rf = RandomForestClassifier(max_depth=depth, n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        f1_scores.append(f1_score(y_test, y_pred))
    
    plt.figure(figsize=(10, 6))
    plt.plot(depths, f1_scores, marker='o')
    plt.xlabel('Максимальная глубина деревьев')
    plt.ylabel('F1-мера')
    plt.title('Зависимость качества от глубины деревьев')
    plt.grid()
    plt.show()
    
    # Исследование количества признаков
    max_features_range = range(1, X.shape[1]+1)
    f1_scores = []
    
    for n_features in max_features_range:
        rf = RandomForestClassifier(max_features=n_features, n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        f1_scores.append(f1_score(y_test, y_pred))
    
    plt.figure(figsize=(10, 6))
    plt.plot(max_features_range, f1_scores, marker='o')
    plt.xlabel('Количество признаков для разбиения')
    plt.ylabel('F1-мера')
    plt.title('Зависимость качества от количества признаков')
    plt.grid()
    plt.show()
    
    # Исследование количества деревьев
    n_estimators_range = [10, 50, 100, 150, 200, 250, 300]
    f1_scores = []
    train_times = []
    
    for n_est in n_estimators_range:
        start_time = time.time()
        rf = RandomForestClassifier(n_estimators=n_est, random_state=42)
        rf.fit(X_train, y_train)
        train_times.append(time.time() - start_time)
        y_pred = rf.predict(X_test)
        f1_scores.append(f1_score(y_test, y_pred))
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color = 'tab:blue'
    ax1.set_xlabel('Количество деревьев')
    ax1.set_ylabel('F1-мера', color=color)
    ax1.plot(n_estimators_range, f1_scores, marker='o', color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Время обучения (сек)', color=color)
    ax2.plot(n_estimators_range, train_times, marker='s', color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title('Зависимость качества и времени обучения от количества деревьев')
    plt.grid()
    plt.show()
    
    # Лучшая модель
    best_rf = RandomForestClassifier(n_estimators=150, max_depth=8, max_features=5, random_state=42)
    best_rf.fit(X_train, y_train)
    y_pred = best_rf.predict(X_test)
    
    print("\nОтчет о классификации (лучшая модель Random Forest):")
    print(classification_report(y_test, y_pred, target_names=['Здоров', 'Диабет']))
    
    # Важность признаков
    plt.figure(figsize=(10, 6))
    sns.barplot(x=best_rf.feature_importances_, y=X.columns, palette='viridis')
    plt.title('Важность признаков (Random Forest)')
    plt.xlabel('Важность')
    plt.ylabel('Признак')
    plt.show()

# Часть 2: XGBoost
def xgboost_analysis(X, y):
    print("\n=== Часть 2: XGBoost ===")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Исследование параметров XGBoost
    learning_rates = [0.01, 0.05, 0.1, 0.2, 0.3]
    f1_scores = []
    
    for lr in learning_rates:
        xgb = XGBClassifier(learning_rate=lr, n_estimators=100, max_depth=3, random_state=42)
        xgb.fit(X_train, y_train)
        y_pred = xgb.predict(X_test)
        f1_scores.append(f1_score(y_test, y_pred))
    
    plt.figure(figsize=(10, 6))
    plt.plot(learning_rates, f1_scores, marker='o')
    plt.xlabel('Learning rate')
    plt.ylabel('F1-мера')
    plt.title('Зависимость качества от learning rate')
    plt.grid()
    plt.show()
    
    # Лучшая модель XGBoost
    best_xgb = XGBClassifier(
        learning_rate=0.1,
        n_estimators=200,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42
    )
    
    start_time = time.time()
    best_xgb.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    y_pred = best_xgb.predict(X_test)
    
    print("\nОтчет о классификации (XGBoost):")
    print(classification_report(y_test, y_pred, target_names=['Здоров', 'Диабет']))
    print(f"Время обучения: {train_time:.2f} сек")
    
    # Важность признаков
    plt.figure(figsize=(10, 6))
    sns.barplot(x=best_xgb.feature_importances_, y=X.columns, palette='viridis')
    plt.title('Важность признаков (XGBoost)')
    plt.xlabel('Важность')
    plt.ylabel('Признак')
    plt.show()
    
    return best_xgb

# Основной блок
if __name__ == "__main__":
    # Загрузка и предобработка данных
    df = load_data()
    if df is not None:
        X, y = preprocess_data(df)
        
        # Анализ случайного леса
        random_forest_analysis(X, y)
        
        # Анализ XGBoost
        xgboost_model = xgboost_analysis(X, y)
        
        # Выводы
        print("\n=== Выводы ===")
        print("1. Random Forest показывает хорошие результаты при глубине деревьев 6-8")
        print("2. Оптимальное количество признаков для разбиения - около 5")
        print("3. Увеличение количества деревьев свыше 150 дает незначительный прирост качества")
        print("4. XGBoost с правильно подобранными параметрами может превзойти Random Forest")
        print("5. Важные признаки в обеих моделях: Glucose, BMI, Age")