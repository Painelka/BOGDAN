import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder


def load_data():
    """Загрузка данных из локального файла или интернета"""
    try:
        df = pd.read_csv('Titanic.csv')
        print("Файл Titanic.csv успешно загружен")
        return df
    except FileNotFoundError:
        print("Ошибка: Файл Titanic.csv не найден. Загружаю из интернета...")
        url = "https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv"
        return pd.read_csv(url)


def preprocess_data(df):
    """Предварительная обработка данных"""
    print("\n=== Предобработка данных ===")
    initial_rows = len(df)
    
    # Удаление пропущенных значений
    df_clean = df.dropna().copy()
    
    # Удаление ненужных столбцов
    cols_to_drop = ['Name', 'Ticket', 'Cabin', 'PassengerId']
    df_clean = df_clean.drop(cols_to_drop, axis=1, errors='ignore')
    
    # Кодирование категориальных признаков
    df_clean['Sex'] = LabelEncoder().fit_transform(df_clean['Sex'])
    df_clean['Embarked'] = LabelEncoder().fit_transform(df_clean['Embarked']) + 1
    
    # Расчет потерь данных
    lost_percent = (initial_rows - len(df_clean)) / initial_rows * 100
    print(f"Осталось записей: {len(df_clean)} из {initial_rows} ({lost_percent:.1f}% потеряно)")
    
    return df_clean


def train_and_evaluate(df):
    """Обучение и оценка модели"""
    print("\n=== Машинное обучение ===")
    
    # Подготовка данных
    X = df.drop('Survived', axis=1)
    y = df['Survived']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Обучение модели
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    
    # Оценка точности
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Точность модели: {accuracy:.4f}")
    
    return model, X, y, X_test, y_test  # Возвращаем y вместе с другими данными


def analyze_feature_importance(model, X, y, X_test, y_test):
    """Анализ важности признаков"""
    # Оценка влияния признака Embarked
    X_no_emb = X.drop('Embarked', axis=1)
    X_train_ne, X_test_ne, y_train_ne, y_test_ne = train_test_split(
        X_no_emb, y, test_size=0.3, random_state=42
    )
    
    model_ne = LogisticRegression(max_iter=1000, random_state=42)
    model_ne.fit(X_train_ne, y_train_ne)
    acc_ne = accuracy_score(y_test_ne, model_ne.predict(X_test_ne))
    
    current_acc = accuracy_score(y_test, model.predict(X_test))
    print(f"\nТочность без 'Embarked': {acc_ne:.4f}")
    print(f"Разница в точности: {current_acc - acc_ne:.4f}")
    
    # Визуализация важности признаков
    plt.figure(figsize=(10, 5))
    plt.bar(X.columns, model.coef_[0])
    plt.title('Важность признаков для предсказания выживаемости')
    plt.ylabel('Коэффициент важности')
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()


def main():
    """Основной рабочий поток"""
    # 1. Загрузка и предобработка данных
    df = load_data()
    df_processed = preprocess_data(df)
    print("\nПервые 5 строк обработанных данных:")
    print(df_processed.head())
    
    # 2. Обучение модели
    model, X, y, X_test, y_test = train_and_evaluate(df_processed)
    
    # 3. Анализ результатов
    print("\nКоэффициенты модели:")
    for feature, coef in zip(X.columns, model.coef_[0]):
        print(f"{feature:15}: {coef:>7.4f}")
    
    analyze_feature_importance(model, X, y, X_test, y_test)


if __name__ == "__main__":
    main()
