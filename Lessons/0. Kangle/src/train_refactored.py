"""Рефакторинг скрипта обучения моделей машинного обучения
для прогнозирования цен на недвижимость.

Основные улучшения:
1. Разделение кода на логические функции
2. Конфигурация через аргументы командной строки
3. Улучшенная обработка данных
4. Модульность и возможность повторного использования
5. Подробная документация на русском языке
"""

import argparse

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor


def load_data(file_path: str) -> pd.DataFrame:
    """Загружает данные из CSV файла.

    Args:
        file_path (str): Путь к CSV файлу с данными

    Returns:
        pd.DataFrame: Загруженные данные

    """
    print(f"Загрузка данных из {file_path}")
    data = pd.read_csv(file_path)
    print(f"Данные загружены. Размер: {data.shape[0]} строк, "
          f"{data.shape[1]} столбцов")
    return data


def preprocess_data(
    data: pd.DataFrame, target_column: str, feature_columns: list
) -> tuple:
    """Предобрабатывает данные: удаляет пропущенные значения,
    разделяет на признаки и целевую переменную.

    Args:
        data (pd.DataFrame): Исходные данные
        target_column (str): Название целевой колонки
        feature_columns (list): Список колонок-признаков

    Returns:
        tuple: (X, y) - признаки и целевая переменная

    """
    print("Предобработка данных...")

    # Удаление строк с пропущенными значениями
    initial_rows = data.shape[0]
    data_clean = data.dropna(axis=0)
    removed_rows = initial_rows - data_clean.shape[0]
    print(f"Удалено строк с пропущенными значениями: {removed_rows}")

    # Проверка наличия колонок
    missing_target = target_column not in data_clean.columns
    missing_features = [
        col for col in feature_columns if col not in data_clean.columns]

    if missing_target:
        raise ValueError(
            f"Целевая колонка '{target_column}' не найдена в данных")
    if missing_features:
        raise ValueError(
            f"Следующие колонки-признаки не найдены: {missing_features}")

    # Создание целевой переменной и признаков
    y = data_clean[target_column]
    X = data_clean[feature_columns]

    print(
        f"После предобработки: {X.shape[0]} образцов, {X.shape[1]} признаков")
    return X, y


def train_decision_tree(
    X_train, y_train, X_val, y_val, max_leaf_nodes: int = 500
) -> tuple:
    """Обучает модель дерева решений и оценивает её.

    Args:
        X_train: Признаки для обучения
        y_train: Целевая переменная для обучения
        X_val: Признаки для валидации
        y_val: Целевая переменная для валидации
        max_leaf_nodes (int): Максимальное количество листьев в дереве

    Returns:
        tuple: (model, mae) - обученная модель и MAE на валидации

    """
    print(
        f"\nОбучение DecisionTreeRegressor с max_leaf_nodes={max_leaf_nodes}")
    model = DecisionTreeRegressor(
        max_leaf_nodes=max_leaf_nodes, random_state=1)
    model.fit(X_train, y_train)

    val_predictions = model.predict(X_val)
    mae = mean_absolute_error(y_val, val_predictions)

    print(f"Валидационная MAE для DecisionTree: {mae:,.0f}")
    return model, mae


def train_random_forest(X_train, y_train, X_val, y_val) -> tuple:
    """Обучает модель случайного леса и оценивает её.

    Args:
        X_train: Признаки для обучения
        y_train: Целевая переменная для обучения
        X_val: Признаки для валидации
        y_val: Целевая переменная для валидации

    Returns:
        tuple: (model, mae) - обученная модель и MAE на валидации

    """
    print("\nОбучение RandomForestRegressor")
    model = RandomForestRegressor(random_state=1)
    model.fit(X_train, y_train)

    val_predictions = model.predict(X_val)
    mae = mean_absolute_error(y_val, val_predictions)

    print(f"Валидационная MAE для RandomForest: {mae:,.0f}")
    return model, mae


def evaluate_leaf_nodes_candidates(
    X_train, y_train, X_val, y_val, candidates: list
) -> dict:
    """Оценивает различные значения max_leaf_nodes для DecisionTreeRegressor.

    Args:
        X_train: Признаки для обучения
        y_train: Целевая переменная для обучения
        X_val: Признаки для валидации
        y_val: Целевая переменная для валидации
        candidates (list): Список кандидатов для max_leaf_nodes

    Returns:
        dict: Словарь с MAE для каждого кандидата

    """
    print("\nОценка различных значений max_leaf_nodes:")
    results = {}

    for nodes in candidates:
        model = DecisionTreeRegressor(max_leaf_nodes=nodes, random_state=0)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        mae = mean_absolute_error(y_val, preds)
        results[nodes] = mae
        print(f"  max_leaf_nodes={nodes}: MAE = {mae:,.0f}")

    best_nodes = min(results, key=results.get)
    print(f"Лучшее значение: max_leaf_nodes={best_nodes} "
          f"с MAE={results[best_nodes]:,.0f}")
    return results


def main():
    """Основная функция скрипта."""
    parser = argparse.ArgumentParser(
        description='Обучение моделей для прогнозирования цен на недвижимость'
    )
    parser.add_argument(
        '--data_path',
        type=str,
        default='../DATA/melb_data.csv',
        help='Путь к CSV файлу с данными (по умолчанию: ../DATA/melb_data.csv)'
    )
    parser.add_argument(
        '--target_column',
        type=str,
        default='Price',
        help='Название целевой колонки (по умолчанию: Price)'
    )
    parser.add_argument(
        '--features',
        type=str,
        nargs='+',
        default=['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude'],
        help='Список колонок-признаков '
             '(по умолчанию: Rooms Bathroom Landsize Lattitude Longtitude)'
    )
    parser.add_argument(
        '--test_size',
        type=float,
        default=0.2,
        help='Доля данных для валидации (по умолчанию: 0.2)'
    )
    parser.add_argument(
        '--max_leaf_nodes',
        type=int,
        default=500,
        help='Максимальное количество листьев для DecisionTree '
             '(по умолчанию: 500)'
    )
    parser.add_argument(
        '--evaluate_nodes',
        action='store_true',
        help='Оценить различные значения max_leaf_nodes'
    )

    args = parser.parse_args()

    print("=" * 60)
    print("РЕФАКТОРИНГ СКРИПТА ОБУЧЕНИЯ МОДЕЛЕЙ МАШИННОГО ОБУЧЕНИЯ")
    print("=" * 60)

    # Загрузка данных
    data = load_data(args.data_path)

    # Вывод общей информации о данных
    print("\nОбщая информация о данных:")
    print(data.describe())

    # Предобработка данных
    X, y = preprocess_data(data, args.target_column, args.features)

    # Разделение данных
    print(f"\nРазделение данных: test_size={args.test_size}")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=args.test_size, random_state=1
    )
    print(f"Обучающая выборка: {X_train.shape[0]} образцов")
    print(f"Валидационная выборка: {X_val.shape[0]} образцов")

    # Оценка различных значений max_leaf_nodes (опционально)
    if args.evaluate_nodes:
        candidates = [50, 100, 250, 500, 750, 1000]
        evaluate_leaf_nodes_candidates(
            X_train, y_train, X_val, y_val, candidates)

    # Обучение Decision Tree
    dt_model, dt_mae = train_decision_tree(
        X_train, y_train, X_val, y_val, args.max_leaf_nodes
    )

    # Обучение Random Forest
    rf_model, rf_mae = train_random_forest(X_train, y_train, X_val, y_val)

    # Сравнение моделей
    print("\n" + "=" * 60)
    print("СРАВНЕНИЕ МОДЕЛЕЙ:")
    print(f"Decision Tree MAE: {dt_mae:,.0f}")
    print(f"Random Forest MAE: {rf_mae:,.0f}")

    if dt_mae < rf_mae:
        print("Decision Tree показал лучший результат")
    elif rf_mae < dt_mae:
        print("Random Forest показал лучший результат")
    else:
        print("Модели показали одинаковый результат")

    print("\nОбучение завершено успешно!")
    return dt_model, rf_model, dt_mae, rf_mae


if __name__ == "__main__":
    main()
