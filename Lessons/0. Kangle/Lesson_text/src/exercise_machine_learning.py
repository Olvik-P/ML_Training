"""Модуль упражнения по машинному обучению.

Содержит функции для обучения модели RandomForest,
поиска оптимального размера дерева и формирования
файла submission для Kaggle-соревнования.
"""

import pandas as pd
from config_train import configure_pandas, load_dataset
from constants import (
    CANDIDATE_MAX_LEAF_NODES,
    FEATURES,
    RANDOM_STATE,
)
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split


def get_mae(
    max_leaf_nodes: int,
    train_x: pd.DataFrame,
    val_x: pd.DataFrame,
    train_y: pd.Series,
    val_y: pd.Series,
) -> float:
    """Вычисляет среднюю абсолютную ошибку для RandomForest.

    Args:
        max_leaf_nodes: Максимальное количество листьев в дереве.
        train_x: Признаки обучающей выборки.
        val_x: Признаки валидационной выборки.
        train_y: Целевая переменная обучающей выборки.
        val_y: Целевая переменная валидационной выборки.

    Returns:
        Средняя абсолютная ошибка (MAE).

    """
    model = RandomForestRegressor(
        max_leaf_nodes=max_leaf_nodes,
        random_state=RANDOM_STATE,
    )
    model.fit(train_x, train_y)
    preds_val = model.predict(val_x)
    return mean_absolute_error(val_y, preds_val)


def prepare_data(
    home_data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Разделяет данные на обучающую и валидационную выборки.

    Args:
        home_data: Исходный DataFrame с данными.

    Returns:
        Кортеж (train_x, val_x, train_y, val_y).

    """
    y: pd.Series = home_data.SalePrice
    x: pd.DataFrame = home_data[FEATURES]
    return train_test_split(x, y, random_state=RANDOM_STATE)


def find_best_tree_size(
    train_x: pd.DataFrame,
    val_x: pd.DataFrame,
    train_y: pd.Series,
    val_y: pd.Series,
) -> tuple[int, float]:
    """Находит оптимальное значение max_leaf_nodes.

    Перебирает кандидатов из CANDIDATE_MAX_LEAF_NODES
    и возвращает лучшее значение с соответствующей MAE.

    Args:
        train_x: Признаки обучающей выборки.
        val_x: Признаки валидационной выборки.
        train_y: Целевая переменная обучающей выборки.
        val_y: Целевая переменная валидационной выборки.

    Returns:
        Кортеж (best_tree_size, best_mae).

    """
    best_mae: float = float('inf')
    best_tree_size: int = CANDIDATE_MAX_LEAF_NODES[0]

    for max_leaf_nodes in CANDIDATE_MAX_LEAF_NODES:
        mae: float = get_mae(
            max_leaf_nodes,
            train_x,
            val_x,
            train_y,
            val_y,
        )
        print(
            f'Максимальное количество листьев: {max_leaf_nodes} \t\t'
            f'Средняя абсолютная ошибка: {mae:.0f}',
        )

        if mae < best_mae:
            best_mae = mae
            best_tree_size = max_leaf_nodes

    print(
        f'\nЛучший max_leaf_nodes: {best_tree_size} с MAE = {best_mae:.0f}',
    )
    return best_tree_size, best_mae


def train_final_model(
    x: pd.DataFrame,
    y: pd.Series,
    best_tree_size: int,
) -> RandomForestRegressor:
    """Обучает финальную модель RandomForest на всех данных.

    Args:
        x: Все признаки.
        y: Все целевые значения.
        best_tree_size: Оптимальный max_leaf_nodes.

    Returns:
        Обученная модель RandomForestRegressor.

    """
    final_model = RandomForestRegressor(
        max_leaf_nodes=best_tree_size,
        random_state=RANDOM_STATE,
    )
    final_model.fit(x, y)
    return final_model


def make_submission(
    model: RandomForestRegressor,
    test: pd.DataFrame,
) -> pd.DataFrame:
    """Формирует DataFrame для submission.

    Args:
        model: Обученная модель.
        test: Тестовый DataFrame.

    Returns:
        DataFrame с колонками Id и SalePrice.

    """
    test_x: pd.DataFrame = test[FEATURES]
    test_preds = model.predict(test_x)

    return pd.DataFrame({
        'Id': test.Id,
        'SalePrice': test_preds,
    })


def main() -> None:
    """Основная функция программы."""
    configure_pandas()
    home_data, test = load_dataset()
    print(f'\nДатасеты для соревнования:\n {home_data.head()}\n {test.head()}')

    train_x, val_x, train_y, val_y = prepare_data(home_data)

    # Базовая модель RandomForest для сравнения
    rf_model = RandomForestRegressor(random_state=RANDOM_STATE)
    rf_model.fit(train_x, train_y)
    rf_val_predictions = rf_model.predict(val_x)
    rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)
    print(f'Validation MAE for Random Forest Model: {rf_val_mae:,.0f}')

    # Поиск оптимального размера дерева
    best_tree_size, _ = find_best_tree_size(
        train_x,
        val_x,
        train_y,
        val_y,
    )

    # Финальная модель и submission
    x: pd.DataFrame = home_data[FEATURES]
    y: pd.Series = home_data.SalePrice
    final_model = train_final_model(x, y, best_tree_size)
    output = make_submission(final_model, test)
    output.to_csv('submission.csv', index=False)


if __name__ == '__main__':
    main()
