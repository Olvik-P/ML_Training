"""Модуль для работы с типами данных и пропущенными значениями."""

import pandas as pd
from config_train import load_dataset
from constants import STR_UNKNOWN


def dtypes_demo(wine_reviews: pd.DataFrame) -> None:
    """Демонстрация типов данных DataFrame и Series."""
    print('\nТипы данных / Dtypes')

    print('Тип данных столбца price: ', wine_reviews.price.dtypes)
    print('Series типа данных каждого столбца:\n', wine_reviews.dtypes)
    print('Преобразование столбца points в тип данных float:\n',
          wine_reviews.points.astype('float64'))


def missing_values_demo(wine_reviews: pd.DataFrame) -> None:
    """Демонстрация работы с пропущенными значениями."""
    print('\nПропущенные данные')

    print('Вина без данных в столбце country:\n',
          wine_reviews[pd.isnull(wine_reviews.country)])
    print('Замена всех NaN на Unknown:\n',
          wine_reviews[pd.isnull(wine_reviews.country)].fillna(STR_UNKNOWN))
    print(
        'Замена одного не пустого значения:\n',
        wine_reviews.taster_twitter_handle.replace("@kerinokeefe", "@kerino")
    )


def main() -> None:
    """Основная функция программы."""
    wine_reviews = load_dataset()
    print(wine_reviews.head())
    dtypes_demo(wine_reviews)
    missing_values_demo(wine_reviews)


if __name__ == '__main__':
    main()
