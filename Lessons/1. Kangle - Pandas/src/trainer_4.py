"""Модуль для группировки и сортировки данных о винах."""

import pandas as pd

from constants import (
    GROUP_COUNTRY,
    GROUP_PROVINCE,
    GROUP_POINTS,
    GROUP_PRICE,
    GROUP_WINERY,
    GROUP_TITLE,
    GROUP_LEN,
)
from config_train import load_dataset


def group_analysis(wine_reviews: pd.DataFrame) -> None:
    """Групповой анализ данных."""
    print('\nГруппировка и сортировка / Групповой анализ')

    print(wine_reviews.groupby(GROUP_POINTS)[GROUP_POINTS].count())

    print('Самое дешевое вино в группе:\n',
          wine_reviews.groupby(GROUP_POINTS)[GROUP_PRICE].min())

    print('Первое вино оцененное от винодельни:\n',
          wine_reviews.groupby(GROUP_WINERY).apply(
              lambda df: df[GROUP_TITLE].iloc[0]
          ).head())

    print('Лучшее вино по стране и в провинции:\n',
          wine_reviews.groupby([GROUP_COUNTRY, GROUP_PROVINCE]).apply(
              lambda df: df.loc[df[GROUP_POINTS].idxmax()]
          ))

    print(
        'Выборка цены вина min и max в стране:\n',
        wine_reviews.groupby(GROUP_COUNTRY)[GROUP_PRICE].agg([len, min, max]),
    )

    print(wine_reviews.loc[wine_reviews[GROUP_COUNTRY] == 'Russia'])


def multiindex_demo(wine_reviews: pd.DataFrame) -> None:
    """Демонстрация работы с мультииндексами."""
    print('\nМультииндексы')

    countries_reviewed = wine_reviews.groupby(
        [GROUP_COUNTRY, GROUP_PROVINCE]
    )[GROUP_TITLE].agg([len])

    print(countries_reviewed)
    mi = countries_reviewed.index
    print(type(mi))

    countries_reviewed_flat = countries_reviewed.reset_index()
    print(countries_reviewed_flat)
    print(type(countries_reviewed_flat))

    return countries_reviewed


def sorting_demo(countries_reviewed: pd.DataFrame) -> None:
    """Демонстрация сортировки данных."""
    print('\nСортировка')

    countries_reviewed = countries_reviewed.reset_index()

    countries_reviewed_country = countries_reviewed.sort_values(by=GROUP_LEN)
    print(countries_reviewed_country)

    countries_reviewed_index = countries_reviewed.sort_index()
    print(countries_reviewed_index)

    countries_reviewed_multycol = countries_reviewed.sort_values(
        by=[GROUP_COUNTRY, GROUP_LEN]
    )
    print(countries_reviewed_multycol)


def main() -> None:
    """Основная функция программы."""
    wine_reviews = load_dataset()
    print('DATASET:\n', wine_reviews.head())

    group_analysis(wine_reviews)
    countries_reviewed = multiindex_demo(wine_reviews)
    sorting_demo(countries_reviewed)


if __name__ == '__main__':
    main()
