"""Модуль для переименования и объединения DataFrame."""

import pandas as pd

from config_train import load_dataset


def demonstrate_renaming(wine_reviews: pd.DataFrame) -> None:
    """Демонстрация переименования столбцов и индексов."""
    print('\nПереименование')
    print('Переименование столбца points:\n',
          wine_reviews.rename(columns={'points': 'score'}))
    print('Переименование некоторых индексов:\n',
          wine_reviews.rename(index={0: 'first', 1: 'second'}))
    print('Переименование индексов:\n', wine_reviews.rename_axis(
        "wines", axis='rows').rename_axis("fields", axis='columns'))


def load_youtube_datasets() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Загрузка YouTube-датасетов для демонстрации объединения."""
    print('Подготовка дополнительных datasets')
    canadian_youtube = load_dataset('data/youtube-new/CAvideos.csv')
    print('\ncanadian_youtube\n', canadian_youtube)
    british_youtube = load_dataset('data/youtube-new/DEvideos.csv')
    print('\nbritish_youtube\n', british_youtube)
    return canadian_youtube, british_youtube


def demonstrate_concatenation(
    canadian_youtube: pd.DataFrame,
    british_youtube: pd.DataFrame,
) -> None:
    """Демонстрация объединения DataFrame через concat."""
    print('\nОбъединение')
    print('Объединенный datasets c помощью concat:\n',
          pd.concat([canadian_youtube, british_youtube]))


def demonstrate_join(
    canadian_youtube: pd.DataFrame,
    british_youtube: pd.DataFrame,
) -> None:
    """Демонстрация объединения DataFrame через join."""
    left = canadian_youtube.set_index(['title', 'trending_date'])
    right = british_youtube.set_index(['title', 'trending_date'])

    print('Объединенный datasets c помощью join:\n',
          left.join(right, lsuffix='_CAN', rsuffix='_UK'))


def main() -> None:
    """Основная функция программы."""
    wine_reviews = load_dataset()
    print(wine_reviews.head())

    demonstrate_renaming(wine_reviews)

    canadian_youtube, british_youtube = load_youtube_datasets()
    demonstrate_concatenation(canadian_youtube, british_youtube)
    demonstrate_join(canadian_youtube, british_youtube)


if __name__ == '__main__':
    main()
