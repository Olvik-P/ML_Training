import pandas as pd

from constants import (
    DATASET_PATH_TRAIN,
    DATASET_PATH_TEST,
    DISPLAY_MAX_ROWS,
    INDEX_COL,
    PROJECT_ROOT,
)


def configure_pandas() -> None:
    """Настройка глобальных опций pandas."""
    pd.set_option('display.max_rows', DISPLAY_MAX_ROWS)


def load_dataset(
    train_path: str | None = None,
    test_path: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Загрузка train и test датасетов.

    Args:
        train_path: Путь к файлу обучающей выборки.
                    Если None, используется DATASET_PATH_TRAIN.
        test_path: Путь к файлу тестовой выборки.
                   Если None, используется DATASET_PATH_TEST.

    Returns:
        Кортеж (train_df, test_df).

    """
    dataset_train = (
        PROJECT_ROOT / train_path
    ) if train_path is not None else DATASET_PATH_TRAIN

    dataset_test = (
        PROJECT_ROOT / test_path
    ) if test_path is not None else DATASET_PATH_TEST

    print(
        f'PANDAS VERSION: {pd.__version__}'
        f'\nDATASET-PATHS:\n {dataset_train}\n {dataset_test}',
    )

    train_df = pd.read_csv(dataset_train, index_col=INDEX_COL)
    test_df = pd.read_csv(dataset_test, index_col=INDEX_COL)
    return train_df, test_df
