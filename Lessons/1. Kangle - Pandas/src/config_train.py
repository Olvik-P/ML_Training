import pandas as pd

from constants import (
    DATASET_PATH,
    DISPLAY_MAX_ROWS,
    INDEX_COL,
    STR_PATH
)

pd.set_option('display.max_rows', DISPLAY_MAX_ROWS)


def load_dataset(dataset_file_name=None) -> pd.DataFrame:
    """Загрузка любого датасета."""

    dataset_file_name = (
        STR_PATH / dataset_file_name
    ) if dataset_file_name is not None else DATASET_PATH

    print(
        f'PANDAS VERSION: {pd.__version__}\nDATASET-PATH: {dataset_file_name}')
    return pd.read_csv(dataset_file_name, index_col=INDEX_COL)
