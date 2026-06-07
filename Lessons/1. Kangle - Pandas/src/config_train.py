import pandas as pd

from constants import (
    DATASET_PATH,
    DISPLAY_MAX_ROWS,
    INDEX_COL
)

pd.set_option('display.max_rows', DISPLAY_MAX_ROWS)


def load_dataset() -> pd.DataFrame:
    """Загрузка датасета с винами."""
    print(f'PANDAS VERSION: {pd.__version__}\nDATASET-PATH: {DATASET_PATH}')
    return pd.read_csv(DATASET_PATH, index_col=INDEX_COL)

