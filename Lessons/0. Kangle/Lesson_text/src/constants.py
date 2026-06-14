from pathlib import Path

# Пути к датасетам
DATASET_FILE_TRAIN = 'data/train.csv'
DATASET_FILE_TEST = 'data/test.csv'
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATASET_PATH_TRAIN = PROJECT_ROOT / DATASET_FILE_TRAIN
DATASET_PATH_TEST = PROJECT_ROOT / DATASET_FILE_TEST
# None — не использовать колонку как индекс (стандартный целочисленный индекс)
INDEX_COL = None
DISPLAY_MAX_ROWS = 5

# Константы для группировки данных (вино)
GROUP_COUNTRY = 'country'
GROUP_PROVINCE = 'province'
GROUP_POINTS = 'points'
GROUP_PRICE = 'price'
GROUP_WINERY = 'winery'
GROUP_TITLE = 'title'
GROUP_LEN = 'len'

STR_UNKNOWN = 'Unknown'

# Константы для упражнения по машинному обучению
RANDOM_STATE: int = 1
FEATURES: list[str] = [
    'LotArea',
    'YearBuilt',
    '1stFlrSF',
    '2ndFlrSF',
    'FullBath',
    'BedroomAbvGr',
    'TotRmsAbvGrd',
]
CANDIDATE_MAX_LEAF_NODES: list[int] = [5, 25, 50, 100, 250, 500]
