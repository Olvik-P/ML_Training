from pathlib import Path

DATASET_FILE = 'data/wine-reviews/winemag-data-130k-v2.csv'
STR_PATH = Path(__file__).resolve().parent.parent
DATASET_PATH = STR_PATH / DATASET_FILE
INDEX_COL = 0
DISPLAY_MAX_ROWS = 5

GROUP_COUNTRY = 'country'
GROUP_PROVINCE = 'province'
GROUP_POINTS = 'points'
GROUP_PRICE = 'price'
GROUP_WINERY = 'winery'
GROUP_TITLE = 'title'
GROUP_LEN = 'len'

STR_UNKNOWN = 'Unknown'
