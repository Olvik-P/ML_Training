import pandas as pd
from constants import DATASET_PATH

pd.set_option('display.max_rows', 5)
print(pd.__version__)
print(DATASET_PATH)

wine_reviews = pd.read_csv(DATASET_PATH, index_col=0)

# Сводные функции

print(wine_reviews.points.describe())
print(wine_reviews.taster_name.describe())
# среднее значение
print('среднее значение', wine_reviews.points.mean())
# список уникальных значений
print('список уникальных значений \n', wine_reviews.taster_name.unique())
# список уникальных значений и то, как часто они встречаются в наборе данны
print('список уникальных значений и то, как часто они встречаются в наборе данных \n',
      wine_reviews.taster_name.value_counts())

# Отображения (Maps)

# map() — первый и немного более простой
# перецентрировать оценки вин относительно нуля

wine_reviews_points_mean = wine_reviews.points.mean()
print('map \n', wine_reviews.points.map(
    lambda x: x - wine_reviews_points_mean))
print(wine_reviews.head(1))

# apply() - пользовательский метод для каждой строки


def remean_points(row):
    row.points = row.points - wine_reviews_points_mean
    return row


print('apple \n', wine_reviews.apply(remean_points, axis='columns'))

print(wine_reviews['country'].drop_duplicates())
