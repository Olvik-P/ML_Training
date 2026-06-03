import pandas as pd
from constants import DATASET_PATH

pd.set_option('display.max_rows', 5)
print(pd.__version__)

wine_reviews = pd.read_csv(DATASET_PATH, index_col=0)

print('Выборка на основе меток:')
print(wine_reviews.set_index('title'))

# Условная выборка
print('вина из Италии, качество которых выше среднего:')
print(wine_reviews.loc[
    (wine_reviews.country == 'Italy') & (wine_reviews.points >= 90)
])

print('произведено в Италии или имеет рейтинг выше среднего')
print(wine_reviews.loc[
    (wine_reviews.country == 'Italy') | (wine_reviews.points >= 90)
])

#  условными селекторами

print(' для выбора вин только из Италии или Франции:')
print(wine_reviews.loc[wine_reviews.country.isin(['Italy', 'France'])])

print(' у которых нет цены в наборе данных:')
print(wine_reviews.loc[wine_reviews.country.notnull()])

# Присваивание данных

print(' присваивание данных:')
wine_reviews['critic'] = 'critic'
print(wine_reviews['critic'])
