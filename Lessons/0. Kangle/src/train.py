import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor


def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(
        max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return mae


# Путь к файлу для чтения
melb_file_path = '../DATA/melb_data.csv'
melbourne_data = pd.read_csv(melb_file_path)

print(melbourne_data.describe())
melbourne_data = melbourne_data.dropna(axis=0)

# Создайте целевой объект и назовите его y
y = melbourne_data.Price
# Создайте X
features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
X = melbourne_data[features]

# Разделите на валидационные и обучающие данные
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

# Определите модель
melb_model = DecisionTreeRegressor(max_leaf_nodes=500, random_state=1)
# Обучите модель
melb_model.fit(train_X, train_y)

# Сделайте валидационные прогнозы и вычислите среднюю абсолютную ошибку
val_predictions = melb_model.predict(val_X)
val_mae = mean_absolute_error(val_predictions, val_y)
print("Валидационная MAE для DecisionTree: {:,.0f}".format(val_mae))

forest_model = RandomForestRegressor(random_state=1)
forest_model.fit(train_X, train_y)
melb_preds = forest_model.predict(val_X)
print("Валидационная MAE для RandomForest: {:,.0f}".format(
    mean_absolute_error(val_y, melb_preds)))
