import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.tree import DecisionTreeRegressor

warnings.filterwarnings('ignore')


def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    """Вычисляет MAE для Decision Tree с заданным количеством листьев"""
    model = DecisionTreeRegressor(
        max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return mae


def evaluate_model(model, X_train, X_val, y_train, y_val, model_name="Model"):
    """Оценка модели с несколькими метриками"""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)

    mae = mean_absolute_error(y_val, y_pred)
    mse = mean_squared_error(y_val, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_val, y_pred)

    print(f"\n{model_name} Результаты:")
    print(f"  MAE:  {mae:,.0f}")
    print(f"  MSE:  {mse:,.0f}")
    print(f"  RMSE: {rmse:,.0f}")
    print(f"  R²:   {r2:.4f}")

    return {
        'model': model,
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'r2': r2,
        'predictions': y_pred
    }


def plot_feature_importance(model, feature_names, title="Важность признаков"):
    """Визуализация важности признаков"""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]

        plt.figure(figsize=(10, 6))
        plt.title(title)
        plt.bar(range(len(importances)), importances[indices])
        plt.xticks(range(len(importances)), [
                   feature_names[i] for i in indices], rotation=45)
        plt.xlabel("Признаки")
        plt.ylabel("Важность")
        plt.tight_layout()
        plt.show()


def main():
    # Загрузка данных
    melb_file_path = '../DATA/melb_data.csv'
    melbourne_data = pd.read_csv(melb_file_path)

    print("=" * 60)
    print("АНАЛИЗ ДАННЫХ MELBOURNE HOUSING")
    print("=" * 60)

    # Базовая информация о данных
    print(f"\nРазмер данных: {melbourne_data.shape}")
    print(f"Колонки: {list(melbourne_data.columns)}")
    print("\nТипы данных:")
    print(melbourne_data.dtypes.value_counts())

    # Пропущенные значения
    missing = melbourne_data.isnull().sum()
    print("\nПропущенные значения:")
    print(missing[missing > 0])

    # Удаление строк с пропущенными значениями
    melbourne_data = melbourne_data.dropna(axis=0)
    print(f"\nДанные после удаления пропусков: {melbourne_data.shape}")

    # Базовые статистики
    print("\nБазовые статистики целевой переменной (Price):")
    print(melbourne_data['Price'].describe())

    # 1. БАЗОВАЯ МОДЕЛЬ (как в оригинале)
    print("\n" + "=" * 60)
    print("1. БАЗОВАЯ МОДЕЛЬ (оригинальные признаки)")
    print("=" * 60)

    # Создайте целевой объект и назовите его y
    y = melbourne_data.Price
    # Создайте X с базовыми признаками
    features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
    X = melbourne_data[features]

    # Разделите на валидационные и обучающие данные
    train_X, val_X, train_y, val_y = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Определите модель Decision Tree
    melb_model = DecisionTreeRegressor(max_leaf_nodes=500, random_state=1)
    # Обучите модель
    melb_model.fit(train_X, train_y)

    # Сделайте валидационные прогнозы и вычислите среднюю абсолютную ошибку
    val_predictions = melb_model.predict(val_X)
    val_mae = mean_absolute_error(val_predictions, val_y)
    print(f"Валидационная MAE для DecisionTree: {val_mae:,.0f}")

    # Random Forest базовая модель
    forest_model = RandomForestRegressor(random_state=1)
    forest_model.fit(train_X, train_y)
    melb_preds = forest_model.predict(val_X)
    print(
        f"Валидационная MAE для RandomForest: {
            mean_absolute_error(val_y, melb_preds):,.0f}"
    )

    # 2. УЛУЧШЕННАЯ МОДЕЛЬ С БОЛЬШИМ КОЛИЧЕСТВОМ ПРИЗНАКОВ
    print("\n" + "=" * 60)
    print("2. УЛУЧШЕННАЯ МОДЕЛЬ С ДОПОЛНИТЕЛЬНЫМИ ПРИЗНАКАМИ")
    print("=" * 60)

    # Выбор более информативных признаков
    improved_features = [
        'Rooms', 'Bathroom', 'Car', 'Landsize', 'BuildingArea',
        'YearBuilt', 'Distance', 'Bedroom2', 'Propertycount',
        'Lattitude', 'Longtitude'
    ]

    # Удаляем строки где BuildingArea или YearBuilt пропущены
    melbourne_data_improved = melbourne_data.dropna(
        subset=['BuildingArea', 'YearBuilt'])
    y_improved = melbourne_data_improved['Price']
    X_improved = melbourne_data_improved[improved_features]

    train_X_imp, val_X_imp, train_y_imp, val_y_imp = train_test_split(
        X_improved, y_improved, test_size=0.2, random_state=42
    )

    # 2.1 Улучшенный Decision Tree с подбором гиперпараметров
    print("\n--- Улучшенный Decision Tree ---")

    # Подбор оптимального количества листьев
    candidate_max_leaf_nodes = [50, 100, 250, 500, 750, 1000, 1500]
    mae_scores = []

    for max_leaf_nodes in candidate_max_leaf_nodes:
        mae = get_mae(max_leaf_nodes, train_X_imp,
                      val_X_imp, train_y_imp, val_y_imp)
        mae_scores.append(mae)
        print(f"max_leaf_nodes: {max_leaf_nodes:4d} \t MAE: {mae:,.0f}")

    best_max_leaf_nodes = candidate_max_leaf_nodes[np.argmin(mae_scores)]
    print(f"\nЛучшее количество листьев: {best_max_leaf_nodes}")

    # Обучение лучшей модели
    best_tree = DecisionTreeRegressor(
        max_leaf_nodes=best_max_leaf_nodes,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42
    )

    tree_results = evaluate_model(
        best_tree, train_X_imp, val_X_imp, train_y_imp, val_y_imp,
        "Улучшенный Decision Tree"
    )

    # 2.2 Улучшенный Random Forest с настройкой гиперпараметров
    print("\n--- Улучшенный Random Forest ---")

    # Базовый Random Forest с настройками
    rf_improved = RandomForestRegressor(
        n_estimators=200,
        max_depth=20,
        min_samples_split=10,
        min_samples_leaf=4,
        max_features='sqrt',
        bootstrap=True,
        random_state=42,
        n_jobs=-1
    )

    rf_results = evaluate_model(
        rf_improved, train_X_imp, val_X_imp, train_y_imp, val_y_imp,
        "Улучшенный Random Forest"
    )

    # 2.3 Gradient Boosting для сравнения
    print("\n--- Gradient Boosting (дополнительная модель) ---")

    gb_model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )

    gb_results = evaluate_model(
        gb_model, train_X_imp, val_X_imp, train_y_imp, val_y_imp,
        "Gradient Boosting"
    )

    # 3. СРАВНЕНИЕ МОДЕЛЕЙ
    print("\n" + "=" * 60)
    print("3. СРАВНЕНИЕ ПРОИЗВОДИТЕЛЬНОСТИ МОДЕЛЕЙ")
    print("=" * 60)

    comparison = pd.DataFrame({
        'Model': [
            'Base Decision Tree', 'Base Random Forest',
            'Improved Decision Tree', 'Improved Random Forest',
            'Gradient Boosting'
        ],
        'MAE': [val_mae, mean_absolute_error(val_y, melb_preds),
                tree_results['mae'], rf_results['mae'], gb_results['mae']],
        'R²': [r2_score(val_y, val_predictions), r2_score(val_y, melb_preds),
               tree_results['r2'], rf_results['r2'], gb_results['r2']]
    })

    print(comparison.to_string(index=False))

    # 4. АНАЛИЗ ВАЖНОСТИ ПРИЗНАКОВ
    print("\n" + "=" * 60)
    print("4. АНАЛИЗ ВАЖНОСТИ ПРИЗНАКОВ (Random Forest)")
    print("=" * 60)

    # Получаем важность признаков
    feature_importance = pd.DataFrame({
        'feature': improved_features,
        'importance': rf_improved.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\nВажность признаков:")
    print(feature_importance.to_string(index=False))

    # 5. КРОСС-ВАЛИДАЦИЯ
    print("\n" + "=" * 60)
    print("5. КРОСС-ВАЛИДАЦИЯ (5-fold)")
    print("=" * 60)

    # Кросс-валидация для Random Forest
    cv_scores = cross_val_score(
        rf_improved, X_improved, y_improved,
        cv=5, scoring='neg_mean_absolute_error'
    )

    print(
        f"MAE кросс-валидация: {
            -cv_scores.mean():,.0f} (+/- {-cv_scores.std() * 2:,.0f})")

    # 6. ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ
    print("\n" + "=" * 60)
    print("6. ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ")
    print("=" * 60)

    # Создаем директорию для графиков если нужно
    import os
    if not os.path.exists('../IMG'):
        os.makedirs('../IMG')

    # График сравнения моделей
    plt.figure(figsize=(10, 6))
    models = comparison['Model']
    mae_values = comparison['MAE']

    bars = plt.bar(models, mae_values, color=[
                   'red', 'orange', 'green', 'blue', 'purple'])
    plt.xlabel('Модели')
    plt.ylabel('MAE (меньше лучше)')
    plt.title('Сравнение MAE разных моделей')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    # Добавляем значения на столбцы
    for bar, value in zip(bars, mae_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5000,
                 f'{value:,.0f}', ha='center', va='bottom', fontsize=9)

    plt.savefig('../IMG/model_comparison.png', dpi=150, bbox_inches='tight')
    print("График сравнения моделей сохранен в ../IMG/model_comparison.png")

    # График важности признаков
    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance['feature'], feature_importance['importance'])
    plt.xlabel('Важность')
    plt.title('Важность признаков в Random Forest')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('../IMG/feature_importance.png', dpi=150, bbox_inches='tight')
    print("График важности признаков сохранен в ../IMG/feature_importance.png")

    # 7. РЕКОМЕНДАЦИИ
    print("\n" + "=" * 60)
    print("7. РЕКОМЕНДАЦИИ ПО УЛУЧШЕНИЮ")
    print("=" * 60)

    print("""
1. ДОБАВЛЕНИЕ ПРИЗНАКОВ:
   - Используйте больше признаков (BuildingArea, YearBuilt, Distance и т.д.)
   - Создайте производные признаки (цена за квадратный метр, возраст здания)
   - Рассмотрите категориальные признаки (Type, Regionname) через one-hot
          encoding

2. НАСТРОЙКА ГИПЕРПАРАМЕТРОВ:
   - Для Decision Tree: подбирайте max_leaf_nodes, min_samples_split,
          min_samples_leaf
   - Для Random Forest: увеличьте n_estimators (200-500), настройте max_depth
   - Используйте GridSearchCV для автоматического подбора

3. ПРЕДОБРАБОТКА ДАННЫХ:
   - Масштабирование числовых признаков
   - Обработка выбросов
   - Заполнение пропущенных значений (медианой/средним)

4. АНСАМБЛИ:
   - Random Forest показал лучшие результаты чем одиночное дерево
   - Gradient Boosting может дать еще лучшее качество
   - Рассмотрите Stacking или Voting ensembles

5. ВАЛИДАЦИЯ:
   - Всегда используйте кросс-валидацию
   - Разделяйте данные на train/validation/test
   - Используйте раннюю остановку для предотвращения переобучения
    """)

    print("\n" + "=" * 60)
    print("ВЫВОДЫ:")
    print("=" * 60)
    print(
        f"1. Random Forest превосходит Decision Tree на {
            ((val_mae - rf_results['mae']) / val_mae * 100):.1f}% по MAE")
    print(
        f"2. Самые важные признаки: {
            feature_importance.iloc[0]['feature']
        }, {feature_importance.iloc[1]['feature']}"
    )
    print(
        f"3. Лучшая модель: Improved Random Forest (MAE: {
            rf_results['mae']:,.0f}, R²: {rf_results['r2']:.4f})"
    )
    print("4. Дальнейшее улучшение возможно через feature engineering")


if __name__ == "__main__":
    main()
