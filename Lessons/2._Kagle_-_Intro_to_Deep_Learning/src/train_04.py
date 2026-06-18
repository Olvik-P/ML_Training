"""Модуль обучения нейронной сети для предсказания расхода топлива (FE).

Демонстрирует работу с глубоким обучением на табличных данных.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tensorflow import keras
from tensorflow.keras import layers

plt.style.use('seaborn-v0_8-whitegrid')
plt.rc('figure', autolayout=True)
plt.rc('axes', labelweight='bold', labelsize='large',
       titleweight='bold', titlesize=18, titlepad=10)
plt.rc('animation', html='html5')

print('Стиль графиков установлен.')


def load_data(  # noqa: N802
    path: str = '../data/fuel.csv',
) -> tuple[pd.DataFrame, pd.Series]:
    """Загружает данные из CSV-файла.

    Parameters
    ----------
    path : str, optional
        Путь к файлу с данными, по умолчанию '../data/fuel.csv'.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        Признаки (X) и целевая переменная (y).

    """
    fuel = pd.read_csv(path)
    X = fuel.copy()  # noqa: N806
    y = X.pop('FE')
    return X, y


def preprocess_data(  # noqa: N802
    X: pd.DataFrame, y: pd.Series,  # noqa: N803
) -> tuple[np.ndarray, np.ndarray]:
    """Выполняет предобработку данных.

    Масштабирует числовые признаки, применяет one-hot кодирование
    к категориальным, логарифмирует целевую переменную.

    Parameters
    ----------
    X : pd.DataFrame
        Матрица признаков.
    y : pd.Series
        Целевая переменная.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Обработанные признаки и целевая переменная.

    """
    preprocessor = make_column_transformer(
        (StandardScaler(),
         make_column_selector(dtype_include=np.number)),
        (OneHotEncoder(sparse_output=False),
         make_column_selector(dtype_include=object)),
    )

    X_processed = preprocessor.fit_transform(X)  # noqa: N806
    y_processed = np.log(y)  # log transform target instead of standardizing

    return X_processed, y_processed


def build_model(input_shape: list[int]) -> keras.Sequential:
    """Создаёт и компилирует Sequential-модель.

    Parameters
    ----------
    input_shape : list[int]
        Форма входных данных (количество признаков).

    Returns
    -------
    keras.Sequential
        Скомпилированная модель.

    """
    model = keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=input_shape),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(1),
    ])

    model.compile(
        optimizer='adam',
        loss='mae',
    )

    return model


def train_model(  # noqa: N802
    model: keras.Sequential,
    X: np.ndarray,  # noqa: N803
    y: np.ndarray,
    batch_size: int = 128,
    epochs: int = 200,
) -> keras.callbacks.History:
    """Обучает модель на переданных данных.

    Parameters
    ----------
    model : keras.Sequential
        Модель для обучения.
    X : np.ndarray
        Матрица признаков.
    y : np.ndarray
        Целевая переменная.
    batch_size : int, optional
        Размер батча, по умолчанию 128.
    epochs : int, optional
        Количество эпох, по умолчанию 200.

    Returns
    -------
    keras.callbacks.History
        История обучения.

    """
    return model.fit(
        X, y,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
    )


def plot_history(
    history: keras.callbacks.History, start_epoch: int = 5,
) -> None:
    """Строит и отображает график функции потерь в процессе обучения.

    Parameters
    ----------
    history : keras.callbacks.History
        История обучения модели.
    start_epoch : int, optional
        Эпоха, с которой начинать отображение графика, по умолчанию 5.

    """
    history_df = pd.DataFrame(history.history)
    history_df.loc[start_epoch:, ['loss']].plot()
    plt.xlabel('Эпоха')
    plt.ylabel('MAE (средняя абсолютная ошибка)')
    plt.title('График обучения модели')
    plt.show()


def main() -> None:
    """Основная функция: загрузка, предобработка, обучение и визуализация."""
    # Загрузка данных
    X, y = load_data()  # noqa: N806

    # Предобработка
    X_processed, y_processed = preprocess_data(X, y)  # noqa: N806

    input_shape = [X_processed.shape[1]]
    print(f"Input shape: {input_shape}")

    # Просмотр исходных данных
    print("Первые строки исходных данных:")
    print(X.head())

    # Просмотр обработанных признаков
    print("\nПервые 10 строк обработанных признаков:")
    print(pd.DataFrame(X_processed[:10, :]).head())

    # Создание и обучение модели
    model = build_model(input_shape)
    history = train_model(model, X_processed, y_processed)

    # Визуализация
    plot_history(history)


if __name__ == "__main__":
    main()
