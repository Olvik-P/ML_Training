# Анализ кода: `train_01.py` — Transfer Learning с VGG16 для бинарной классификации изображений

## Общее назначение

Скрипт [`train_01.py`](Lessons/4._Computer_Vision/src/train_01.py) реализует **бинарную классификацию изображений** с использованием техники **Transfer Learning** (перенос обучения). В качестве предобученной основы используется архитектура **VGG16**, обученная на датасете ImageNet. Сверточная часть VGG16 замораживается, а поверх неё достраиваются новые полносвязные слои, которые обучаются под конкретную задачу.

---

## 1. Импорты и настройки (строки 1–41)

```python
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing import image_dataset_from_directory
```

**Назначение импортов:**

| Библиотека | Роль |
|---|---|
| `os` | Управление переменными окружения (для воспроизводимости) |
| `warnings` | Подавление предупреждений TensorFlow |
| `matplotlib.pyplot` | Визуализация графиков обучения |
| `numpy` | Численные операции, генерация случайных чисел |
| `pandas` | Обработка истории обучения (DataFrame) |
| `tensorflow` / `keras` | Фреймворк глубокого обучения |
| `VGG16` | Предобученная архитектура из `keras.applications` |
| `image_dataset_from_directory` | Загрузка изображений из структуры папок |

---

## 2. Воспроизводимость результатов (строки 14–19)

```python
def set_seed(seed=31415):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
```

Функция [`set_seed()`](Lessons/4._Computer_Vision/src/train_01.py:15) фиксирует все источники случайности:

- `np.random.seed` — для NumPy.
- `tf.random.set_seed` — для TensorFlow.
- `PYTHONHASHSEED` — фиксирует хеширование строк в Python (влияет на порядок в `set`/`dict`).
- `TF_DETERMINISTIC_OPS` — включает детерминированные операции в TensorFlow (ценой производительности).

> **Зачем:** Обеспечение воспроизводимости экспериментов — критически важно в исследованиях и разработке.

---

## 3. Функция преобразования данных (строки 23–25)

```python
def convert_to_float(image, label):
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    return image, label
```

[`convert_to_float()`](Lessons/4._Computer_Vision/src/train_01.py:23) преобразует пиксели изображения из целочисленного формата (uint8, диапазон 0–255) в float32 (диапазон 0.0–1.0). Это стандартная нормализация, необходимая для стабильной работы нейронной сети.

---

## 4. Загрузка данных (строки 45–68)

### 4.1. Создание датасетов из директорий

```python
ds_train_ = image_dataset_from_directory(
    "./data/train",
    labels="inferred",
    label_mode="binary",
    image_size=[128, 128],
    interpolation="nearest",
    batch_size=64,
    shuffle=True,
)
```

**Параметры:**

| Параметр | Значение | Пояснение |
|---|---|---|
| `"./data/train"` | путь | Обучающая выборка в папке `train/` |
| `labels="inferred"` | авто | Метки извлекаются из имён подпапок |
| `label_mode="binary"` | бинарный | Метки — 0 или 1 (float32) |
| `image_size=[128, 128]` | 128×128 | Все изображения ресайзятся к этому размеру |
| `interpolation="nearest"` | nearest | Метод интерполяции при ресайзе (ближайший сосед) |
| `batch_size=64` | 64 | Размер батча |
| `shuffle=True` | да | Перемешивание только для train |

Для валидационной выборки [`ds_valid_`](Lessons/4._Computer_Vision/src/train_01.py:54) `shuffle=False`, так как порядок не важен.

### 4.2. Оптимизация пайплайна

```python
AUTOTUNE = tf.data.experimental.AUTOTUNE
ds_train = ds_train_.map(convert_to_float).cache().prefetch(buffer_size=AUTOTUNE)
ds_valid = ds_valid_.map(convert_to_float).cache().prefetch(buffer_size=AUTOTUNE)
```

**Цепочка оптимизаций:**

1. **`.map(convert_to_float)`** — нормализация пикселей.
2. **`.cache()`** — кеширование датасета в памяти после первой эпохи. Резко ускоряет последующие эпохи.
3. **`.prefetch(buffer_size=AUTOTUNE)`** — асинхронная предзагрузка следующего батча во время обучения текущего. `AUTOTUNE` позволяет TensorFlow автоматически подобрать оптимальный размер буфера.

> **Важно:** `.cache()` сохраняет данные **после** `.map()`, поэтому нормализация выполняется только один раз.

---

## 5. Загрузка предобученной модели VGG16 (строки 71–77)

```python
pretrained_base = VGG16(
    weights="imagenet",
    include_top=False,
    input_shape=(128, 128, 3),
)
pretrained_base.trainable = False
```

**Что такое VGG16?**

VGG16 — это глубокая сверточная нейросеть, предложенная в 2014 году (Simonyan & Zisserman). Состоит из 16 слоёв с весами: 13 сверточных (convolutional) и 3 полносвязных (fully connected). В данном случае `include_top=False` отсекает полносвязные слои, оставляя только **сверточную часть** — «экстрактор признаков».

**Параметры:**

| Параметр | Значение | Пояснение |
|---|---|---|
| `weights="imagenet"` | ImageNet | Загружаются веса, обученные на 1.2M изображений из 1000 классов |
| `include_top=False` | без головы | Удалены полносвязные слои классификатора |
| `input_shape=(128,128,3)` | 128×128×3 | Входной размер (RGB) |

**`pretrained_base.trainable = False`** — замораживает все веса сверточной части. Они не будут обновляться при обучении. Это ключевой элемент Transfer Learning: мы используем VGG16 как готовый экстрактор признаков.

---

## 6. Построение модели (строки 79–86)

```python
model = keras.Sequential([
    pretrained_base,          # экстрактор признаков (заморожен)
    layers.Flatten(),         # преобразует карты признаков в вектор
    layers.Dense(6, activation="relu"),   # полносвязный слой, 6 нейронов
    layers.Dense(1, activation="sigmoid"), # выходной слой для бинарной классификации
])
```

**Архитектура:**

```
Вход (128×128×3)
    ↓
VGG16 (заморожен) → выход: (4×4×512) карт признаков
    ↓
Flatten → вектор длины 8192
    ↓
Dense(6, ReLU) → 6 нейронов
    ↓
Dense(1, Sigmoid) → вероятность класса 1
```

**Комментарий к слоям:**

- **`Flatten()`** — преобразует многомерный тензор карт признаков (4×4×512 = 8192) в одномерный вектор.
- **`Dense(6, activation="relu")`** — первый обучаемый слой. Всего 6 нейронов — это **очень маленький** слой. Обычно используют 256–512 нейронов. Это потенциальное **бутылочное горлышко** (bottleneck), которое может ограничить выразительность модели.
- **`Dense(1, activation="sigmoid")`** — выходной слой. Сигмоида выдаёт вероятность от 0 до 1. Порог классификации обычно 0.5.

---

## 7. Компиляция модели (строки 88–92)

```python
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["binary_accuracy"],
)
```

| Параметр | Значение | Пояснение |
|---|---|---|
| `optimizer="adam"` | Adam | Адаптивный оптимизатор с автоматической подстройкой learning rate |
| `loss="binary_crossentropy"` | Бинарная кросс-энтропия | Стандартная функция потерь для бинарной классификации |
| `metrics=["binary_accuracy"]` | Доля правильных ответов | Метрика для мониторинга |

---

## 8. Обучение (строки 94–99)

```python
history = model.fit(
    ds_train,
    validation_data=ds_valid,
    epochs=30,
    verbose=0,
)
```

- **`epochs=30`** — 30 полных проходов по обучающей выборке.
- **`verbose=0`** — без вывода прогресса в консоль (чистый вывод).
- **`validation_data=ds_valid`** — валидация после каждой эпохи.

---

## 9. Визуализация результатов (строки 101–103)

```python
history_frame = pd.DataFrame(history.history)
history_frame.loc[:, ["loss", "val_loss"]].plot()
history_frame.loc[:, ["binary_accuracy", "val_binary_accuracy"]].plot()
```

[`history.history`](Lessons/4._Computer_Vision/src/train_01.py:101) — словарь с историей метрик по эпохам. Преобразуется в DataFrame для удобного построения графиков:

1. **График потерь** (`loss` и `val_loss`) — позволяет диагностировать переобучение (если `val_loss` растёт, а `loss` падает).
2. **График точности** (`binary_accuracy` и `val_binary_accuracy`) — показывает, как растёт качество модели.

---

## Ключевые концепции

### Transfer Learning (перенос обучения)

Использование модели, обученной на большой задаче (ImageNet, 1000 классов), в качестве отправной точки для новой задачи. Преимущества:

- Требуется **значительно меньше данных**.
- Обучение происходит **быстрее**.
- Модель уже умеет распознавать базовые признаки (края, текстуры, формы).

### Fine-tuning vs Feature Extraction

В данном коде используется **Feature Extraction** — веса VGG16 заморожены. Альтернативный подход — **Fine-tuning**: разморозка части слоёв и дообучение всей сети с маленьким learning rate.

### Data Pipeline Optimization

Использование `.cache()` и `.prefetch()` — стандартная практика для ускорения обучения на больших датасетах. Без этого GPU будет простаивать в ожидании данных.

---

## Потенциальные проблемы и улучшения

### 1. Слишком маленький Dense-слой (6 нейронов)

```python
# Текущий код:
layers.Dense(6, activation="relu"),
```

**Проблема:** 6 нейронов — это крайне мало. VGG16 выдаёт 8192 признака, которые сжимаются до 6. Это может привести к потере информации.

**Рекомендация:** Увеличить до 256–512 нейронов и добавить Dropout для регуляризации:

```python
layers.Dense(512, activation="relu"),
layers.Dropout(0.5),
```

### 2. Отсутствие регуляризации

Нет Dropout, L1/L2-регуляризации или BatchNormalization. При 30 эпохах модель может переобучиться.

### 3. Размер изображения 128×128

VGG16 изначально спроектирован для 224×224. Уменьшение до 128×128 снижает качество признаков. Однако это ускоряет обучение.

### 4. Отсутствие Data Augmentation

Аугментация (повороты, сдвиги, отражения) значительно улучшает обобщающую способность. Можно добавить:

```python
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])
```

### 5. Использование Nearest интерполяции

`interpolation="nearest"` даёт ступенчатый ресайз. Лучше использовать `"bilinear"` или `"bicubic"` для более гладкого результата.

### 6. Отсутствие Early Stopping

```python
early_stop = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True,
)
model.fit(..., callbacks=[early_stop])
```

Это предотвратит переобучение и сэкономит время.

### 7. Отсутствие ReduceLROnPlateau

```python
reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=3,
)
```

Уменьшает learning rate при плато валидационной потери.

---

## Улучшенная версия кода

```python
# Улучшенная архитектура
model = keras.Sequential([
    pretrained_base,
    layers.GlobalAveragePooling2D(),  # вместо Flatten — меньше параметров
    layers.Dense(256, activation="relu"),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(1, activation="sigmoid"),
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    loss="binary_crossentropy",
    metrics=["binary_accuracy", keras.metrics.AUC(name="auc")],
)

callbacks = [
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3),
]

history = model.fit(
    ds_train,
    validation_data=ds_valid,
    epochs=30,
    callbacks=callbacks,
)
```

**Изменения:**

| Что изменено | Почему |
|---|---|
| `GlobalAveragePooling2D` вместо `Flatten` | Резко уменьшает количество параметров, снижает переобучение |
| 256 нейронов вместо 6 | Больше выразительная способность |
| `BatchNormalization` | Стабилизирует обучение |
| `Dropout(0.5)` | Регуляризация |
| `learning_rate=1e-4` | Меньший шаг для стабильности |
| `EarlyStopping` | Остановка при переобучении |
| `ReduceLROnPlateau` | Адаптивное снижение learning rate |
| `AUC` метрика | Дополнительная метрика для несбалансированных данных |

---

## Когда полезен этот код

1. **Классификация медицинских изображений** (рентген, КТ, МРТ) — бинарные задачи (болезнь/здоров).
2. **Контроль качества на производстве** — дефект/норма.
3. **Бинарная классификация в мобильных приложениях** — после оптимизации и квантизации.
4. **Образовательные цели** — изучение Transfer Learning на понятном примере.

---

## Выводы

Скрипт [`train_01.py`](Lessons/4._Computer_Vision/src/train_01.py) представляет собой **минимально работоспособный пример Transfer Learning** для бинарной классификации. Он демонстрирует:

- Загрузку и предобработку данных через `image_dataset_from_directory`.
- Оптимизацию пайплайна (cache + prefetch).
- Использование предобученной VGG16.
- Построение и компиляцию модели.
- Обучение и визуализацию результатов.

**Основные недостатки:** маленький Dense-слой (6 нейронов), отсутствие регуляризации, аугментации и колбэков. Код подходит для быстрого прототипирования, но требует доработки для production-использования.