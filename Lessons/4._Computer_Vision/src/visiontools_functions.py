"""
Функции, эмулирующие поведение learntools.computer_vision.visiontools
для урока "The Sliding Window" (Урок 4 Computer Vision).

Содержит:
- edge, blur, bottom_sobel, emboss, sharpen, circle — ядра свёртки
- read_image — загрузка изображения
- show_kernel — отображение ядра
- show_extraction — визуализация извлечения признаков (свёртка + пулинг)
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


# ──────────────────────────────────────────────
# Ядра свёртки (kernels)
# ──────────────────────────────────────────────

# Edge detection (обнаружение границ)
edge = tf.constant([
    [-1, -1, -1],
    [-1,  8, -1],
    [-1, -1, -1],
], dtype=tf.float32)

# Blur (размытие)
blur = tf.constant([
    [1, 2, 1],
    [2, 4, 2],
    [1, 2, 1],
], dtype=tf.float32) / 16.0

# Bottom Sobel (выделение горизонтальных границ снизу)
bottom_sobel = tf.constant([
    [-1, -2, -1],
    [0,  0,  0],
    [1,  2,  1],
], dtype=tf.float32)

# Emboss (тиснение)
emboss = tf.constant([
    [-2, -1, 0],
    [-1,  1, 1],
    [0,  1, 2],
], dtype=tf.float32)

# Sharpen (повышение резкости)
sharpen = tf.constant([
    [0, -1,  0],
    [-1,  5, -1],
    [0, -1,  0],
], dtype=tf.float32)


# ──────────────────────────────────────────────
# Вспомогательные функции
# ──────────────────────────────────────────────

def circle(shape, val=1.0, r_shrink=4, thickness=1):
    """
    Создаёт изображение окружности (только контур).

    Параметры:
        shape (tuple): (height, width) — размер изображения.
        val (float): значение для пикселей окружности (по умолч. 1.0).
        r_shrink (int): коэффициент сжатия радиуса (по умолч. 4).
        thickness (int): толщина линии окружности в пикселях (по умолч. 1).

    Возвращает:
        np.ndarray: изображение окружности размером shape.
    """
    h, w = shape
    Y, X = np.ogrid[:h, :w]
    center_y, center_x = (h - 1) / 2, (w - 1) / 2

    # Основной радиус
    radius = min(h, w) / 2 - min(h, w) / r_shrink

    # Внутренний и внешний радиусы (для создания линии заданной толщины)
    inner_radius = radius - thickness / 2
    outer_radius = radius + thickness / 2

    # Создаём маску для окружности (только контур)
    # Пиксели должны быть между inner_radius и outer_radius
    dist_sq = (X - center_x) ** 2 + (Y - center_y) ** 2
    mask = (dist_sq >= inner_radius ** 2) & (dist_sq <= outer_radius ** 2)

    img = np.zeros(shape, dtype=np.float32)
    img[mask] = val
    return img


def read_image(path, channels=1):
    """
    Читает изображение из файла и возвращает тензор.

    Параметры:
        path (str): путь к файлу изображения.
        channels (int): количество каналов (1 — grayscale, 3 — RGB).

    Возвращает:
        tf.Tensor: тензор изображения с размерностью [height, width, channels].
    """
    img_str = tf.io.read_file(path)
    if channels == 1:
        img = tf.image.decode_jpeg(img_str, channels=1)
    else:
        img = tf.image.decode_jpeg(img_str, channels=channels)
    img = tf.image.convert_image_dtype(img, dtype=tf.float32)
    return img


def show_kernel(kernel, digits=2, text_size=20):
    """
    Отображает ядро свёртки в виде тепловой карты с подписями значений.

    Параметры:
        kernel (tf.Tensor): ядро свёртки (2D или 3D).
        digits (int): количество знаков после запятой (по умолч. 2).
        text_size (int): размер шрифта подписей (по умолч. 20).
    """
    krn = tf.squeeze(kernel).numpy()
    plt.imshow(krn, cmap='magma')
    plt.axis('off')

    for i in range(krn.shape[0]):
        for j in range(krn.shape[1]):
            plt.text(
                j, i, f'{krn[i, j]:.{digits}f}',
                ha='center', va='center',
                fontsize=text_size,
                color='white' if abs(krn[i, j]) < krn.max() / 2 else 'black',
            )


def show_extraction(image, kernel,
                    conv_stride=1,
                    conv_padding='valid',
                    pool_size=2,
                    pool_stride=2,
                    pool_padding='same',
                    subplot_shape=(1, 4),
                    figsize=(14, 6)):
    """
    Визуализирует процесс извлечения признаков:
    исходное изображение -> свёртка -> ReLU -> пулинг.

    Параметры:
        image (tf.Tensor): входное изображение [H, W, 1].
        kernel (tf.Tensor): ядро свёртки [K_h, K_w].
        conv_stride (int): шаг свёртки (по умолч. 1).
        conv_padding (str): тип паддинга свёртки ('valid' или 'same').
        pool_size (int): размер окна пулинга (по умолч. 2).
        pool_stride (int): шаг пулинга (по умолч. 2).
        pool_padding (str): тип паддинга пулинга ('valid' или 'same').
        subplot_shape (tuple): расположение подграфиков (rows, cols).
        figsize (tuple): размер фигуры (width, height).
    """
    # Подготовка ядра: [K_h, K_w] -> [K_h, K_w, 1, 1]
    kernel_4d = tf.reshape(kernel, shape=(*kernel.shape, 1, 1))

    # Свёртка
    conv_output = tf.nn.conv2d(
        input=image[tf.newaxis, ...],  # [1, H, W, 1]
        filters=kernel_4d,
        strides=conv_stride,
        padding=conv_padding.upper(),
    )

    # ReLU активация
    relu_output = tf.nn.relu(conv_output)

    # Max пулинг
    pool_output = tf.nn.max_pool2d(
        input=relu_output,
        ksize=pool_size,
        strides=pool_stride,
        padding=pool_padding.upper(),
    )

    titles = ['Original', 'Convolution', 'ReLU', 'MaxPool']
    outputs = [image, conv_output, relu_output, pool_output]

    plt.figure(figsize=figsize)
    for i, (out, title) in enumerate(zip(outputs, titles)):
        plt.subplot(*subplot_shape, i + 1)
        plt.imshow(tf.squeeze(out), cmap='magma')
        plt.axis('off')
        plt.title(title)
    plt.tight_layout()
    plt.show()


# ──────────────────────────────────────────────
# Функция 1: Загрузка и подготовка изображений
# ──────────────────────────────────────────────

def load_and_prepare_images(image_dir='../input/computer-vision-resources/'):
    """
    Загружает и подготавливает изображения для экспериментов.

    Параметры:
        image_dir (str): путь к директории с изображениями.

    Возвращает:
        list[tuple[tf.Tensor, str]]: список кортежей (изображение, название).
    """
    circle_64 = tf.expand_dims(circle([64, 64], val=1.0, r_shrink=4), axis=-1)
    kaggle_k = read_image(image_dir + 'k.jpg', channels=1)
    car = read_image(image_dir + 'car_illus.jpg', channels=1)
    car = tf.image.resize(car, size=[200, 200])

    images = [
        (circle_64, "circle_64"),
        (kaggle_k, "kaggle_k"),
        (car, "car"),
    ]
    return images


# ──────────────────────────────────────────────
# Функция 2: Отображение изображений
# ──────────────────────────────────────────────

def display_images(images, figsize=(14, 4)):
    """
    Отображает список изображений в один ряд.

    Параметры:
        images (list): список кортежей (изображение, название).
        figsize (tuple): размер фигуры (width, height).
    """
    plt.figure(figsize=figsize)
    for i, (img, title) in enumerate(images):
        plt.subplot(1, len(images), i + 1)
        plt.imshow(tf.squeeze(img))
        plt.axis('off')
        plt.title(title)
    plt.show()


# ──────────────────────────────────────────────
# Функция 3: Отображение ядер свёртки
# ──────────────────────────────────────────────

def display_kernels(kernels_list=None, figsize=(14, 4)):
    """
    Отображает список ядер свёртки.

    Параметры:
        kernels_list (list[tuple[tf.Tensor, str]], optional):
            список кортежей (ядро, название). Если None, используются все ядра.
        figsize (tuple): размер фигуры (width, height).
    """
    if kernels_list is None:
        kernels_list = [
            (edge, "edge"),
            (blur, "blur"),
            (bottom_sobel, "bottom_sobel"),
            (emboss, "emboss"),
            (sharpen, "sharpen"),
        ]

    plt.figure(figsize=figsize)
    for i, (krn, title) in enumerate(kernels_list):
        plt.subplot(1, len(kernels_list), i + 1)
        show_kernel(krn, digits=2, text_size=20)
        plt.title(title)
    plt.show()


# ──────────────────────────────────────────────
# Функция 4: Извлечение признаков (полный пайплайн)
# ──────────────────────────────────────────────

def feature_extraction_pipeline(
    image,
    kernel,
    conv_stride=1,
    conv_padding='valid',
    pool_size=2,
    pool_stride=2,
    pool_padding='same',
    subplot_shape=(1, 4),
    figsize=(14, 6),
):
    """
    Выполняет полный пайплайн извлечения признаков:
    свёртка -> ReLU -> MaxPool и визуализирует результат.

    Параметры:
        image (tf.Tensor): входное изображение [H, W, 1].
        kernel (tf.Tensor): ядро свёртки.
        conv_stride (int): шаг свёртки.
        conv_padding (str): паддинг свёртки ('valid' или 'same').
        pool_size (int): размер окна пулинга.
        pool_stride (int): шаг пулинга.
        pool_padding (str): паддинг пулинга ('valid' или 'same').
        subplot_shape (tuple): расположение подграфиков.
        figsize (tuple): размер фигуры.

    Возвращает:
        dict: словарь с этапами обработки:
            {'original': tf.Tensor, 'conv': tf.Tensor,
             'relu': tf.Tensor, 'pool': tf.Tensor}
    """
    # Подготовка ядра: [K_h, K_w] -> [K_h, K_w, 1, 1]
    kernel_4d = tf.reshape(kernel, shape=(*kernel.shape, 1, 1))

    # Свёртка
    conv_output = tf.nn.conv2d(
        input=image[tf.newaxis, ...],
        filters=kernel_4d,
        strides=conv_stride,
        padding=conv_padding.upper(),
    )

    # ReLU
    relu_output = tf.nn.relu(conv_output)

    # MaxPool
    pool_output = tf.nn.max_pool2d(
        input=relu_output,
        ksize=pool_size,
        strides=pool_stride,
        padding=pool_padding.upper(),
    )

    # Визуализация
    titles = ['Original', 'Convolution', 'ReLU', 'MaxPool']
    outputs = [image, conv_output, relu_output, pool_output]

    plt.figure(figsize=figsize)
    for i, (out, title) in enumerate(zip(outputs, titles)):
        plt.subplot(*subplot_shape, i + 1)
        plt.imshow(tf.squeeze(out), cmap='magma')
        plt.axis('off')
        plt.title(title)
    plt.tight_layout()
    plt.show()

    return {
        'original': image,
        'conv': conv_output,
        'relu': relu_output,
        'pool': pool_output,
    }


# ──────────────────────────────────────────────
# Функция 5: Одномерная свёртка временного ряда
# ──────────────────────────────────────────────

def conv1d_time_series(
    ts_data,
    kernel,
    stride=1,
    padding='VALID',
):
    """
    Выполняет одномерную свёртку временного ряда и визуализирует результат.

    Параметры:
        ts_data (pd.DataFrame): временной ряд с колонкой значений.
        kernel (tf.Tensor): одномерное ядро свёртки.
        stride (int): шаг свёртки (по умолч. 1).
        padding (str): паддинг ('VALID' или 'SAME').

    Возвращает:
        pd.Series: отфильтрованный временной ряд.
    """
    import pandas as pd

    # Подготовка данных: [batch, timesteps, channels]
    ts_np = ts_data.to_numpy()
    ts_np = tf.expand_dims(ts_np, axis=0)  # [1, T, 1]
    ts_np = tf.cast(ts_np, dtype=tf.float32)

    # Подготовка ядра: [kernel_size, in_channels, out_channels]
    kern = tf.reshape(kernel, shape=(*kernel.shape, 1, 1))

    # Одномерная свёртка
    ts_filter = tf.nn.conv1d(
        input=ts_np,
        filters=kern,
        stride=stride,
        padding=padding,
    )

    # Форматирование как Pandas Series
    filtered_series = pd.Series(tf.squeeze(ts_filter).numpy())
    filtered_series.plot()
    plt.title('1D Convolution Result')
    plt.show()

    return filtered_series


# ──────────────────────────────────────────────
# Функция 6: Полный демонстрационный запуск
# ──────────────────────────────────────────────

def run_full_demo(image_dir='../input/computer-vision-resources/'):
    """
    Запускает полную демонстрацию:
    1. Загрузка и отображение изображений
    2. Отображение ядер свёртки
    3. Пример извлечения признаков

    Параметры:
        image_dir (str): путь к директории с изображениями.
    """
    # Шаг 1: Загрузка изображений
    images = load_and_prepare_images(image_dir)
    display_images(images)

    # Шаг 2: Отображение ядер
    display_kernels()

    # Шаг 3: Пример извлечения признаков
    image = images[0][0]  # circle_64
    kernel = bottom_sobel

    feature_extraction_pipeline(
        image, kernel,
        conv_stride=1,
        conv_padding='valid',
        pool_size=2,
        pool_stride=2,
        pool_padding='same',
        subplot_shape=(1, 4),
        figsize=(14, 6),
    )
