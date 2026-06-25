import tensorflow as tf
import matplotlib.pyplot as plt
import warnings
import numpy as np

plt.rc('figure', autolayout=True)
plt.rc('axes', labelweight='bold', labelsize='large',
       titleweight='bold', titlesize=18, titlepad=10)
plt.rc('image', cmap='magma')
warnings.filterwarnings("ignore")  # для чистоты выходных ячеек

# ============================================================
# ФИЧА #1: Интерактивный выбор ядра свёртки (несколько детекторов)
# ============================================================
KERNELS = {
    'Laplacian (границы)': tf.constant([
        [-1, -1, -1],
        [-1,  8, -1],
        [-1, -1, -1],
    ], dtype=tf.float32),
    'Sobel X (верт. границы)': tf.constant([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1],
    ], dtype=tf.float32),
    'Sobel Y (гор. границы)': tf.constant([
        [-1, -2, -1],
        [0,  0,  0],
        [1,  2,  1],
    ], dtype=tf.float32),
    'Box Blur (размытие)': tf.constant([
        [1/9, 1/9, 1/9],
        [1/9, 1/9, 1/9],
        [1/9, 1/9, 1/9],
    ], dtype=tf.float32),
    'Sharpen (резкость)': tf.constant([
        [0, -1,  0],
        [-1,  5, -1],
        [0, -1,  0],
    ], dtype=tf.float32),
    'Horizontal Lines': tf.constant([
        [-1, -1, -1],
        [2,  2,  2],
        [-1, -1, -1],
    ], dtype=tf.float32),
}

# Выбираем ядро (можно заменить на любой ключ из KERNELS)
SELECTED_KERNEL_NAME = 'Laplacian (границы)'
kernel = KERNELS[SELECTED_KERNEL_NAME]

# ============================================================
# Чтение изображения
# ============================================================
image_path = '../img/car_feature.jpg'
image = tf.io.read_file(image_path)
image = tf.io.decode_jpeg(image)

# Переформатирование для совместимости с батчами
image = tf.image.convert_image_dtype(image, dtype=tf.float32)
image = tf.expand_dims(image, axis=0)  # shape: (1, H, W, C)

# ============================================================
# ФИЧА #2: Визуализация карт признаков (Feature Maps) для нескольких каналов
# ============================================================
# Применяем несколько разных ядер параллельно, как в реальном Conv2D-слое
# Каждое ядро -> отдельный канал на выходе

# Собираем все ядра в один 4D тензор: (H, W, in_channels, out_channels)
multi_kernel = tf.stack(list(KERNELS.values()), axis=-1)  # (3, 3, 1, 6)
multi_kernel = tf.reshape(multi_kernel, [3, 3, 1, len(KERNELS)])

# Однопроходная свёртка со всеми ядрами сразу
feature_maps = tf.nn.conv2d(
    input=image,
    filters=multi_kernel,
    strides=1,
    padding='SAME',
)
feature_maps_activated = tf.nn.relu(feature_maps)  # ReLU после свёртки

# Визуализация всех карт признаков
kernel_names = list(KERNELS.keys())
n_kernels = len(KERNELS)
cols = 3
rows = int(np.ceil(n_kernels / cols))

plt.figure(figsize=(cols * 5, rows * 4))
for i in range(n_kernels):
    plt.subplot(rows, cols, i + 1)
    plt.imshow(tf.squeeze(feature_maps_activated[..., i]), cmap='magma')
    plt.axis('off')
    plt.title(f'{kernel_names[i]}', fontsize=12)
plt.suptitle('Фича #2: Карты признаков после Conv2D + ReLU (6 ядер)',
             fontsize=16, y=1.02)
plt.tight_layout()
plt.show()

# ============================================================
# Демонстрация выбранного ядра (как в оригинале, но с выбором)
# ============================================================
kernel_4d = tf.reshape(kernel, [*kernel.shape, 1, 1])

# Шаг фильтрации
image_filter = tf.nn.conv2d(
    input=image,
    filters=kernel_4d,
    strides=1,
    padding='SAME',
)

# Шаг обнаружения
image_detect = tf.nn.relu(image_filter)

# Покажем, что у нас получилось
plt.figure(figsize=(12, 6))
plt.subplot(131)
plt.imshow(tf.squeeze(image), cmap='gray')
plt.axis('off')
plt.title(f'Вход\n(ядро: {SELECTED_KERNEL_NAME})', fontsize=12)
plt.subplot(132)
plt.imshow(tf.squeeze(image_filter))
plt.axis('off')
plt.title('Фильтр (до ReLU)', fontsize=12)
plt.subplot(133)
plt.imshow(tf.squeeze(image_detect))
plt.axis('off')
plt.title('Обнаружение (после ReLU)', fontsize=12)
plt.suptitle('Фича #1: Выбранное ядро свёртки', fontsize=16, y=1.02)
plt.tight_layout()
plt.show()

# ============================================================
# ФИЧА #3: Сравнение pooling-стратегий: Max vs Average vs Min
# ============================================================
pool_window = (2, 2)
pool_strides = (2, 2)

# Max Pooling
image_maxpool = tf.nn.pool(
    input=image_detect,
    window_shape=pool_window,
    pooling_type='MAX',
    strides=pool_strides,
    padding='SAME',
)

# Average Pooling
image_avgpool = tf.nn.pool(
    input=image_detect,
    window_shape=pool_window,
    pooling_type='AVG',
    strides=pool_strides,
    padding='SAME',
)

# Min Pooling (через отрицание + MAX + обратное отрицание)
image_minpool = -tf.nn.pool(
    input=-image_detect,
    window_shape=pool_window,
    pooling_type='MAX',
    strides=pool_strides,
    padding='SAME',
)

# Визуализация сравнения
plt.figure(figsize=(16, 4))

plt.subplot(141)
plt.imshow(tf.squeeze(image_detect))
plt.axis('off')
plt.title('До Pooling\n(после ReLU)', fontsize=12)

plt.subplot(142)
plt.imshow(tf.squeeze(image_maxpool))
plt.axis('off')
plt.title('MaxPooling\n(сохраняет сильные активации)', fontsize=12)

plt.subplot(143)
plt.imshow(tf.squeeze(image_avgpool))
plt.axis('off')
plt.title('AveragePooling\n(сглаживает, меньше шума)', fontsize=12)

plt.subplot(144)
plt.imshow(tf.squeeze(image_minpool))
plt.axis('off')
plt.title('MinPooling\n(подавляет яркие области)', fontsize=12)

plt.suptitle('Фича #3: Сравнение pooling-стратегий (окно 2×2, stride 2)',
             fontsize=16, y=1.02)
plt.tight_layout()
plt.show()

# Дополнительно: визуализация "механики" MaxPooling
# Показываем, какие именно пиксели были выбраны из окна 2×2
sample_patch = image_detect[0, 4:6, 4:6, :]  # небольшой патч 2×2
pooled_value = tf.reduce_max(sample_patch)

print('=== Демонстрация механики MaxPooling ===')
print(f'Патч 2×2 (значения):\n{sample_patch.numpy().squeeze().round(3)}')
print(f'MaxPooling результат: {pooled_value.numpy():.3f}')
print('(Выбирается максимальное значение из окна 2×2)')
