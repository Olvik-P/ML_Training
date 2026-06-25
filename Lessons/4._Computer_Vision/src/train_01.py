import os
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing import image_dataset_from_directory


# Reproducability
def set_seed(seed=31415):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"


# Data Pipeline
def convert_to_float(image, label):
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    return image, label


set_seed(31415)

# Set Matplotlib defaults
plt.rc("figure", autolayout=True)
plt.rc(
    "axes",
    labelweight="bold",
    labelsize="large",
    titleweight="bold",
    titlesize=18,
    titlepad=10,
)
plt.rc("image", cmap="magma")
warnings.filterwarnings("ignore")  # to clean up output cells


# Load training and validation sets
ds_train_ = image_dataset_from_directory(
    "../data/train",
    labels="inferred",
    label_mode="binary",
    image_size=[128, 128],
    interpolation="nearest",
    batch_size=8,
    shuffle=True,
)
ds_valid_ = image_dataset_from_directory(
    "../data/valid",
    labels="inferred",
    label_mode="binary",
    image_size=[128, 128],
    interpolation="nearest",
    batch_size=8,
    shuffle=False,
)

AUTOTUNE = tf.data.experimental.AUTOTUNE
ds_train = ds_train_.map(
    convert_to_float).cache().prefetch(buffer_size=AUTOTUNE)
ds_valid = ds_valid_.map(
    convert_to_float).cache().prefetch(buffer_size=AUTOTUNE)

# Загружаем предобученную VGG16 из официальных весов
pretrained_base = VGG16(
    weights="imagenet",  # или None для необученной
    include_top=False,  # убираем полносвязные слои
    input_shape=(128, 128, 3),
)

pretrained_base.trainable = False

model = keras.Sequential(
    [
        pretrained_base,
        layers.Flatten(),
        layers.Dense(6, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(1, activation="sigmoid"),
    ]
)

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["binary_accuracy"],
)

start_time = time.time()

history = model.fit(
    ds_train,
    validation_data=ds_valid,
    epochs=30,
    verbose=1,
)

end_time = time.time()
print(f"\n⏱️ Время обучения: {(end_time - start_time)/60:.2f} минут")

history_frame = pd.DataFrame(history.history)
history_frame.loc[:, ["loss", "val_loss"]].plot()
history_frame.loc[:, ["binary_accuracy", "val_binary_accuracy"]].plot()

plt.show()  # ← добавьте эту строку
