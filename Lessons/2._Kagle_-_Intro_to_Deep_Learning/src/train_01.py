import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow import keras
from tensorflow.keras import layers

# Создаём сеть с 1 линейным нейроном
model = keras.Sequential([
    layers.Dense(units=1, input_shape=[3])
])
