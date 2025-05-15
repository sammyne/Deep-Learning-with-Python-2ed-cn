# 代码清单 7-1 Sequential类
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential(
    [layers.Dense(64, activation="relu"), layers.Dense(10, activation="softmax")]
)
