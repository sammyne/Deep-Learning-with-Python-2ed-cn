# 代码清单 7-6 利用 name参数命名模型和层
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential(name="my_example_model")
model.add(layers.Dense(64, activation="relu", name="my_first_layer"))
model.add(layers.Dense(10, activation="softmax", name="my_last_layer"))
model.build((None, 3))

print(model.summary())
