# 代码清单 7-9 带有两个 Dense层的简单函数式模型
from tensorflow import keras
from tensorflow.keras import layers

inputs = keras.Input(shape=(3,), name="my_input")
features = layers.Dense(64, activation="relu")(inputs)
outputs = layers.Dense(10, activation="softmax")(features)
model = keras.Model(inputs=inputs, outputs=outputs)

print(inputs.shape)
print(inputs.dtype)

print(features.shape)

print(model.summary())
