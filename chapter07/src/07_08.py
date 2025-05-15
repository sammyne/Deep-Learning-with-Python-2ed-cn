# 代码清单 7-7 提前声明模型的输入形状
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential()
model.add(keras.Input(shape=(3,)))
model.add(layers.Dense(64, activation="relu"))

# 代码清单 7-8 使用 summary()跟踪模型输出形状的变化
print(model.summary())

model.add(layers.Dense(10, activation="softmax"))
print(model.summary())
