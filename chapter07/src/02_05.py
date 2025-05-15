# 代码清单 7-2 逐步构建序贯模型
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential()
model.add(layers.Dense(64, activation="relu"))
model.add(layers.Dense(10, activation="softmax"))

# 代码清单 7-3 尚未完成构建的模型没有权重
print(model.weights)

# 代码清单 7-4 通过第一次调用模型来完成构建
model.build(input_shape=(None, 3))
# 现在可以检索模型权重
print(model.weights)

# 代码清单 7-5 summary()方法
print(model.summary())
