# 代码清单 5-13 向模型中添加 L2 权重正则化
from tensorflow.keras.datasets import imdb
from tensorflow import keras
from tensorflow.keras import layers, regularizers
import numpy as np
import matplotlib.pyplot as plt

(train_data, train_labels), _ = imdb.load_data(num_words=10000)


def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.0
    return results


train_data = vectorize_sequences(train_data)
model = keras.Sequential(
    [
        layers.Dense(16, activation="relu"),
        layers.Dense(16, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ]
)

model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["accuracy"])
history_original = model.fit(
    train_data, train_labels, epochs=20, batch_size=512, validation_split=0.4
)

model = keras.Sequential(
    [
        layers.Dense(16, kernel_regularizer=regularizers.l2(0.002), activation="relu"),
        layers.Dense(16, kernel_regularizer=regularizers.l2(0.002), activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ]
)
model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["accuracy"])
history_l2_reg = model.fit(
    train_data, train_labels, epochs=20, batch_size=512, validation_split=0.4
)

# 代码清单 5-14 Keras 中不同的权重正则化项
model = keras.Sequential(
    [
        layers.Dense(
            16,
            kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.001),
            activation="relu",
        ),
        layers.Dense(
            16,
            kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.001),
            activation="relu",
        ),
        layers.Dense(1, activation="sigmoid"),
    ]
)
model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["accuracy"])
history_l1_l2_reg = model.fit(
    train_data, train_labels, epochs=20, batch_size=512, validation_split=0.4
)

epochs = range(1, 21)
plt.plot(
    epochs,
    history_original.history["val_loss"],
    "b--",
    label="Validation loss of original model",
)
plt.plot(
    epochs,
    history_l2_reg.history["val_loss"],
    "b",
    label="Validation loss of L2-regularized model",
)
plt.plot(
    epochs,
    history_l1_l2_reg.history["val_loss"],
    "r",
    label="Validation loss of L1+L2-regularized model",
)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()
