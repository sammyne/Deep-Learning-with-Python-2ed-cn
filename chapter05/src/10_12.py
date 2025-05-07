# 代码清单 5-10 初始模型
from tensorflow.keras.datasets import imdb
from tensorflow import keras
from tensorflow.keras import layers
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

# 代码清单 5-11 容量更小的模型
model = keras.Sequential(
    [
        layers.Dense(4, activation="relu"),
        layers.Dense(4, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ]
)

# 代码清单 5-12 容量更大的模型
model = keras.Sequential(
    [
        layers.Dense(512, activation="relu"),
        layers.Dense(512, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ]
)
model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["accuracy"])
history_larger = model.fit(
    train_data, train_labels, epochs=20, batch_size=512, validation_split=0.4
)

model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["accuracy"])
history_smaller = model.fit(
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
    history_smaller.history["val_loss"],
    "b",
    label="Validation loss of smaller model",
)
plt.plot(
    epochs,
    history_larger.history["val_loss"],
    "r",
    label="Validation loss of larger model",
)
# plt.title("Validation loss of original model")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()
