from tensorflow.keras.datasets import mnist
import numpy as np

(train_images, train_labels), (test_data, test_labels) = mnist.load_data()
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype("float32") / 255

test_data = test_data.reshape((10000, 28 * 28))
train_data = train_images.astype("float32") / 255

from tensorflow import keras
from tensorflow.keras import layers


def get_model():
    model = keras.Sequential(
        [layers.Dense(512, activation="relu"), layers.Dense(10, activation="softmax")]
    )
    model.compile(
        optimizer="rmsprop",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# 代码清单 5-5 留出验证（注意，为简单起见省略了标签）
num_validation_samples = 10000
np.random.shuffle(train_images)
validation_data = train_images[:num_validation_samples]
training_data = train_images[num_validation_samples:]

validation_labels = train_labels[:num_validation_samples]
training_labels = train_labels[num_validation_samples:]

model = get_model()
model.fit(training_data, train_labels, epochs=10, batch_size=128, validation_split=0.2)
validation_score = model.evaluate(validation_data, validation_labels)
print(validation_score)

# 代码清单 5-6 K 折交叉验证（注意，为简单起见省略了标签）
data = train_images
labels = train_labels

k = 3
num_validation_samples = len(data) // k
np.random.shuffle(data)
validation_scores = []
for fold in range(k):
    validation_data = data[
        num_validation_samples * fold : num_validation_samples * (fold + 1)
    ]
    training_data = np.concatenate(
        (
            data[: num_validation_samples * fold],
            data[num_validation_samples * (fold + 1) :],
        )
    )

    validation_labels = labels[
        num_validation_samples * fold : num_validation_samples * (fold + 1)
    ]
    training_labels = np.concatenate(
        (
            labels[: num_validation_samples * fold],
            labels[num_validation_samples * (fold + 1) :],
        )
    )

    model = get_model()
    model.fit(
        training_data, train_labels, epochs=10, batch_size=128, validation_split=0.2
    )
    validation_score = model.evaluate(validation_data, validation_labels)
    validation_scores.append(validation_score)

validation_score = np.average(validation_scores)

model = get_model()
model.fit(data, labels, epochs=10, batch_size=128, validation_split=0.2)

test_score = model.evaluate(test_data, test_labels)

print(validation_score)
print(test_score)
