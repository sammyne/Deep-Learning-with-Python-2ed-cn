# 代码清单 7-10 多输入、多输出的函数式模型
from tensorflow import keras
from tensorflow.keras import layers

vocabulary_size = 10000
num_tags = 100
num_departments = 4

title = keras.Input(shape=(vocabulary_size,), name="title")
text_body = keras.Input(shape=(vocabulary_size,), name="text_body")
tags = keras.Input(shape=(num_tags,), name="tags")

features = layers.Concatenate()([title, text_body, tags])
features = layers.Dense(64, activation="relu")(features)

priority = layers.Dense(1, activation="sigmoid", name="priority")(features)
department = layers.Dense(num_departments, activation="softmax", name="department")(
    features
)

model = keras.Model(inputs=[title, text_body, tags], outputs=[priority, department])

# 代码清单 7-12 通过给定输入和目标组成的字典来训练模型
import numpy as np

num_samples = 1280

title_data = np.random.randint(0, 2, size=(num_samples, vocabulary_size))
text_body_data = np.random.randint(0, 2, size=(num_samples, vocabulary_size))
tags_data = np.random.randint(0, 2, size=(num_samples, num_tags))

priority_data = np.random.random(size=(num_samples, 1))
department_data = np.random.randint(0, 2, size=(num_samples, num_departments))

model.compile(
    optimizer="rmsprop",
    loss={"priority": "mean_squared_error", "department": "categorical_crossentropy"},
    metrics={"priority": ["mean_absolute_error"], "department": ["accuracy"]},
)
model.fit(
    {"title": title_data, "text_body": text_body_data, "tags": tags_data},
    {"priority": priority_data, "department": department_data},
    epochs=1,
)
model.evaluate(
    {"title": title_data, "text_body": text_body_data, "tags": tags_data},
    {"priority": priority_data, "department": department_data},
)
priority_preds, department_preds = model.predict(
    {"title": title_data, "text_body": text_body_data, "tags": tags_data}
)

print(priority_preds)
print(department_preds)

keras.utils.plot_model(model, "data/ticket_classifier.png")
keras.utils.plot_model(
    model, "data/ticket_classifier_with_shape_info.png", show_shapes=True
)

# 代码清单 7-13 检索函数式模型某一层的输入或输出
print(model.layers)
print(model.layers[3].input)
print(model.layers[3].output)

# 代码清单 7-14 重复使用中间层的输出，创建一个新模型
eatures = model.layers[4].output
difficulty = layers.Dense(3, activation="softmax", name="difficulty")(features)

new_model = keras.Model(
    inputs=[title, text_body, tags], outputs=[priority, department, difficulty]
)
keras.utils.plot_model(
    new_model, "data/updated_ticket_classifier.png", show_shapes=True
)
