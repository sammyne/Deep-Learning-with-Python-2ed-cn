# 代码清单 4-23 加载波士顿房价数据集
from tensorflow.keras.datasets import boston_housing
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np

(train_data, train_targets), (test_data, test_targets) = (boston_housing.load_data())

print(train_data.shape)
print(test_data.shape)

# 代码清单 4-24 数据标准化
mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std
test_data -= mean
test_data /= std

# 代码清单 4-25 模型定义
def build_model():
    model = keras.Sequential([
        layers.Dense(64, activation="relu"),
        layers.Dense(64, activation="relu"),
        layers.Dense(1)
    ])
    model.compile(optimizer="rmsprop", loss="mse", metrics=["mae"])
    return model

# 代码清单 4-26 K 折交叉验证
k = 4
num_val_samples = len(train_data) // k
num_epochs = 100
all_scores = []
for i in range(k):
    # 准备验证数据：第 k 个分区的数据
    print(f"Processing fold #{i}")
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]
    partial_train_data = np.concatenate([
        train_data[:i * num_val_samples],
        train_data[(i + 1) * num_val_samples:]],
        axis=0)
    partial_train_targets = np.concatenate([
            train_targets[:i * num_val_samples],
            train_targets[(i + 1) * num_val_samples:],
        ],
        axis=0)

    model = build_model()
    model.fit(partial_train_data, partial_train_targets, epochs=num_epochs, batch_size=16, verbose=0)
    val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
    all_scores.append(val_mae)

print(all_scores)
print(np.mean(all_scores))

# 代码清单 4-27 保存每折的验证分数
num_epochs = 500
all_mae_histories = []
for i in range(k):
    # 准备验证数据：第 k个分区的数据
    print(f"Processing fold #{i}")
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]
    partial_train_data = np.concatenate([
            train_data[:i * num_val_samples],
            train_data[(i + 1) * num_val_samples:],
        ],
        axis=0)
    partial_train_targets = np.concatenate([
            train_targets[:i * num_val_samples],
            train_targets[(i + 1) * num_val_samples:],
        ],
        axis=0)

    model = build_model()
    history = model.fit(partial_train_data, partial_train_targets, validation_data=(val_data, val_targets), epochs=num_epochs, batch_size=16, verbose=0)
    mae_history = history.history["val_mae"]
    all_mae_histories.append(mae_history)

# 代码清单 4-28 计算每轮的 K 折验证分数平均值
average_mae_history = [np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]

# 代码清单 4-29 绘制验证 MAE 曲线
plt.plot(range(1, len(average_mae_history) + 1), average_mae_history)
plt.xlabel("Epochs")
plt.ylabel("Validation MAE")
plt.show()

# 代码清单 4-30 绘制验证 MAE 曲线（剔除前 10 个数据点）
truncated_mae_history = average_mae_history[10:]
plt.plot(range(1, len(truncated_mae_history) + 1), truncated_mae_history)
plt.xlabel("Epochs")
plt.ylabel("Validation MAE")
plt.show()

# 代码清单 4-31 训练最终模型
# 一个全新的已编译模型
model = build_model()
# 在所有训练数据上训练模型
model.fit(train_data, train_targets, epochs=130, batch_size=16, verbose=0)
test_mse_score, test_mae_score = model.evaluate(test_data, test_targets)
print(test_mae_score )


predictions = model.predict(test_data)
print(predictions[0])
