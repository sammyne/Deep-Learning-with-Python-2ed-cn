import pathlib

from tensorflow import keras
from tensorflow.keras.utils import image_dataset_from_directory

new_base_dir = pathlib.Path("data/cats_vs_dogs_small")

test_dataset = image_dataset_from_directory(
    new_base_dir / "test", image_size=(180, 180), batch_size=32
)

# 代码清单 8-13 在测试集上评估模型
test_model = keras.models.load_model("convnet_from_scratch.keras")
test_loss, test_acc = test_model.evaluate(test_dataset)
print(f"Test accuracy: {test_acc:.3f}")
