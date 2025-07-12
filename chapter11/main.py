import os

import datasets
from torch.utils.data import DataLoader

def text_dataset_from_dir(dir):
    out = []
    for f in filter(lambda x: x.is_dir(), os.scandir(dir)):
        label = 1 if f.name == "pos" else 0
        v = datasets.load_dataset("text", data_dir=f.path).map(lambda x: {"label": label})["train"]
        out.append(v)

    ds =  datasets.concatenate_datasets(out).with_format("torch") 

    # return DataLoader(ds, batch_size=32) 


# for f in os.scandir("aclImdb/world"):
#     print(f.is_dir(), f.path, f.name)

train_ds = text_dataset_from_dir("aclImdb/train")
print(len(train_ds)*32)

for v in train_ds:
    break
    inputs, targets = v["text"], v["label"]

    print("inputs.len", len(inputs))
    print('inputs[0]:', inputs[0])

    print("targets.shape: ", targets.shape)
    print("targets[0]: ", targets[0])
    print("targets[0].dtype: ", targets[0].dtype)

    break

    print("inputs.shape:", inputs.shape)
    print("inputs.dtype:", inputs.dtype)
    print("targets.shape:", targets.shape)
    print("targets.dtype:", targets.dtype)
    print("inputs[0]:", inputs[0])
    print("targets[0]:", targets[0])
    break