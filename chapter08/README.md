# 08. 计算机视觉深度学习入门

> 建议在 Google Colab 上运行，本地 CPU 运行非常慢。

## 准备数据
1. 创建 [kaggle](https://www.kaggle.com/) 账号；
1. kaggle 官网的个人页面验证手机号；
  - 缺少这一步会导致无法加入竞赛，进而导致无法下载数据集；
1. 在 [data](https://www.kaggle.com/competitions/dogs-vs-cats/overview) 页面点击加入竞赛；
  - 缺少这一步会导致后续的数据下载报 403 错误；
1. 从 [个人设置](https://www.kaggle.com/settings) 页面创建并下载 token 得到 kaggle.json 文件，并将其存放到 ~/.kaggle/kaggle.json 路径；
1. 下载数据集
    ```bash
    uv run kaggle competitions download -c dogs-vs-cats -p data/
    ```
1. 解压数据
  ```bash
  unzip data/dogs-vs-cats.zip -d data
  ```
1. 解压训练数据
  ```bash
  unzip data/train.zip -d data
  ```
