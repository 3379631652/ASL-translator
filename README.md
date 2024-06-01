# ASL-translator

本项目基于 YOLOv8 实现了美国手语（ASL）的翻译，并使用 TTS（文本转语音）将 ASL 转换为语音输出。

## 文件描述

- `train.py`：用于下载数据集并训练模型的文件
- `best.pt`：训练好的模型文件
- `translator.py`：可以调用模型将 ASL 转换为语音输出

## 注意

1. 请参照 `requirements.txt` 搭建环境
2. 请将 `translator.py` 和 `best.pt` 放在一个目录下
3. 如果要训练自己的模型，数据集可以通过 `train.py` 从 Roboflow 下载,并填写自己的api key
![image](https://github.com/3379631652/ASL-translator/assets/90537351/4b7037eb-db03-4a69-b7ee-d7956d518e83)


