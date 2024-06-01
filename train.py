import os
import torch
import random
import numpy as np
from roboflow import Roboflow
from ultralytics import YOLO

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# 数据集的下载
rf = Roboflow(api_key="")   #请在这里输入你的api key！！
project = rf.workspace("meredith-lo-pmqx7").project("asl-project")
version = project.version(16)
dataset = version.download("yolov8")
print(f"Dataset downloaded to: {dataset.location}")

# 检查数据集的目录结构
data_path = os.path.join(dataset.location, "data.yaml")
print(f"Data path: {data_path}")
print(os.listdir(dataset.location))
assert os.path.exists(data_path), f"data.yaml not found at {data_path}"

# 初始化YOLOv8模型
model = YOLO("yolov8n.pt")  # 使用了预训练模型（yolov8n.pt）

# 训练模型
model.train(data=data_path, epochs=50, batch=16, imgsz=640)

# 评估模型
results = model.val()
print(results)

# 保存最好的模型
best_model_path = "runs/train/exp/weights/best.pt"

# 加载最佳权重文件
model = YOLO(best_model_path)

# 对单张图片进行推理（替换为你的测试图片路径）
test_image_path = "path/to/test/image.jpg"
results = model(test_image_path)
results.show()  # 显示结果
results.save("path/to/save/results/")  # 保存结果
