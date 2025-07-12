import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import torch
from YOLOv8.model.yolo import Model

img = torch.randn((1, 3, 640, 640))
model = Model()
print(model(img)[0].shape)
