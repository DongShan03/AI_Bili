import os, torch
from model import MobileNetV2 as Net
batch_size = 32
net_name = "MobileNetV2"
data_name = "flower_data"
epochs = 2
num_classes = 5
learn_rate = 0.0002
dir_root = os.path.dirname(__file__)
data_root = os.path.join(dir_root, "..", "data", data_name)
class_indices = os.path.join(data_root, "class_indices.json")
save_path = os.path.join(dir_root, net_name + '_' + data_name + ".pth")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = Net(num_classes=num_classes, init_weights=True).to(device)
