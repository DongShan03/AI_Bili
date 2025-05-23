import os
batch_size = 32
net_name = "GoogleNet"
data_name = "flower_data"
epochs = 20
num_classes = 5
learn_rate = 0.0004
dir_root = os.path.dirname(__file__)
data_root = os.path.join(dir_root, "..", "data", data_name)
class_indices = os.path.join(data_root, "class_indices.json")
save_path = os.path.join(dir_root, net_name + '_' + data_name + ".pth")
