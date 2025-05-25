import os, torch
#! 在这更换要使用的网络
from model import efficientnet_b1 as Net

cfg = {
    "batch_size": 32,
    "net_name": getattr(Net, "__name__"),
    "data_name": "flower_data",
    "epochs": 50,
    "num_classes": 5,
    "learn_rate": 0.0002,
    "img_size": Net.img_size,
    "dir_root": os.path.dirname(__file__),
    "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
}

cfg.update({"data_root": os.path.join(cfg['dir_root'], "..", "data", cfg["data_name"]),
            "save_path": os.path.join(cfg['dir_root'], cfg['net_name'] + '_' + cfg['data_name'] + ".pth"),
            "net": Net(num_classes=cfg['num_classes']).to(cfg["device"]),
            "structure": os.path.join(cfg['dir_root'], cfg['net_name'] + "_structure" + '.txt')})

cfg.update({'class_indices': os.path.join(cfg['data_root'], "class_indices.json")})
