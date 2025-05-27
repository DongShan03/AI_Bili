import os, torch
#! 在这更换要使用的网络
from model import convnext_small as Net

cfg = {
    "batch_size": 32,
    "net_name": getattr(Net, "__name__"),
    "data_name": "flower_data",
    "epochs": 20,
    "num_classes": 5,
    "learn_rate": 5e-4,
    "weight_decay": 5e-2,
    "transfer_learning": False,           #! 是否使用(迁移)学习
    "train_size": Net.train_size,
    "eval_size": Net.eval_size,
    "dir_root": os.path.dirname(__file__),
    "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
}

cfg.update({"data_root": os.path.join(cfg['dir_root'], "..", "data", cfg["data_name"]),
            "save_path": os.path.join(cfg['dir_root'], "model_trained", cfg['net_name'] + '_' + cfg['data_name'] + ".pth"),
            "structure": os.path.join(cfg['dir_root'], "model_structure", cfg['net_name'] + "_structure" + '.txt'),
            "net": Net(num_classes=cfg['num_classes']).to(cfg["device"]),
            })

cfg.update({'class_indices': os.path.join(cfg['data_root'], "class_indices.json")})
