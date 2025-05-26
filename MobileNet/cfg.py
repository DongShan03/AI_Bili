import os, torch
from modelV3 import mobilenet_v3_large as Net

cfg = {
    "batch_size": 32,
    "net_name": getattr(Net, "__name__"),
    "data_name": "flower_data",
    "epochs": 20,
    "num_classes": 5,
    "learn_rate": 0.0002,
    "dir_root": os.path.dirname(__file__),
    "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
}

cfg.update({"data_root": os.path.join(cfg['dir_root'], "..", "data", cfg["data_name"]),
            "save_path": os.path.join(cfg['dir_root'], cfg['net_name'] + '_' + cfg['data_name'] + ".pth"),
            "net": Net(num_classes=cfg['num_classes']).to(cfg["device"])})

cfg.update({'class_indices': os.path.join(cfg['data_root'], "class_indices.json")})
