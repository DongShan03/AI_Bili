import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import torch
from VIT.model import vit_base_patch16_224 as Net
from lightning.fabric import Fabric

cfg = {
    "batch_size": 16,
    "net_name": getattr(Net, "__name__"),
    "data_name": "flower_data",
    "epochs": 5,
    "num_classes": 5,
    "learn_rate": 0.0002,
    "transfer_learning": True,           #! 是否使用(迁移)学习
    "train_size": Net.train_size,
    "eval_size": Net.eval_size,
    "dir_root": os.path.dirname(__file__),
    "fabric": Fabric(
        accelerator="auto",  # 自动检测硬件
        devices="auto",      # 使用所有可用设备
        strategy="ddp",      # 分布式数据并行
        precision="16-mixed"  # 可选混合精度训练
    ),
}

cfg.update({"data_root": os.path.join(cfg['dir_root'], "..", "data", cfg["data_name"]),
            "save_path": os.path.join(cfg['dir_root'], "model_trained", cfg['net_name'] + '_' + cfg['data_name'] + ".pth"),
            "structure": os.path.join(cfg['dir_root'], "model_structure", cfg['net_name'] + "_structure" + '.txt'),
            "net": Net(num_classes=cfg['num_classes']),
            })

cfg.update({'class_indices': os.path.join(cfg['data_root'], "class_indices.json")})
cfg["fabric"].launch()
