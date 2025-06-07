import os, sys
sys.path.append(os.path.dirname(__file__))
from transforms import get_data_transform
import torch
from backbone.ssd_model import backBone, SSD300
from my_dataset import VOCDataset

cfg = {
    "epochs": 20,
    "batch_size": 16,
    "lr": 0.0005,
    "num_classes": 20,
    "transform": get_data_transform(),
    "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    "data_name": "VOC2012",
    "file_dir": os.path.dirname(__file__),
    #! resume 所指的模型仅用于继续训练
    "resume": "ssd300-3.pth",
    #! start_epoch不需要更改,如果启用了resume,start_epoch会自动调整到对应的值
    "start_epoch": 0,
}

cfg.update({
    "data_root": os.path.join(cfg["file_dir"], "..", "data", cfg["data_name"]),
    "save_path": os.path.join(cfg["file_dir"], "save_weights"),
    #! weights指的是本地训练后权重文件,这个文件会用于模型预测和计算mAP
    "weights": "ssd300-0.pth",
    "model": SSD300(backBone(), num_classes=cfg["num_classes"]+1),
})

cfg["resume"] = os.path.join(cfg["save_path"], cfg["resume"]) if cfg["resume"] != "" else ""
