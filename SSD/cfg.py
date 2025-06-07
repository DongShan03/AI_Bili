import os, sys
sys.path.append(os.path.dirname(__file__))
from transforms import get_data_transform
import torch
from backbone.ssd_model import backBone, SSD300
from my_dataset import VOCDataset

cfg = {
    "epochs": 50,
    "batch_size": 24,
    "lr": 0.0005,
    "num_classes": 20,
    "transform": get_data_transform(),
    "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    "data_name": "VOC2012",
    "file_dir": os.path.dirname(__file__),
    "save_name": "ssd300",
}

cfg.update({
    "data_root": os.path.join(cfg["file_dir"], "..", "data", cfg["data_name"]),
    "save_path": os.path.join(cfg["file_dir"], "save_weights"),
    "model": SSD300(backBone(), num_classes=cfg["num_classes"]+1),

    #! resume 所指的模型仅用于继续训练,自动选择已保存的最新模型
    "resume": "",
    #! start_epoch不需要更改,如果启用了resume,start_epoch会自动调整到对应的值
    "start_epoch": 0,
    #! weights指的是本地训练后权重文件,这个文件会用于模型预测和计算mAP,未指定模型的时候与resume相同
    "weights": "",
})

def cfg_update():
    nums = []
    #! 遍历模型文件夹
    files = os.listdir(cfg["save_path"])
    for file in files:
        if file.endswith(".pth") & file.startswith(cfg["save_name"]):
            nums.append(int(file.split('-')[1].split('.')[0]))

    #! 如果没有训练过，resume和weights都为空
    #! 这里默认resume和weights都使用save_path下最新的模型
    if (len(nums) == 0):
        cfg["resume"] = cfg["weights"] = ""
    else:
        #! 取模型编号最大的模型用于继续训练或者预测
        num = sorted(nums, reverse=True)[0]
        #! 启动一次至少训练10个epoch
        if (cfg["epochs"] < num+1+10):
            cfg["epochs"] = num+1+10
        cfg["resume"] = os.path.join(cfg["save_path"], cfg["save_name"] + f"-{num}.pth")
        #! 如果resume为空，那么weights就等于resume
        if cfg["weights"] == "":
            cfg["weights"] = cfg["resume"]

cfg_update()

if __name__ == "__main__":
    print(cfg["weights"])
