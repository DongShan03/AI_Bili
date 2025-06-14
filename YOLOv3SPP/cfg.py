"""
整合所有参数
"""
import os, sys, re
sys.path.append(os.path.dirname(__file__))
import torch, yaml
#* 这个模型的学习率配置文件存放在net_cfg/hyp.yaml
cfg = {
    "epochs": 80,
    "batch_size": 24,
    "num_classes": 20,
    "data_name": "yolo_data_VOC2012",
    "save_name": "YOLOv3SPP",
}

cfg.update({
    "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    "file_dir": os.path.dirname(__file__),
    "muliti_scale": True,
    "freeze_layer": False,
    "img_size": 512,
    "single_cls": False,        #! 单类训练
    "amp": True,
    "rect": True,
    "cache_images": False,
    "notest": False,
    "savebest": False,
})

cfg.update({
    "data_root": os.path.join(cfg["file_dir"], "..", "data", cfg["data_name"]),
    "save_path": os.path.join(cfg["file_dir"], "save_weights"),

    #! resume 所指的模型仅用于继续训练,自动选择已保存的最新模型
    "resume": "",
    #! start_epoch不需要更改,如果启用了resume,start_epoch会自动调整到对应的值
    "start_epoch": 0,
    #! weights指的是本地训练后权重文件,这个文件会用于模型预测和计算mAP,未指定模型的时候与resume相同
    "weights": "",
})

def cfg_update():

    yaml_file = os.path.join(cfg["file_dir"], "net_cfg", "hyp.yaml")
    with open(yaml_file) as f:
        cfg.update({
            "hyp": yaml.load(f, Loader=yaml.FullLoader)
        })

    nums = []
    #! 遍历模型文件夹
    if not os.path.exists(cfg["save_path"]):
        os.mkdir(cfg["save_path"])
        nums = []
    else:
        files = os.listdir(cfg["save_path"])
        for file in files:
            result = re.findall(cfg['save_name'] + r"-(\d+).pth", file)
            if len(result) == 0:
                continue
            nums.append(int(result[0]))

    #! 如果没有训练过，resume和weights都为空
    #! 这里默认resume和weights都使用save_path下最新的模型
    if (len(nums) == 0):
        cfg["resume"] = cfg["weights"] = ""
        file_name = os.path.join(cfg["save_path"], cfg["save_name"] + "--1.pt")
        if os.path.exists(file_name):
            cfg["resume"] = file_name
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
    print(cfg["resume"])
