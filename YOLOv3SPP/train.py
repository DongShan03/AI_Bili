import os, sys, math
sys.path.append(os.path.dirname(__file__))
from cfg import cfg
import glob
from build_utils.parse_config import parse_data_cfg


def train():
    device = cfg["device"]
    print("Using {} device training.".format(device.type))
    if not os.path.exists(cfg["save_path"]):
        os.mkdir(cfg["save_path"])

    accumulate = max(round(64 / cfg["batch_size"]), 1)
    imgsz_train = cfg["img_size"]
    imgsz_test = cfg["img_size"]
    multi_scale = cfg["muliti_scale"]

    results_file = cfg["save_name"] + "_results.txt"
    results_file = os.path.join(cfg["save_path"], results_file)
    data_transform = cfg["transform"]
    YOLO_root = cfg["data_root"]
    #! 图像需要是32的倍数
    gs = 32
    assert math.fmod(cfg["img_size"], gs) == 0, "Image sizes must be a multiple of 32!"
    grid_min, grid_max = imgsz_test // gs, imgsz_test // gs
    if multi_scale:
        imgsz_min = cfg["img_size"] // 1.5
        imgsz_max = cfg["img_size"] // 0.667
        grid_min, grid_max = imgsz_min // gs, imgsz_max // gs
        imgsz_min, imgsz_max = int(grid_min * gs), int(grid_max * gs)
        imgsz_train = imgsz_max
        print("Using multi_scale training, imgsz range[{}, {}]".format(imgsz_min, imgsz_max))

    train_path = os.path.join(YOLO_root, "train")
    val_path = os.path.join(YOLO_root, "val")
    nc = 1 if cfg["single_cls"] else int(cfg["num_classes"])  # number of classes
    cfg["hyp"]["cls"] *= nc / 80  # update coco-tuned hyp['cls'] to current dataset
    cfg["hyp"]["obj"] *= imgsz_test / 320

    model = Darknet(cfg).to(device)
