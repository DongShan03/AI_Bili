import os
import numpy as np


def parse_model_cfg(path: str):
    if not path.endswith(".cfg") or not os.path.exists(path):
        raise Exception("Invalid config file path: {}".format(path))

    with open(path, "r", encoding="utf-8") as f:
        lines = f.read().split("\n")

    lines = [line for line in lines if line and not line.startswith("#")]
    lines = [line.strip() for line in lines]

    model_defs = []
    for line in lines:
        if line.startswith("["):
            model_defs.append({})
            #* 记录module类型
            model_defs[-1]["type"] = line[1:-1].strip()
            if model_defs[-1]["type"] == "convolutional":
                model_defs[-1]["batch_normalize"] = 0
        else:
            key, val = line.split("=")
            key = key.strip()
            val = val.strip()

            if key == "anchors":
                val = val.replace(" ", "")
                model_defs[-1][key] = np.array([float(x) for x in val.split(",")]).reshape((-1, 2))
            elif (key in ["from", "layers", "mask"]) or (key == "size" and "." in val):
                model_defs[-1][key] = [int(x) for x in val.split(",")]
            else:
                if val.isnumeric():
                    model_defs[-1][key] = int(val) if (int(val) - float(val) == 0) else float(val)
                else:
                    model_defs[-1][key] = val

    # check all fields are supported
    supported = ['type', 'batch_normalize', 'filters', 'size', 'stride', 'pad', 'activation', 'layers', 'groups',
                'from', 'mask', 'anchors', 'classes', 'num', 'jitter', 'ignore_thresh', 'truth_thresh', 'random',
                'stride_x', 'stride_y', 'weights_type', 'weights_normalization', 'scale_x_y', 'beta_nms', 'nms_kind',
                'iou_loss', 'iou_normalizer', 'cls_normalizer', 'iou_thresh', 'probability']

    for x in model_defs[1:]:
        for k in x:
            assert k in supported, "Unrecognized field '{}' in line {}".format(k, x)

    return model_defs


def parse_data_cfg(path):
    # Parses the data configuration file
    if not os.path.exists(path) and os.path.exists('data' + os.sep + path):  # add data/ prefix if omitted
        path = 'data' + os.sep + path

    with open(path, 'r') as f:
        lines = f.readlines()

    options = dict()
    for line in lines:
        line = line.strip()
        if line == '' or line.startswith('#'):
            continue
        key, val = line.split('=')
        options[key.strip()] = val.strip()

    return options
