import os
import numpy as np


def parse_model_cfg(cfg_name: str):
    if not cfg_name.endswith(".cfg"):
        cfg_name += ".cfg"
    cfg_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model_cfg", cfg_name)
    with open(cfg_path, "r") as f:
        lines = f.read().split("\n")
    lines = [x for x in lines if x and not x.startswith("#")]
    lines = [x.rstrip().lstrip() for x in lines]
    mdefs = []
    for line in lines:
        if line.startswith("["):
            mdefs.append({})
            mdefs[-1]["type"] = line[1:-1].rstrip()
            if mdefs[-1]["type"] == "convolutional":
                mdefs[-1]["batch_normalize"] = 0

        else:
            key, val = line.split("=")
            key = key.rstrip()

            if key == "anchors":
                mdefs[-1][key] = np.array([float(x) for x in val.split(",")]).reshape(-1, 2)
            elif (key in ["from", "layers", "mask"]) or (key == "size" and "," in val):
                mdefs[-1][key] = [int(x) for x in val.split(",")]
            else:
                val = val.strip()
                if val.isnumeric():
                    mdefs[-1][key] = int(val) if (int(val) - float(val) == 0) else float(val)
                else:
                    mdefs[-1][key] = val
    # Check all fields are supported
    supported = ['type', 'batch_normalize', 'filters', 'size', 'stride', 'pad', 'activation', 'layers', 'groups',
                'from', 'mask', 'anchors', 'classes', 'num', 'jitter', 'ignore_thresh', 'truth_thresh', 'random',
                'stride_x', 'stride_y', 'weights_type', 'weights_normalization', 'scale_x_y', 'beta_nms', 'nms_kind',
                'iou_loss', 'iou_normalizer', 'cls_normalizer', 'iou_thresh', 'atoms', 'na', 'nc']

    f = []
    for x in mdefs[1:]:
        [f.append(k) for k in x if k not in f]
    u = [x for x in f if x not in supported]  # unsupported fields

    assert not any(u)
    return mdefs
