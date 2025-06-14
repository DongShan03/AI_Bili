import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import json, random

import torch
import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

from YOLOv4.utils import utils
from YOLOv4.utils.datasets import letterbox
from YOLOv4.model.model import Darknet
from YOLOv4.utils.draw_box_utils import draw_objs
from YOLOv4.opt import opt


def predict(img_o, input_size, model, device, category_index, logging_info=True):
    img = letterbox(img_o, new_shape=input_size, auto=True)[0]

    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).to(device).float()
    img /= 255.0
    img = img.unsqueeze(0)

    t1 = utils.time_synchronized()
    pred = model(img)[0]
    t2 = utils.time_synchronized()
    if logging_info:
        print("inference time: {:.4f}s".format(t2 - t1))

    pred = utils.non_max_suppression(pred, conf_thres=0.1, iou_thres=0.6, multi_label=True)[0]
    t3 = utils.time_synchronized()
    if logging_info:
        print("nms time: {:.4f}s".format(t3 - t2))

    if pred is None:
        return None

    pred[:, :4] = utils.scale_coords(img.shape[2:], pred[:, :4], img_o.shape).round()
    bboxes = pred[:, :4].detach().cpu().numpy()
    scores = pred[:, 4].detach().cpu().numpy()
    classes = pred[:, 5].detach().cpu().numpy().astype(np.int16) + 1

    pil_img = Image.fromarray(img_o[:, :, ::-1])
    plot_img = draw_objs(pil_img, bboxes, classes, scores,
                        category_index=category_index,
                        box_thresh=0.2, line_thickness=3,
                        font='arial.ttf', font_size=20)
    return plot_img


def main():
    img_size = opt.img_size
    weights_path = opt.weights

    json_path = os.path.join(opt.data_root, "pascal_voc_classes.json")
    json_file = open(json_path, 'r')
    class_dict = json.load(json_file)
    json_file.close()
    category_index = {str(v): str(k) for k, v in class_dict.items()}

    src_path = os.path.join(opt.data_root, "val", "images")
    imgs_name = os.listdir(src_path)
    img_name = random.choice(imgs_name)
    dst_path = os.path.join(opt.data_root, "test")
    src = os.path.join(src_path, img_name)
    result = os.path.join(dst_path, img_name.split(".")[0] + "_" + opt.save_name + "_result.jpg")

    input_size = (img_size, img_size)
    device = opt.device
    model = Darknet(cfg=opt.cfg, img_size=input_size)
    weights_dict = torch.load(weights_path, map_location='cpu')
    weights_dict = weights_dict["model"] if "model" in weights_dict else weights_dict
    model.load_state_dict(weights_dict)
    model.to(device)

    model.eval()
    with torch.no_grad():
        img = torch.zeros((1, 3, img_size, img_size), device=device)
        model(img)

        img_o = cv2.imread(src)
        assert img_o is not None, "Image Not Found " + src
        plot_img = predict(img_o, input_size, model, device, category_index)

        if plot_img is None:
            print("No target detected.")
            exit(0)
        plt.imshow(plot_img)
        plt.show()
        plot_img.save(result)

if __name__ == "__main__":
    main()
