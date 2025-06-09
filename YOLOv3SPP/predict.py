import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import json, random

import torch
import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

from YOLOv3SPP.utils import utils
from YOLOv3SPP.utils.datasets import letterbox
from YOLOv3SPP.model.model import Yolov3SPP
from YOLOv3SPP.utils.draw_box_utils import draw_objs
from YOLOv3SPP.cfg import cfg


def main():
    img_size = cfg["img_size"]
    weights_path = cfg["weights"]

    json_path = os.path.join(cfg["data_root"], "pascal_voc_classes.json")
    json_file = open(json_path, 'r')
    class_dict = json.load(json_file)
    json_file.close()
    category_index = {str(v): str(k) for k, v in class_dict.items()}

    src_path = os.path.join(cfg["data_root"], "val", "images")
    imgs_name = os.listdir(src_path)
    img_name = random.choice(imgs_name)
    dst_path = os.path.join(cfg["data_root"], "test")
    src = os.path.join(src_path, img_name)
    result = os.path.join(dst_path, img_name.split(".")[0] + "_" + cfg["save_name"] + "_result.jpg")

    input_size = (img_size, img_size)
    device = cfg["device"]
    model = Yolov3SPP(img_size=input_size)
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
        img = letterbox(img_o, new_shape=input_size, auto=True, color=(0, 0, 0))[0]

        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(device).float()
        img /= 255.0
        img = img.unsqueeze(0)

        t1 = utils.time_synchronized()
        pred = model(img)[0]
        t2 = utils.time_synchronized()
        print("inference time: {:.4f}s".format(t2 - t1))

        pred = utils.non_max_suppression(pred, conf_thres=0.1, iou_thres=0.6, multi_label=True)[0]
        t3 = utils.time_synchronized()
        print("nms time: {:.4f}s".format(t3 - t2))

        if pred is None:
            print("No target detected.")
            exit(0)

        pred[:, :4] = utils.scale_coords(img.shape[2:], pred[:, :4], img_o.shape).round()
        bboxes = pred[:, :4].detach().cpu().numpy()
        scores = pred[:, 4].detach().cpu().numpy()
        classes = pred[:, 5].detach().cpu().numpy().astype(np.int16) + 1

        pil_img = Image.fromarray(img_o[:, :, ::-1])
        plot_img = draw_objs(pil_img, bboxes, classes, scores,
                            category_index=category_index,
                            box_thresh=0.2, line_thickness=3,
                            font='arial.ttf', font_size=20)

        plt.imshow(plot_img)
        plt.show()
        plot_img.save(result)

if __name__ == "__main__":
    main()
