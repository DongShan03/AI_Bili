import os, sys
sys.path.append(os.path.dirname(__file__))
import time
import json

import torch
import torchvision
from PIL import Image
import matplotlib.pyplot as plt

from torchvision import transforms
from network_files.faster_rcnn_framework import FasterRCNN, FastRCNNPredictor, AnchorsGenerator
from backbone.resnet50_fpn_model import resnet50_fpn_backbone
from change_backbone_with_fpn import create_model
from draw_box_utils import draw_objs



def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def main():
    # get devices
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # create model
    model = create_model(num_classes=21)

    # load train weights
    weights_path = os.path.join(os.path.dirname(__file__), "save_weights", "resNetFpn-model-2.pth")
    assert os.path.exists(weights_path), "{} file dose not exist.".format(weights_path)
    weights_dict = torch.load(weights_path, map_location='cpu')
    weights_dict = weights_dict["model"] if "model" in weights_dict else weights_dict
    model.load_state_dict(weights_dict)
    model.to(device)

    # read class_indict
    label_json_path = os.path.join(os.path.dirname(__file__), "..", "data", "VOC2012", "pascal_voc_classes.json")
    assert os.path.exists(label_json_path), "json file {} dose not exist.".format(label_json_path)
    with open(label_json_path, 'r') as f:
        class_dict = json.load(f)

    category_index = {str(v): str(k) for k, v in class_dict.items()}

    # load image
    img_name = "test1"
    img_path = os.path.join(os.path.dirname(__file__), "..", "data", "VOC2012", "test", img_name + ".jpg")
    original_img = Image.open(img_path).convert("RGB")

    # from pil image to tensor, do not normalize image
    data_transform = transforms.Compose([transforms.ToTensor()])
    img = data_transform(original_img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    model.eval()  # 进入验证模式
    with torch.no_grad():
        # init
        img_height, img_width = img.shape[-2:]
        init_img = torch.zeros((1, 3, img_height, img_width), device=device)
        model(init_img)

        t_start = time_synchronized()
        predictions = model(img.to(device))[0]
        t_end = time_synchronized()
        print("inference+NMS time: {}".format(t_end - t_start))

        predict_boxes = predictions["boxes"].to("cpu").numpy()
        predict_classes = predictions["labels"].to("cpu").numpy()
        predict_scores = predictions["scores"].to("cpu").numpy()

        if len(predict_boxes) == 0:
            print("没有检测到任何目标!")
        plot_img = draw_objs(original_img,
                            predict_boxes,
                            predict_classes,
                            predict_scores,
                            category_index=category_index,
                            box_thresh=0.5,
                            line_thickness=3,
                            font='arial.ttf',
                            font_size=20)
        plt.imshow(plot_img)
        plt.show()
        # 保存预测的图片结果
        plot_img.save(os.path.join(os.path.dirname(__file__), "..", "data", "VOC2012", "test", img_name + "_result.jpg"))


if __name__ == '__main__':
    main()
