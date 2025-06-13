import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import json, random
from tqdm import tqdm
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
from YOLOv4.utils.datasets import LoadStreams
from YOLOv4.predict import predict


def main():
    img_size = opt.img_size
    weights_path = opt.weights

    json_path = os.path.join(opt.data_root, "pascal_voc_classes.json")
    json_file = open(json_path, 'r')
    class_dict = json.load(json_file)
    json_file.close()
    category_index = {str(v): str(k) for k, v in class_dict.items()}

    video_path = os.path.join(opt.file_dir, "video")
    video_names = sorted(os.listdir(video_path))
    all_videos = []
    for name in video_names:
        if "out" not in name:
            all_videos.append(os.path.join(video_path, name))
        if "out" in name:
            all_videos.remove(os.path.join(video_path, name.replace(".out", "")))

    if len(all_videos) == 0:
        print("No video to detect")
        return

    input_size = (img_size, img_size)
    device = opt.device
    model = Darknet(cfg=opt.cfg, img_size=input_size)
    weights_dict = torch.load(weights_path, map_location='cpu')
    weights_dict = weights_dict["model"] if "model" in weights_dict else weights_dict
    model.load_state_dict(weights_dict)
    model.to(device)

    model.eval()
    # plt.ion()
    with torch.no_grad():
        for path in all_videos:
            cap = cv2.VideoCapture(path)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            out = cv2.VideoWriter(path.replace(path.split(".")[-1], "out." + path.split(".")[-1]),\
                                cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            for _ in tqdm(range(num_frames), desc="Detecting Video:" + path.split(os.sep)[-1]):
                _, frame = cap.read()
                if frame is None:
                    break
                plot_img = predict(frame, input_size=input_size, model=model, device=device, category_index=category_index, logging_info=False)
                if plot_img is not None:
                    out.write(np.array(plot_img)[:, :, ::-1])
                else:
                    out.write(frame)

                # plt.imshow(plot_img)
                # plt.pause(0.05)
                # plt.clf()
            cap.release()
            out.release()
if __name__ == "__main__":
    main()
