import os, json
from pycocotools.coco import COCO
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

data_root = os.path.join(os.path.dirname(__file__), "COCO2017")
file_name = "instances_val2017.json"
json_path = os.path.join(data_root, "annotations" , file_name)
# load coco data
coco = COCO(annotation_file=json_path)

# get all coco class labels
coco_classes = dict([(v["id"], v["name"]) for k, v in coco.cats.items()])
with open(os.path.join(data_root, "coco91_indices.json"), 'w', encoding='utf-8') as f:
    json.dump(coco_classes, f)
