"""
本脚本有两个功能：
1.将voc数据集标注信息(.xml)转为yolo标注格式(.txt)，并将图像文件复制到相应文件夹
2.根据json标签文件，生成对应names标签(my_data_label.names)
from http://github.com/WZMIAOMIAO/deep-learning-for-image-processing/blob/master/pytorch_object_detection/yolov3_spp/trans_voc2yolo.py
"""

import os
from tqdm import tqdm
from lxml import etree
import json
import shutil

VOC_ROOT = os.path.join(os.path.dirname(__file__), "..", "data", "VOC2012")
save_file_root = os.path.join(os.path.dirname(__file__), "..", "data", "yolo_data_VOC2012")
label_json_path = os.path.join(VOC_ROOT, "pascal_voc_classes.json")
label_to_json_path = os.path.join(save_file_root, "pascal_voc_classes.json")

voc_images_path = os.path.join(VOC_ROOT, "JPEGImages")
voc_xml_path = os.path.join(VOC_ROOT, "Annotations")
train_txt_path = os.path.join(VOC_ROOT, "train.txt")
val_txt_path = os.path.join(VOC_ROOT, "val.txt")

assert os.path.exists(voc_images_path), "VOC images数据集不存在"
assert os.path.exists(voc_xml_path), "VOC xml数据集不存在"
assert os.path.exists(train_txt_path), "VOC train.txt不存在"
assert os.path.exists(val_txt_path), "VOC val.txt不存在"
if not os.path.exists(save_file_root):
    os.makedirs(save_file_root)

def parse_xml_to_dict(xml):
    if len(xml) == 0:
        return {xml.tag: xml.text}

    result = {}
    for child in xml:
        child_result = parse_xml_to_dict(child)
        if child.tag != "object":
            result[child.tag] = child_result[child.tag]
        else:
            if child.tag not in result:
                result[child.tag] = []
            result[child.tag].append(child_result[child.tag])
    return {xml.tag: result}


def translate_info(file_names: list, save_root: str, class_dict: dict, train_val="train"):
    save_txt_path = os.path.join(save_root, train_val, "labels")
    if not os.path.exists(save_txt_path):
        os.makedirs(save_txt_path)
    save_imgs_path = os.path.join(save_root, train_val, "images")
    if not os.path.exists(save_imgs_path):
        os.makedirs(save_imgs_path)

    for file in tqdm(file_names, desc="translate {} file...".format(train_val)):
        img_path = os.path.join(voc_images_path, file + ".jpg")
        assert os.path.exists(img_path), "{} file does not exist.".format(img_path)

        xml_path = os.path.join(voc_xml_path, file + ".xml")
        assert os.path.exists(xml_path), "{} file does not exist.".format(xml_path)

        with open(xml_path) as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str)
        data = parse_xml_to_dict(xml)["annotation"]
        img_height = int(data["size"]["height"])
        img_width = int(data["size"]["width"])

        assert "object" in data.keys(), "xml contains no object"

        if len(data["object"]) == 0:
            print("Warning: in '{}' xml, there are no objects.".format(xml_path))
            continue

        with open(os.path.join(save_txt_path, file + ".txt"), "w") as f:
            for index, obj in enumerate(data["object"]):
                xmin = float(obj["bndbox"]["xmin"])
                xmax = float(obj["bndbox"]["xmax"])
                ymin = float(obj["bndbox"]["ymin"])
                ymax = float(obj["bndbox"]["ymax"])
                class_name = obj["name"]
                #! 目标id从0开始
                class_index = class_dict[class_name] - 1

                if xmax <= xmin or ymax <= ymin:
                    print("Warning: in '{}' xml, there are some bbox w/h <=0".format(xml_path))
                    continue

                xc = xmin + (xmax - xmin) / 2
                yc = ymin + (ymax - ymin) / 2
                w = xmax - xmin
                h = ymax - ymin

                #! 转换为相对坐标
                xc = round(xc / img_width, 6)
                yc = round(yc / img_height, 6)
                w = round(w / img_width, 6)
                h = round(h / img_height, 6)

                info = [str(i) for i in [class_index, xc, yc, w, h]]
                if index == 0:
                    f.write(" ".join(info))
                else:
                    f.write("\n" + " ".join(info))

        path_copy_to = os.path.join(save_imgs_path, img_path.split(os.sep)[-1])
        if not os.path.exists(path_copy_to):
            shutil.copyfile(img_path, path_copy_to)


def create_class_names(class_dict: dict):
    keys = class_dict.keys()
    with open(os.path.join(save_file_root, "my_data_label.names"), "w") as w:
        for index, k in enumerate(keys):
            if index + 1 == len(keys):
                w.write(k)
            else:
                w.write(k + "\n")

def main():
    json_file = open(label_json_path, "r")
    class_dict = json.load(json_file)
    shutil.copyfile(label_json_path, label_to_json_path)

    with open(train_txt_path, "r") as r:
        train_file_names = [i for i in r.read().splitlines() if len(i.strip()) > 0]
    translate_info(train_file_names, save_file_root, class_dict, "train")

    with open(val_txt_path, "r") as r:
        val_file_names = [i for i in r.read().splitlines() if len(i.strip()) > 0]
    translate_info(val_file_names, save_file_root, class_dict, "val")

    create_class_names(class_dict)

if __name__ == "__main__":
    main()
