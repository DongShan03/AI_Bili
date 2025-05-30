from PIL import Image
import json, os
import torch
from utils import preprocess
from cfg import cfg
from model import repvgg_model_convert

imgs_path = []

def get_pic():
    data_transform = preprocess.data_transform["val"]
    imgs = []

    for img_path in os.listdir(os.path.join(cfg["data_root"], "test")):
        if img_path.endswith(".jpg"):
            imgs_path.append(os.path.join("test", img_path))
            img = Image.open(os.path.join(cfg["data_root"], "test", img_path))
            img = data_transform(img)
            imgs.append(img)

    batch_list = torch.stack(imgs, dim=0)
    return batch_list

def main():
    try:
        json_file = open(cfg["class_indices"], "r")
        class_indict = json.load(json_file)
    except Exception as e:
        print(e)
        exit(-1)
    if not os.path.exists(cfg["save_path"]):
        raise FileExistsError("The model file {} is not exist!!!".format(cfg["save_path"]))
    cfg["net"].load_state_dict(torch.load(cfg["save_path"]))
    cfg["net"] = repvgg_model_convert(cfg["net"], save_path=cfg["save_path_deploy"]).to(cfg["device"])
    cfg["net"].eval()
    with torch.no_grad():
        output = torch.squeeze(cfg["net"](get_pic().to(cfg["device"])))
        predict = torch.softmax(output, dim=1)
        probs, classes = torch.max(predict, dim=1)
        for idx, (prob, cla) in enumerate(zip(probs, classes)):
            print("num{} image: {}, class: {}, probability: {:.3f}" \
                    .format(idx+1, imgs_path[idx], class_indict[str(cla.cpu().numpy())], prob.item()))

if __name__ == "__main__":
    main()
