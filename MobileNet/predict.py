from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import json, os
import torch
from cfg import *

def get_pic():
    data_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    img_path = os.path.join(dir_root, "test1.jpg")
    img = Image.open(img_path)
    img = data_transform(img)
    img = torch.unsqueeze(img, dim=0)
    return img

def main():
    try:
        json_file = open(class_indices, "r")
        class_indict = json.load(json_file)
    except Exception as e:
        raise FileExistsError("{} is not exist".format(class_indices))
    net.load_state_dict(torch.load(save_path))
    net.eval()
    with torch.no_grad():
        output = torch.squeeze(net(get_pic()))
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()
    print("This pic should be: {}, and the probability is {:.3f}".format(class_indict[str(predict_cla)], predict[predict_cla].item()))

if __name__ == "__main__":
    main()
