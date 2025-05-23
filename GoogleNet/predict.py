from PIL import Image
from model import GoogleNet
from torchvision import transforms
import matplotlib.pyplot as plt
import json, os
import torch
from cfg import *

def get_pic():
    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    img_path = os.path.join(dir_root, "test.jpg")
    img = Image.open(img_path)
    img = data_transform(img)
    img = torch.unsqueeze(img, dim=0)
    return img

def main():
    try:
        json_file = open(class_indices, "r")
        class_indict = json.load(json_file)
    except Exception as e:
        print(e)
        exit(-1)
    model = GoogleNet(num_classes=num_classes, aux_logits=True, init_weights=False)
    model.load_state_dict(torch.load(save_path))
    model.eval()
    with torch.no_grad():
        output = torch.squeeze(model(get_pic()))
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()
    print("This pic should be: {}, and the probability is {:.3f}".format(class_indict[str(predict_cla)], predict[predict_cla].item()))

if __name__ == "__main__":
    main()
