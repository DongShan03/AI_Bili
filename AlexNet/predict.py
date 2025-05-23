from PIL import Image
from model import AlexNet
from torchvision import transforms
import matplotlib.pyplot as plt
import json, os
import torch

data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

dir_path = os.path.dirname(__file__)
img_path = os.path.join(dir_path, "test.jpg")
img = Image.open(img_path)
img = data_transform(img)
img = torch.unsqueeze(img, dim=0)

try:
    json_file = open(os.path.join(dir_path, "class_indices.json"), "r")
    class_indict = json.load(json_file)
except Exception as e:
    print(e)
    exit(-1)

model = AlexNet(num_classes=5)
model_weight_path = os.path.join(dir_path, "AlexNet.pth")
model.load_state_dict(torch.load(model_weight_path))
model.eval()
with torch.no_grad():
    output = torch.squeeze(model(img))
    predict = torch.softmax(output, dim=0)
    predict_cla = torch.argmax(predict).numpy()
print("This pic should be: {}, and the probability is {:.3f}".format(class_indict[str(predict_cla)], predict[predict_cla].item()))
