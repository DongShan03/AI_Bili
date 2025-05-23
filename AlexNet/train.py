from torchvision import transforms, datasets, utils
import torch.optim as optim
import torch.nn as nn
from model import AlexNet
import os, json, time, torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
data_transform = {
    "train": transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]),
    "val": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]),
}

dir_root = os.path.dirname(__file__)
image_path = os.path.join(dir_root, ".." ,"data", "flower_data")


train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                    transform=data_transform['train'])

validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                    transform=data_transform['val'])

train_num = len(train_dataset)
validate_num = len(validate_dataset)
flower_list = train_dataset.class_to_idx
class_indict = dict((val, key) for key, val in flower_list.items())
json_str = json.dumps(class_indict, indent=4)
with open(os.path.join(dir_root, "class_indices.json"), "w") as json_file:
    json_file.write(json_str)

batch_size = 32
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size,
    shuffle=True, num_workers=0
)
validate_loader = torch.utils.data.DataLoader(
    validate_dataset, batch_size=batch_size,
    shuffle=True, num_workers=0
)

net = AlexNet(num_classes=5, init_weights=True).to(device)
loss_function = nn.CrossEntropyLoss()
# pata = list(net.parameters())
optimizer = optim.Adam(net.parameters(), lr=0.0002)
save_path = os.path.join(dir_root, "AlexNet.pth")
best_acc = 0.0

for epoch in range(20):
    net.train()
    running_loss = 0.0
    t1 = time.perf_counter()
    for step, data in enumerate(train_loader, start=0):
        images, labels = data
        optimizer.zero_grad()
        outputs = net(images.to(device))
        loss = loss_function(outputs, labels.to(device))
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        rate = (step + 1) / len(train_loader)
        a = "*" * int(rate * 50)
        b = "." * int((1 - rate) * 50)
        print("\rtrain loss: {:^3.0f}%[{}->{}]{:.3f}".format(int(rate*100), a, b, loss), end="")
    print()             #* print中\r表示回到行首
    print("Train time: %.3f" % (time.perf_counter() - t1))

    net.eval()
    acc = 0.0
    with torch.no_grad():
        for data_test in validate_loader:
            test_images, test_labels = data_test
            outputs = net(test_images.to(device))
            predict_y = torch.max(outputs, 1)[1]
            acc += (predict_y == test_labels.to(device)).sum().item()
        accurate_test = acc / validate_num
        if accurate_test > best_acc:
            best_acc = accurate_test
            torch.save(net.state_dict(), save_path)
        print('[epoch %d] train_loss: %.3f  test_accuracy: %.3f' %
            (epoch + 1, running_loss / step, acc / validate_num))

print("Finished Training!")
