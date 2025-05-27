import torch, sys, os, json
from tqdm import tqdm
from torchvision import transforms, datasets
from cfg import cfg

class Preprocess:
    def __init__(self):
        super().__init__()
        self.data_transform = {
            "train": transforms.Compose([
                transforms.RandomResizedCrop(cfg["train_size"]),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]),
            "val": transforms.Compose([
                transforms.Resize(int(cfg["eval_size"]*1.143)),
                transforms.CenterCrop(cfg["eval_size"]),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]),
        }

        train_dataset = datasets.ImageFolder(root=os.path.join(cfg["data_root"], "train"),
                                            transform=self.data_transform['train'])

        validate_dataset = datasets.ImageFolder(root=os.path.join(cfg["data_root"], "val"),
                                            transform=self.data_transform['val'])

        if not os.path.exists(cfg["class_indices"]):
            flower_list = train_dataset.class_to_idx
            class_indict = dict((val, key) for key, val in flower_list.items())
            json_str = json.dumps(class_indict, indent=4)
            with open(cfg["class_indices"], "w") as json_file:
                json_file.write(json_str)
        else:
            print("class_indices.json文件已存在!")

        self.train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=cfg["batch_size"],
            shuffle=True, num_workers=0
        )
        self.validate_loader = torch.utils.data.DataLoader(
            validate_dataset, batch_size=cfg["batch_size"],
            shuffle=True, num_workers=0
        )



def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    optimizer.zero_grad()

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        loss.backward()
        accu_loss += loss.detach()

        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch+1,
                                                                            accu_loss.item() / (step + 1),
                                                                            accu_num.item() / sample_num)

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


@torch.no_grad()
def evaluate(model, data_loader, device, epoch):
    loss_function = torch.nn.CrossEntropyLoss()

    model.eval()

    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    accu_loss = torch.zeros(1).to(device)  # 累计损失

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout, colour="red")
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        accu_loss += loss

        data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch+1,
                                                                            accu_loss.item() / (step + 1),
                                                                            accu_num.item() / sample_num)

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


preprocess = Preprocess()
