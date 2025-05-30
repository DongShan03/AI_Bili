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
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]),
            "val": transforms.Compose([
                transforms.Resize(cfg["eval_size"]+32),
                transforms.CenterCrop(cfg["eval_size"]),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
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

        self.train_loader = cfg["fabric"].setup_dataloaders(
            torch.utils.data.DataLoader(
                train_dataset, batch_size=cfg["batch_size"],
                shuffle=True, num_workers=0
        ))
        self.validate_loader = cfg["fabric"].setup_dataloaders(
            torch.utils.data.DataLoader(
                validate_dataset, batch_size=cfg["batch_size"],
                shuffle=True, num_workers=0
        ))



def train_one_epoch(model, fabric, optimizer, data_loader, epoch):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    accu_loss = fabric.to_device(torch.zeros(1))  # 累计损失
    accu_num = fabric.to_device(torch.zeros(1))   # 累计预测正确的样本数
    optimizer.zero_grad()

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        pred = model(fabric.to_device(images))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, fabric.to_device(labels)).sum()

        loss = loss_function(pred, fabric.to_device(labels))
        fabric.backward(loss)
        accu_loss += loss.detach()

        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                            accu_loss.item() / (step + 1),
                                                                            accu_num.item() / sample_num)

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


@torch.no_grad()
def evaluate(model, fabric, data_loader, epoch):
    loss_function = torch.nn.CrossEntropyLoss()

    model.eval()

    accu_num = fabric.to_device(torch.zeros(1))   # 累计预测正确的样本数
    accu_loss = fabric.to_device(torch.zeros(1))  # 累计损失

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout, colour="red")
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        pred = model(fabric.to_device(images))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, fabric.to_device(labels)).sum()

        loss = loss_function(pred, fabric.to_device(labels))
        accu_loss += loss

        data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                            accu_loss.item() / (step + 1),
                                                                            accu_num.item() / sample_num)

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


preprocess = Preprocess()
