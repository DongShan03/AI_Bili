import torch, sys, os, json, math
from tqdm import tqdm
from torchvision import transforms, datasets
from cfg import cfg
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from ConfusionMatrix import ConfusionMatrix
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



def train_one_epoch(model, optimizer, data_loader, device, epoch, lr_scheduler):
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

        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}, lr: {:.5f}".format(epoch+1,
                                                                            accu_loss.item() / (step + 1),
                                                                            accu_num.item() / sample_num,
                                                                            optimizer.param_groups[0]['lr'])

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()
        lr_scheduler.step()

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

def get_params_groups(model, weight_decay=1e-5):
    parameter_group_vars = {
        "decay": {
            "params": [],
            "weight_decay": weight_decay,
        },
        "no_decay": {
            "params": [],
            "weight_decay": 0.0,
        }
    }
    parameter_group_names = {
        "decay": {
            "params": [],
            "weight_decay": weight_decay,
        },
        "no_decay": {
            "params": [],
            "weight_decay": 0.0,
        }
    }
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or name.endswith(".bias"):
            group_name = "no_decay"
        else:
            group_name = "decay"
        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)
    # print("Param groups = %s" % json.dumps(parameter_group_names, indent=2))
    return list(parameter_group_vars.values())

def create_lr_scheduler(optimizer, num_step, epochs, warmup=True,
                        warmup_epochs=1, warmup_factor=1e-3, end_factor=1e-6):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        if warmup and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            return warmup_factor * (1 - alpha) + alpha
        else:
            current_step = (x - warmup_epochs * num_step)
            cosine_steps = (epochs - warmup_epochs) * num_step
            return ((1 + math.cos(current_step * math.pi / cosine_steps)) / 2) * (1 - end_factor) + end_factor
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)

preprocess = Preprocess()
def plot_final():
    json_file = open(cfg["class_indices"], 'r')
    class_indict = json.load(json_file)
    labels = [label for _, label in class_indict.items()]
    confusion = ConfusionMatrix(num_classes=cfg["num_classes"], labels=labels)
    cfg["net"].eval()
    data_loader = tqdm(preprocess.validate_loader, file=sys.stdout, colour="green")
    with torch.no_grad():
        for step, val_data in enumerate(data_loader):
            val_images, val_labels = val_data
            outputs = cfg["net"](val_images.to(cfg["device"]))
            outputs = torch.softmax(outputs, dim=1)
            outputs = torch.argmax(outputs, dim=1)
            confusion.update(outputs.to("cpu").numpy(), val_labels.to("cpu").numpy())
            data_loader.desc = "Confusion Matrix evaluate"
    confusion.plot(save_path=os.path.join(cfg["dir_root"], "confusion_matrix.png"))
    confusion.summary()
