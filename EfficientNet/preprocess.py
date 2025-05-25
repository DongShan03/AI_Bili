from torchvision import transforms, datasets, utils
from cfg import cfg
import os, json, torch

def get_data_loader():
    data_transform = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(cfg["img_size"]),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]),
        "val": transforms.Compose([
            transforms.Resize(cfg["img_size"]+32),
            transforms.CenterCrop(cfg["img_size"]),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]),
    }

    train_dataset = datasets.ImageFolder(root=os.path.join(cfg["data_root"], "train"),
                                        transform=data_transform['train'])

    validate_dataset = datasets.ImageFolder(root=os.path.join(cfg["data_root"], "val"),
                                        transform=data_transform['val'])

    train_num = len(train_dataset)
    validate_num = len(validate_dataset)
    if not os.path.exists(cfg["class_indices"]):
        flower_list = train_dataset.class_to_idx
        class_indict = dict((val, key) for key, val in flower_list.items())
        json_str = json.dumps(class_indict, indent=4)
        with open(cfg["class_indices"], "w") as json_file:
            json_file.write(json_str)
    else:
        print("class_indices.json文件已存在!")

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg["batch_size"],
        shuffle=True, num_workers=0
    )
    validate_loader = torch.utils.data.DataLoader(
        validate_dataset, batch_size=cfg["batch_size"],
        shuffle=True, num_workers=0
    )
    return train_num, validate_num, train_loader, validate_loader
