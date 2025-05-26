from torchvision import transforms, datasets
from cfg import cfg
import os, json, torch

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

        self.train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=cfg["batch_size"],
            shuffle=True, num_workers=0
        )
        self.validate_loader = torch.utils.data.DataLoader(
            validate_dataset, batch_size=cfg["batch_size"],
            shuffle=True, num_workers=0
        )

preprocess = Preprocess()
