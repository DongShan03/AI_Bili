import os, torch, sys, math
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import transforms
from RetinaNet.network_files.retinanet import RetinaNet
from RetinaNet.backbone.resnet50_fpn_model import resnet50_fpn_backbone
from RetinaNet.backbone.feature_pyramid_network import LastLevelP6P7
from RetinaNet.cfg import cfg
from my_dataset import VOC2012DataSet
from train_utils import train_eval_utils as utils
from tensorboardX import SummaryWriter


def create_model(num_classes, load_pretrain_weights=True):
    weight_dir = os.path.join(os.path.dirname(__file__), "save_weights")
    # 注意，这里的backbone默认使用的是FrozenBatchNorm2d，即不会去更新bn参数
    # 目的是为了防止batch_size太小导致效果更差(如果显存很小，建议使用默认的FrozenBatchNorm2d)
    # 如果GPU显存很大可以设置比较大的batch_size就可以将norm_layer设置为普通的BatchNorm2d
    # trainable_layers包括['layer4', 'layer3', 'layer2', 'layer1', 'conv1']， 5代表全部训练
    # resnet50 imagenet weights url: https://download.pytorch.org/models/resnet50-0676ba61.pth
    backbone = resnet50_fpn_backbone(pretrain_path=os.path.join(weight_dir, "resnet50.pth"),
                                    returned_layers=[2, 3, 4],
                                    extra_blocks=LastLevelP6P7(256, 256),
                                    trainable_layers=3)
    # 训练自己数据集时不要修改这里的91，修改的是传入的num_classes参数
    model = RetinaNet(backbone=backbone, num_classes=91)

    # 载入预训练权重
    # https://download.pytorch.org/models/retinanet_resnet50_fpn_coco-eeacb38b.pth
    weights_dict = torch.load(os.path.join(weight_dir, "retinanet_resnet50_fpn--1.pt"), map_location='cpu')
    # 删除分类器部分的权重，因为自己的数据集类别与预训练数据集类别(91)不一定致，如果载入会出现冲突
    del_keys = ["head.classification_head.cls_logits.weight", "head.classification_head.cls_logits.bias"]
    for k in del_keys:
        del weights_dict[k]

    print(model.load_state_dict(weights_dict, strict=False))

    return model


def main():
    device = cfg["device"]
    epochs = cfg["epochs"]
    data_transform = {
        "train": transforms.Compose([transforms.ToTensor(),
                                    transforms.RandomHorizontalFlip(0.5),
                                    transforms.RandomVerticalFlip(0.3)]),
        "val": transforms.Compose([transforms.ToTensor()])
    }

    VOC_root = cfg["data_root"]
    batch_size = cfg["batch_size"]
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using %g dataloader workers' % nw)
    train_data_set = VOC2012DataSet(VOC_root, data_transform["train"], True)
    train_data_loader = torch.utils.data.DataLoader(train_data_set, batch_size=batch_size,
                                                shuffle=True, num_workers=nw,
                                                collate_fn=utils.collate_fn)
    val_data_set = VOC2012DataSet(VOC_root, data_transform["val"], False)
    val_data_loader = torch.utils.data.DataLoader(val_data_set, batch_size=1,
                                                shuffle=False, num_workers=nw,
                                                collate_fn=utils.collate_fn)
    #! 20种物体加上背景
    model = create_model(num_classes=cfg["num_classes"]).to(device)

    start_epoch = 0
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=cfg["lr"], momentum=0.9, weight_decay=0.0005)

    if cfg["resume"] != "":
        checkpoint = torch.load(cfg["resume"], map_location=device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        print("The training resume from {} epoch".format(start_epoch))

    lf = lambda x: ((1 + math.cos(x * math.pi / epochs)) / 2) * (1 - 0.01) + 0.01  # cosine
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    lr_scheduler.last_epoch = start_epoch

    for epoch in range(start_epoch, epochs):
        utils.train_one_epoch(model, optimizer, train_data_loader, device, epoch, print_freq=50, warmup=True)
        lr_scheduler.step()
        utils.evaluate(model, val_data_loader, device=device)
        save_file = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch,
        }
        torch.save(save_file, os.path.join(cfg['save_dir'], cfg["save_name"] + "-{}.pth".format(epoch)))

if __name__ == "__main__":
    main()
