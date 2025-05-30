import torch, os
import torchvision.transforms as transforms
from my_dataset import VOC2012DataSet
import torchvision


def create_model(num_classes, weights_path):
    backbone = MobileNetV2(weights_path=weights_path).features
    backbone.out_channels = 1280  # 设置对应backbone输出特征矩阵的channels

    anchor_generator = AnchorsGenerator(sizes=((32, 64, 128, 256, 512),),
                                        aspect_ratios=((0.5, 1.0, 2.0),))

    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],  # 在哪些特征层上进行roi pooling
                                                    output_size=[7, 7],   # roi_pooling输出特征矩阵尺寸
                                                    sampling_ratio=2)  # 采样率

    model = FasterRCNN(backbone=backbone,
                       num_classes=num_classes,
                       rpn_anchor_generator=anchor_generator,
                       box_roi_pool=roi_pooler)

    return model

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch_size = 4
    lr = 0.005
    num_epochs = 20
    print("Using {} device training.".format(device.type))

    weights_path = os.path.join(os.path.dirname(__file__), "backbone", "mobilenet_v2.pth")
    save_dir = os.path.join(os.path.dirname(__file__), "save_weights")

    if not os.path.exists("save_weights"):
        os.makedirs("save_weights")

    data_transform = {
        "train": transforms.Compose([transforms.ToTensor(),
                                    transforms.RandomHorizontalFlip(0.5),
                                    transforms.RandomVerticalFlip(0.3)]),
        "val": transforms.Compose([transforms.ToTensor()])
    }

    VOC_root = os.path.join(os.path.dirname(__file__), "..", "data", "VOC2012")
    train_data_set = VOC2012DataSet(VOC_root, data_transform["train"], True)
    train_data_loader = torch.utils.data.DataLoader(train_data_set, batch_size=batch_size,
                                                shuffle=True, num_workers=0,
                                                collate_fn=utils.collate_fn)
    val_data_set = VOC2012DataSet(VOC_root, data_transform["val"], False)
    val_data_loader = torch.utils.data.DataLoader(val_data_set, batch_size=1,
                                                shuffle=False, num_workers=0,
                                                collate_fn=utils.collate_fn)
    #! 20种物体加上背景
    model = create_model(num_classes=21, weights_path=weights_path).to(device)

    for param in model.backbone.parameters():
        param.requires_grad = False

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=0.0005)
    num_epochs = 5
    for epoch in range(num_epochs):
        utils.train_one_epoch(model, optimizer, train_data_loader, device, epoch, print_freq=50)
        utils.evaluate(model, val_data_loader, device=device)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    torch.save(model.state_dict(), os.path.join(save_dir, "pretrain.pth"))

    #* 冻结部分参数
    for name, param in model.backbone.named_parameters():
        split_name = name.split(".")[0]
        if split_name in ["0", "1", "2", "3"]:
            param.requires_grad = False
        else:
            param.requires_grad = True

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=0.0005)
    #! 每5步降低到之前的0.33
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.33)
    for epoch in range(num_epochs):
        utils.train_one_epoch(model, optimizer, train_data_loader, device, epoch,
                            print_freq=50, warmup=True)
        lr_scheduler.step()
        utils.evaluate(model, val_data_loader, device=device)
        if epoch > 10:
            save_files = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "epoch": epoch
            }
            torch.save(save_files, os.path.join(save_dir, "mobile-model-{}.pth".format(epoch)))

if __name__ == "__main__":
    main()
