import os, torch
import transforms
from network_files.faster_rcnn_framework import FasterRCNN, FasterRCNNPredictor
from backbone.resnet50_fon_model import resnet50_fpn_backbone
from my_dataset import VOC2012DataSet
from train_utils import train_eval_utils as utils

def create_model(num_classes):
    backbone = resnet50_fpn_backbone()
    #! 这里是加载预训练权重的模型，所以是91
    model = FasterRCNN(backbone=backbone, num_classes=91)
    weight_dict = torch.load(os.path.join(os.path.dirname(__file__), "backbone", "fasterrcnn_resnet50_fpn.pth"))
    missing_keys, unexpected_keys = model.load_state_dict(weight_dict, strict=False)
    if len(missing_keys) != 0 or len(unexpected_keys) != 0:
        print("missing_keys: ", missing_keys)
        print("unexpected_keys: ", unexpected_keys)

    in_features = model.roi_heads.box_predictor.cls_scores.in_features
    model.roi_heads.box_predictor = FasterRCNNPredictor(in_features, num_classes)

    return model


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_transform = {
        "train": transforms.Compose([transforms.ToTensor(),
                                    transforms.RandomHorizontalFlip(0.5)]),
        "val": transforms.Compose([transforms.ToTensor()])
    }

    VOC_root = args.data_path
    train_data_set = VOC2012DataSet(VOC_root, data_transform["train"], True)
    train_data_loader = torch.utils.data.DataLoader(train_data_set, batch_size=batch_size,
                                                shuffle=True, num_workers=0,
                                                collate_fn=utils.collate_fn)
    val_data_set = VOC2012DataSet(VOC_root, data_transform["val"], False)
    val_data_loader = torch.utils.data.DataLoader(val_data_set, batch_size=1,
                                                shuffle=False, num_workers=0,
                                                collate_fn=utils.collate_fn)
    #! 20种物体加上背景
    model = create_model(num_classes=21).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,, step_size=5, gamma=0.33)

    if args.resume != "":
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        print("The training resume from {} epoch".format(args.start_epoch))

    for epoch in range(args.start_epoch, args.num_epochs):
        utils.train_one_epoch(model, optimizer, train_data_loader, device, epoch, print_freq=50, warm_up=True)
        lr_scheduler.step()
        utils.evaluate(model, val_data_loader, device=device)
        save_file = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch,
        }
        torch.save(save_file, os.path.join(args.save_dir, "resNetFpn_model_{}.pth".format(epoch)))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch FasterRCNN Training')
    parser.add_argument('--batch_size', default=4, type=int, help='batch_size')
    parser.add_argument('--lr', default=0.005, type=float, help='learning rate')
    parser.add_argument('--num_epochs', default=20, type=int, help='number of epochs')
    parser.add_argument('--data_path', default=os.path.join(os.path.dirname(__file__), "..", "data", "VOC2012"), type=str, help='dataset path')
    parser.add_argument('--save_dir', default=os.path.join(os.path.dirname(__file__), "save_weights"), type=str, help='weights save path')
    parser.add_argument('--resume', default="", type=str, help='resume form checkpoint')
    parser.add_argument("--start_epoch", default=0, type=int, help="start epoch")

    args = parser.parse_args()
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    main(args)
