import os, sys
sys.path.append(os.path.dirname(__file__))
import datetime

import torch
import transforms
from network_files.faster_rcnn_framework import FasterRCNN, AnchorsGenerator
from my_dataset import VOC2012DataSet
from train_utils.group_by_aspect_ratio import GroupedBatchSampler, create_aspect_ratio_groups
from train_utils import train_eval_utils as utils
from backbone.feature_pyramid_network import BackboneWithFPN, LastLevelMaxPool


def get_channel_list(backbone):
    img = torch.randn(1, 3, 224, 224)
    outputs = backbone(img)
    # for k, v in outputs.items():
    #     print(f"{k} shape: {v.shape}" )
    # 提供给fpn每个特征层channel
    in_channel_list = [v.shape[1] for k, v in outputs.items()]
    return in_channel_list


def create_model(num_classes):
    import torchvision
    from torchvision.models.feature_extraction import create_feature_extractor


    backbone = torchvision.models.mobilenet_v2(pretrained=True)

    return_nodes = {
        "features.6": "0",
        "features.12": "1",
        "features.16": "2"
    }
    backbone = create_feature_extractor(backbone, return_nodes=return_nodes)
    # 提供给fpn每个特征层channel
    in_channel_list = get_channel_list(backbone)




    backbone_with_fpn = BackboneWithFPN(backbone, return_nodes=return_nodes,
                                        in_channels_list=in_channel_list,
                                        out_channels=256,
                                        extra_blocks=LastLevelMaxPool(),
                                        re_getter=False)

    anchor_sizes = ((64,), (128,), (256,), (512,))
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    anchor_generator = AnchorsGenerator(sizes=anchor_sizes,
                                        aspect_ratios=aspect_ratios)

    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=[f'{i}' for i in range(len(in_channel_list))],  # 在哪些特征层上进行RoIAlign pooling
                                                    output_size=[7, 7],  # RoIAlign pooling输出特征矩阵尺寸
                                                    sampling_ratio=2)  # 采样率

    model = FasterRCNN(backbone=backbone_with_fpn,
                    num_classes=num_classes,
                    rpn_anchor_generator=anchor_generator,
                    box_roi_pool=roi_pooler)

    return model


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("Using {} device training.".format(device.type))

    # 用来保存coco_info的文件
    results_file = os.path.join(args.save_path, "results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))

    data_transform = {
        "train": transforms.Compose([transforms.ToTensor(),
                                    transforms.RandomHorizontalFlip(0.5)]),
        "val": transforms.Compose([transforms.ToTensor()])
    }

    VOC_root = args.data_path

    # load train data set
    train_dataset = VOC2012DataSet(VOC_root, data_transform["train"], True)
    train_sampler = None

    # 是否按图片相似高宽比采样图片组成batch
    # 使用的话能够减小训练时所需GPU显存，默认使用
    if args.aspect_ratio_group_factor >= 0:
        train_sampler = torch.utils.data.RandomSampler(train_dataset)
        # 统计所有图像高宽比例在bins区间中的位置索引
        group_ids = create_aspect_ratio_groups(train_dataset, k=args.aspect_ratio_group_factor)
        # 每个batch图片从同一高宽比例区间中取
        train_batch_sampler = GroupedBatchSampler(train_sampler, group_ids, args.batch_size)

    # 注意这里的collate_fn是自定义的，因为读取的数据包括image和targets，不能直接使用默认的方法合成batch
    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using %g dataloader workers' % nw)
    if train_sampler:
        # 如果按照图片高宽比采样图片，dataloader中需要使用batch_sampler
        train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                        batch_sampler=train_batch_sampler,
                                                        pin_memory=True,
                                                        num_workers=nw,
                                                        collate_fn=train_dataset.collate_fn)
    else:
        train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                        batch_size=batch_size,
                                                        shuffle=True,
                                                        pin_memory=True,
                                                        num_workers=nw,
                                                        collate_fn=train_dataset.collate_fn)

    # load validation data set
    val_dataset = VOC2012DataSet(VOC_root, data_transform["val"], False)
    val_data_set_loader = torch.utils.data.DataLoader(val_dataset,
                                                    batch_size=1,
                                                    shuffle=False,
                                                    pin_memory=True,
                                                    num_workers=nw,
                                                    collate_fn=val_dataset.collate_fn)

    # create model num_classes equal background + 20 classes
    model = create_model(num_classes=args.num_classes + 1)

    model.to(device)

    # define optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params,
                                lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    scaler = torch.GradScaler("cuda") if args.amp else None

    # learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=3,
                                                gamma=0.33)

    # 如果指定了上次训练保存的权重文件地址，则接着上次结果接着训练
    if args.resume != "":
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        if args.amp and "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])
        print("the training process from epoch{}...".format(args.start_epoch))

    train_loss = []
    learning_rate = []
    val_map = []

    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch, printing every 10 iterations
        mean_loss, lr = utils.train_one_epoch(model, optimizer, train_data_loader,
                                            device=device, epoch=epoch,
                                            print_freq=50, warmup=True,
                                            scaler=scaler)
        train_loss.append(mean_loss.item())
        learning_rate.append(lr)

        # update the learning rate
        lr_scheduler.step()

        # evaluate on the test dataset
        coco_info = utils.evaluate(model, val_data_set_loader, device=device)

        # write into txt
        with open(results_file, "a") as f:
            # 写入的数据包括coco指标还有loss和learning rate
            result_info = [f"{i:.4f}" for i in coco_info + [mean_loss.item()]] + [f"{lr:.6f}"]
            txt = "epoch:{} {}".format(epoch, '  '.join(result_info))
            f.write(txt + "\n")

        val_map.append(coco_info[1])  # pascal mAP

        # save weights
        save_files = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch}
        if args.amp:
            save_files["scaler"] = scaler.state_dict()
        torch.save(save_files, os.path.join(args.save_path, "resNetFpn-model-{}.pth".format(epoch)))

    # plot loss and lr curve
    # if len(train_loss) != 0 and len(learning_rate) != 0:
    #     from plot_curve import plot_loss_and_lr
    #     plot_loss_and_lr(train_loss, learning_rate)

    # # plot mAP curve
    # if len(val_map) != 0:
    #     from plot_curve import plot_map
    #     plot_map(val_map)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__)
    save_path = os.path.join(os.path.dirname(__file__), "save_weights")
    # 训练设备类型
    parser.add_argument('--device', default='cuda:0', help='device')
    # 训练数据集的根目录(VOCdevkit)
    parser.add_argument('--data_path', default=os.path.join(os.path.dirname(__file__), "..", "data", "VOC2012"), help='dataset')
    # 检测目标类别数(不包含背景)
    parser.add_argument('--num_classes', default=20, type=int, help='num_classes')
    # 文件保存地址
    parser.add_argument('--save_path', default=save_path, help='path where to save')
    # 若需要接着上次训练，则指定上次训练保存权重文件地址
    # parser.add_argument('--resume', default=os.path.join(save_path, "resNetFpn-model-0.pth"), type=str, help='resume from checkpoint')
    parser.add_argument('--resume', default="", type=str, help='resume from checkpoint')
    # 指定接着从哪个epoch数开始训练
    parser.add_argument('--start_epoch', default=0, type=int, help='start epoch')
    # 训练的总epoch数
    parser.add_argument('--epochs', default=15, type=int, metavar='N',
                        help='number of total epochs to run')
    # 学习率
    parser.add_argument('--lr', default=0.005, type=float,
                        help='initial learning rate, 0.02 is the default value for training '
                             'on 8 gpus and 2 images_per_gpu')
    # SGD的momentum参数
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    # SGD的weight_decay参数
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    # 训练的batch size
    parser.add_argument('--batch_size', default=4, type=int, metavar='N',
                        help='batch size when training.')
    parser.add_argument('--aspect-ratio-group-factor', default=3, type=int)
    # 是否使用混合精度训练(需要GPU支持混合精度)
    parser.add_argument("--amp", default=False, help="Use torch.cuda.amp for mixed precision training")

    args = parser.parse_args()

    # 检查保存权重文件夹是否存在，不存在则创建
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    main(args)
