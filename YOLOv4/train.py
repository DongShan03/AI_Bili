import sys, os, math, warnings
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from YOLOv4.opt import opt
from YOLOv4.model.model import Darknet, YOLOLayer, JDELayer
from YOLOv4.utils.datasets import LoadImagesAndLabels, LoadImagesAndLabels9
from YOLOv4.utils.coco_utils import get_coco_api_from_dataset
from YOLOv4.utils.train_eval_utils import train_one_epoch, evaluate

import numpy as np
import torch.optim.lr_scheduler as lr_scheduler

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from tensorboardX import SummaryWriter

warnings.filterwarnings('ignore')

def train():
    device = opt.device
    print("Using {} device training.".format(device.type))
    accumulate = max(round(64 / opt.batch_size), 1)
    imgsz_train =imgsz_test = opt.img_size
    multi_scale = opt.muliti_scale
    hyp = opt.hyp
    tb_writer = SummaryWriter(log_dir=os.path.join(opt.file_dir, "log"))

    results_file = opt.save_name + "_results.txt"
    results_file = os.path.join(opt.save_path, results_file)
    YOLO_root = opt.data_root
    gs = 64
    assert math.fmod(opt.img_size, gs) == 0, "Image sizes must be a multiple of 64!"
    grid_min, grid_max = imgsz_test // gs, imgsz_test // gs
    if multi_scale:
        imgsz_min = opt.img_size // 1.5
        imgsz_max = opt.img_size // 0.667
        grid_min, grid_max = imgsz_min // gs, imgsz_max // gs
        imgsz_min, imgsz_max = int(grid_min * gs), int(grid_max * gs)
        imgsz_train = imgsz_max
        print("Using multi_scale training, imgsz range[{}, {}]".format(imgsz_min, imgsz_max))

    train_path = os.path.join(YOLO_root, "train")
    val_path = os.path.join(YOLO_root, "val")
    nc = 1 if opt.single_cls else int(opt.num_classes)  # number of classes
    hyp["cls"] *= nc / 80  # update coco-tuned hyp['cls'] to current dataset
    hyp["obj"] *= imgsz_test / 320

    model = Darknet(opt.cfg).to(device)
    # 是否冻结权重，只训练predictor的权重
    if opt.freeze_layer:
        # 索引减一对应的是predictor的索引，YOLOLayer并不是predictor
        output_layer_indices = [idx - 1 for idx, module in enumerate(model.module_list) if isinstance(module, YOLOLayer) or isinstance(module, JDELayer)]

        # 冻结除了predictor和YOLOLayer外的所有参数
        freeze_layer_indeces = [
            x for x in range(len(model.module_list)) \
                if (x not in output_layer_indices) and \
                (x - 1 not in output_layer_indices)
        ]

        for idx in freeze_layer_indeces:
            for param in model.module_list[idx].parameters():
                param.requires_grad = False
    else:
        pass

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(pg, lr=hyp["lr0"], momentum=hyp["momentum"],
                                weight_decay=hyp["weight_decay"], nesterov=True)
    scaler = torch.GradScaler("cuda") if device.type == "cuda" else None



    epochs = opt.epochs
    start_epoch = 0
    best_map = 0.0
    if opt.resume.endswith(".pt") or opt.resume.endswith(".pth"):
        ckpt = torch.load(opt.resume, map_location=device, weights_only=False)
        try:
            pre_weights_dict = ckpt["model"]
            missing_keys, unexpected_keys = model.load_state_dict(pre_weights_dict, strict=False)
            if len(missing_keys) != 0 or len(unexpected_keys) != 0:
                print("missing_keys: ", missing_keys)
                print("unexpected_keys: ", unexpected_keys)
            # ckpt["model"] = {k: v for k, v in ckpt["model"].items() if model.state_dict()[k].numel() == v.numel()}
            model.load_state_dict(ckpt["model"], strict=False)
        except KeyError as e:
            s = "%s is not compatible with .pth weights: missing key %s" % (opt.resume, e.args[0])
            raise KeyError(s) from e

        #* 如果lr太小就把这一段注释掉
        if ckpt.get("optimizer", None) is not None:
            optimizer.load_state_dict(ckpt["optimizer"])
            if "best_map" in ckpt.keys():
                best_map = ckpt["best_map"]

        if ckpt.get("training_result", None) is not None:
            with open(results_file, "w") as file:
                file.write(ckpt["training_result"])

        if ckpt.get("epoch", None) is not None:
            start_epoch = ckpt["epoch"] + 1

        if opt.epochs < start_epoch:
            epochs = start_epoch + 10

        if opt.amp and "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        del ckpt

    lf = lambda x: ((1 + math.cos(x * math.pi / epochs)) / 2) * (1 - hyp['lrf']) + hyp['lrf']  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    scheduler.last_epoch = start_epoch - 1 if start_epoch > 0 else start_epoch
    scheduler.step()

    train_dataset = LoadImagesAndLabels(
        train_path, imgsz_train, opt.batch_size, augment=True,
        hyp=hyp, rect=opt.rect, cache_images=opt.cache_images,
        single_cls=opt.single_cls
    )
    val_dataset = LoadImagesAndLabels(
        val_path, imgsz_test, opt.batch_size,
        hyp=hyp, rect=True, cache_images=opt.cache_images,
        single_cls=opt.single_cls
    )
    nw = min([os.cpu_count(), opt.batch_size if opt.batch_size > 1 else 0, 8])  # number of workers
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        num_workers=nw,
        shuffle=False,
        pin_memory=True,
        collate_fn=train_dataset.collate_fn
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=opt.batch_size,
        num_workers=nw,
        shuffle=False,
        pin_memory=True,
        collate_fn=val_dataset.collate_fn
    )


    model.nc = nc
    model.hyp = hyp
    #* GIoU Loss ratio
    model.gr = 1.0

    coco = get_coco_api_from_dataset(val_dataset)
    print("starting traning for %g epochs..." % epochs)
    print('Using %g dataloader workers' % nw)

    for epoch in range(start_epoch, epochs):
        mloss, lr = train_one_epoch(
            model, optimizer, train_dataloader,
            device, epoch,
            accumulate=accumulate,  # 迭代多少batch才训练完64张图片
            img_size=imgsz_train,  # 输入图像的大小
            multi_scale=multi_scale,
            grid_min=grid_min,  # grid的最小尺寸
            grid_max=grid_max,  # grid的最大尺寸
            gs=gs,  # grid step: 32
            print_freq=50,  # 每训练多少个step打印一次信息
            warmup=True,
            scaler=scaler
        )
        # update scheduler
        scheduler.step()
        if opt.no_test is False or epoch == epochs - 1:
            result_info = evaluate(model, val_dataloader, coco=coco, device=device)
            coco_mAP = result_info[0]
            voc_mAP = result_info[1]
            coco_mAR = result_info[2]

            if tb_writer:
                tags = ['train/giou_loss', 'train/obj_loss', 'train/cls_loss', 'train/loss', "learning_rate",
                        "mAP@[IoU=0.50:0.95]", "mAP@[IoU=0.5]", "mAR@[IoU=0.50:0.95]"]

                for x, tag in zip(mloss.tolist() + [lr, coco_mAP, voc_mAP, coco_mAR], tags):
                    tb_writer.add_scalar(tag, x, epoch)

                with open(results_file, 'a') as f:
                     # 记录coco的12个指标加上训练总损失和lr
                    result_info = [str(round(i, 4)) for i in result_info + [mloss.tolist()[-1]]] + [str(round(lr, 6))]
                    txt = "epoch: {} {}".format(epoch, '  '.join(result_info))
                    f.write(txt + "\n")

                if coco_mAP > best_map:
                    best_map = coco_mAP

                if opt.save_best is False:
                    with open(results_file, 'r') as f:
                        save_files = {
                            'model': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'training_results': f.read(),
                            'epoch': epoch,
                            'best_map': best_map}
                        if opt.amp:
                            save_files["scaler"] = scaler.state_dict()
                        torch.save(save_files, os.path.join(opt.save_path, opt.save_name + f"-{epoch}.pth"))
                else:
                    if best_map == coco_mAP:
                        with open(results_file, 'r') as f:
                            save_files = {
                                'model': model.state_dict(),
                                'optimizer': optimizer.state_dict(),
                                'training_results': f.read(),
                                'epoch': epoch,
                                'best_map': best_map}
                            if opt.amp:
                                save_files["scaler"] = scaler.state_dict()
                        torch.save(save_files, os.path.join(opt.save_path, opt.save_name + f"-best-{epoch}.pth"))


if __name__ == "__main__":
    train()
