import sys, os, math, warnings, random
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from YOLOv8.opt import opt
from YOLOv8.model.yolo import Model, Detect
from YOLOv8.utils.utils import *
from YOLOv8.utils.coco_utils import get_coco_api_from_dataset
from YOLOv8.utils.train_eval_utils import train_one_epoch, evaluate
from YOLOv8.utils.loss import ComputeLoss
from YOLOv8.utils.build import build_dataloader, build_yolo_dataset

import numpy as np
import torch.optim.lr_scheduler as lr_scheduler
import torch
from tensorboardX import SummaryWriter

warnings.filterwarnings('ignore')

def train():
    device = opt.device
    print("Using {} device training.".format(device.type))
    accumulate = max(round(64 / opt.batch_size), 1)
    imgsz_train = imgsz_test = opt.img_size
    multi_scale = opt.muliti_scale
    hyp = opt.hyp
    hyp = HYP(hyp)

    tb_writer = SummaryWriter(log_dir=os.path.join(opt.file_dir, "log"))

    results_file = opt.save_name + "_results.txt"
    results_file = os.path.join(opt.save_path, results_file)
    YOLO_root = opt.data_root

    train_path = os.path.join(YOLO_root, "train")
    val_path = os.path.join(YOLO_root, "val")
    nc = 1 if opt.single_cls else int(opt.num_classes)  # number of classes
    include_classes = range(1, nc + 1)

    model = Model(cfg=opt.cfg, phi=opt.phi).to(device)

    nl = model.model[-1].nl
    hyp.box *= 3. / nl  # scale to layers
    hyp.cls *= nc / 80. * 3. / nl  # scale to classes and layers
    hyp.dfl *= (imgsz_train / 640) ** 2 * 3. / nl  # scale to image size and layers

    gs = 64
    assert math.fmod(opt.img_size, gs) == 0, "Image sizes must be a multiple of 64!"
    grid_min, grid_max = imgsz_test // gs, imgsz_test // gs
    #* 模型内部集成了多尺度训练
    if multi_scale:
        imgsz_min = opt.img_size // 1.2
        imgsz_max = opt.img_size // 0.833
        grid_min, grid_max = imgsz_min // gs, imgsz_max // gs
        imgsz_min, imgsz_max = int(grid_min * gs), int(grid_max * gs)
        imgsz_train = imgsz_max
        print("Using multi_scale training, imgsz range[{}, {}]".format(imgsz_min, imgsz_max))

    # 是否冻结权重，只训练predictor的权重
    if opt.freeze_layer:
        # 索引减一对应的是predictor的索引，YOLOLayer并不是predictor
        output_layer_indices = [idx - 1 for idx, module in enumerate(model.module_list) \
                            if isinstance(module, Detect)]

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
    optimizer = torch.optim.SGD(pg, lr=hyp.lr0, momentum=hyp.momentum,
                                weight_decay=hyp.weight_decay, nesterov=True)
    scaler = torch.GradScaler("cuda") if device.type == "cuda" else None

    epochs = opt.epochs
    start_epoch = 1
    best_map = 0.0

    if opt.resume.endswith(".pt") or opt.resume.endswith(".pth"):
        ckpt = torch.load(opt.resume, map_location=device, weights_only=False)
        try:
            pre_weights_dict = ckpt.get("model", ckpt)
            missing_keys, unexpected_keys = model.load_state_dict(pre_weights_dict, strict=False)
            if len(missing_keys) != 0 or len(unexpected_keys) != 0:
                print("missing_keys: ", missing_keys)
                print("unexpected_keys: ", unexpected_keys)

            pre_weights_dict = {k: v for k, v in pre_weights_dict.items() if model.state_dict()[k].numel() == v.numel()}
            model.load_state_dict(pre_weights_dict, strict=False)
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
            epochs = start_epoch + 20 - 1

        if opt.mixed_precision and "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        del ckpt
    lf = lambda x: ((1 + math.cos(x * math.pi / epochs)) / 2) * (1 - hyp.lrf) + hyp.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    scheduler.last_epoch = min(start_epoch - 1, 0)
    scheduler.step()

    train_dataset = build_yolo_dataset(
        train_path, imgsz_train, opt.batch_size, mode="train",
        hyp=hyp, cache=opt.cache_images, rect=opt.rect,
        single_cls=opt.single_cls,
        classes=include_classes, fraction=opt.fraction
    )

    val_dataset = build_yolo_dataset(
        val_path, imgsz_test, opt.batch_size, mode="val",
        hyp=hyp, cache=opt.cache_images, rect=True,
        single_cls=opt.single_cls,
        classes=include_classes
    )
    opt.batch_size = min(opt.batch_size, len(train_dataset))
    nw = min([os.cpu_count(), opt.batch_size if opt.batch_size > 1 else 0, 8])  # number of workers

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        num_workers=nw,
        shuffle=not opt.rect,
        pin_memory=True,
        collate_fn=train_dataset.collate_fn
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=opt.batch_size,
        num_workers=nw,
        shuffle=False,
        pin_memory=False,
        collate_fn=val_dataset.collate_fn
    )

    model.nc = nc
    model.hyp = hyp
    #* GIoU Loss ratio

    ema = ModelEMA(model)

    computeLoss = ComputeLoss(model)
    coco = get_coco_api_from_dataset(val_dataset)
    print("starting training for %g epochs..." % epochs)
    print("beginning training from %g epochs..." % start_epoch)

    for epoch in range(start_epoch, epochs + 1):
        mloss, lr = train_one_epoch(
            model, optimizer, train_dataloader,
            computeLoss,
            device, epoch, epochs,
            accumulate=accumulate,  # 迭代多少batch才训练完64张图片
            img_size=imgsz_train,  # 输入图像的大小
            multi_scale=multi_scale,
            grid_min=grid_min,  # grid的最小尺寸
            grid_max=grid_max,  # grid的最大尺寸
            gs=gs,  # grid step: 32
            warmup=True,
            scaler=scaler,
            use_OTA= not hasattr(hyp, 'loss_ota') or hyp.loss_ota == 1
        )
        # update scheduler
        scheduler.step()
        ema.update_attr(model)

        if opt.no_test is False or epoch == epochs:
            result_info = evaluate(model, val_dataloader, coco=coco, device=device)
            coco_mAP = result_info[0]
            voc_mAP = result_info[1]
            coco_mAR = result_info[2]

            if tb_writer:
                tags = ['train/ciou_loss', 'train/obj_loss', 'train/cls_loss', 'train/loss', "learning_rate",
                        "mAP@[IoU=0.50:0.95]", "mAP@[IoU=0.5]", "mAR@[IoU=0.50:0.95]"]

                for x, tag in zip(mloss.tolist() + [lr, coco_mAP, voc_mAP, coco_mAR], tags):
                    tb_writer.add_scalar(tag, x, epoch)

                result_tags = ["AP@[IoU=0.50:0.95]", "AP@[IoU=0.5]", "AP@[IoU=0.75]",
                               "AP@[area=small]", "AP@[area=middle]", "AP@[area=large]",
                               "AR@[1 per image]", "AR@[10 per image]", "AR@[100 per image]",
                               "AR@[area=small]", "AR@[area=middle]", "AR@[area=large]", "mloss", 'lr']
                with open(results_file, 'a') as f:
                    # 记录coco的12个指标加上训练总损失和lr
                    if epoch == 1:
                        f.write('Epoch  ' + '  '.join(result_tags) + "\n")
                    result_info = ["%.4f"%i for i in result_info + [mloss.tolist()[-1]]] + ["%.6f"%lr]
                    txt = "epoch: {:>3d}  {}".format(epoch, '  '.join(result_info))
                    f.write(txt + "\n")
                    f.close()

                if coco_mAP > best_map:
                    best_map = coco_mAP

                if opt.save_best is False:
                    with open(results_file, 'r') as f:
                        save_files = {
                            'model': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'training_results': f.read(),
                            'epoch': epoch,
                            'best_map': best_map
                            }
                        if opt.mixed_precision:
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
                                'best_map': best_map
                                }
                            if opt.mixed_precision:
                                save_files["scaler"] = scaler.state_dict()
                        torch.save(save_files, os.path.join(opt.save_path, opt.save_name + f"-best-{epoch}.pth"))


if __name__ == "__main__":
    train()
