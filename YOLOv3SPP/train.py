import os, sys, math
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from cfg import cfg
from tensorboardX import SummaryWriter
import torch
from model.model import Yolov3SPP, YOLOLayer
from YOLOv3SPP.utils.datasets import LoadImagesAndLabels
from YOLOv3SPP.utils.coco_utils import get_coco_api_from_dataset
from YOLOv3SPP.utils.train_eval_utils import train_one_epoch, evaluate

def train():
    device = cfg["device"]
    print("Using {} device training.".format(device.type))
    if not os.path.exists(cfg["save_path"]):
        os.mkdir(cfg["save_path"])

    accumulate = max(round(64 / cfg["batch_size"]), 1)
    imgsz_train = cfg["img_size"]
    imgsz_test = cfg["img_size"]
    multi_scale = cfg["muliti_scale"]
    hyp = cfg["hyp"]
    tb_writer = SummaryWriter(log_dir=os.path.join(cfg["file_dir"], "log")),

    results_file = cfg["save_name"] + "_results.txt"
    results_file = os.path.join(cfg["save_path"], results_file)
    # data_transform = cfg["transform"]
    YOLO_root = cfg["data_root"]
    #! 图像需要是32的倍数
    gs = 32
    assert math.fmod(cfg["img_size"], gs) == 0, "Image sizes must be a multiple of 32!"
    grid_min, grid_max = imgsz_test // gs, imgsz_test // gs
    if multi_scale:
        imgsz_min = cfg["img_size"] // 1.5
        imgsz_max = cfg["img_size"] // 0.667
        grid_min, grid_max = imgsz_min // gs, imgsz_max // gs
        imgsz_min, imgsz_max = int(grid_min * gs), int(grid_max * gs)
        imgsz_train = imgsz_max
        print("Using multi_scale training, imgsz range[{}, {}]".format(imgsz_min, imgsz_max))

    train_path = os.path.join(YOLO_root, "train")
    val_path = os.path.join(YOLO_root, "val")
    nc = 1 if cfg["single_cls"] else int(cfg["num_classes"])  # number of classes
    hyp["cls"] *= nc / 80  # update coco-tuned hyp['cls'] to current dataset
    hyp["obj"] *= imgsz_test / 320

    model = Yolov3SPP().to(device)

     # 是否冻结权重，只训练predictor的权重
    if cfg["freeze_layer"]:
        # 索引减一对应的是predictor的索引，YOLOLayer并不是predictor
        output_layer_indices = [idx - 1 for idx, module in enumerate(model.module_list) if isinstance(module, YOLOLayer)]

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
        #* 如果freeze_layer为false     默认仅训练除darknet53之后的部分
        darknet_end_layer = 74
        for idx in range(darknet_end_layer + 1):
            for param in model.module_list[idx].parameters():
                param.requires_grad = False

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(pg, lr=hyp["lr0"], momentum=hyp["momentum"],
                                weight_decay=hyp["weight_decay"], nesterov=True)
    scaler = torch.GradScaler("cuda") if device.type == "cuda" else None
    epochs = cfg["epochs"]
    start_epoch = 0
    best_map = 0.0
    if cfg["resume"].endswith(".pt") or cfg["resume"].endswith(".pth"):
        ckpt = torch.load(cfg["resume"], map_location=device, weights_only=False)
        try:
            ckpt["model"] = {k: v for k, v in ckpt["model"].items() if model.state_dict()[k].numel() == v.numel()}
            model.load_state_dict(ckpt["model"], strict=False)
        except KeyError as e:
            s = "%s is not compatible with .pth weights: missing key %s" % (cfg["resume"], e.args[0])
            raise KeyError(s) from e

        #* 如果lr太小就把这一段注释掉
        if ckpt["optimizer"] is not None:
            optimizer.load_state_dict(ckpt["optimizer"])
            if "best_map" in ckpt.keys():
                best_map = ckpt["best_map"]

        if ckpt.get("training_result") is not None:
            with open(results_file, "w") as file:
                file.write(ckpt["training_result"])

        start_epoch = ckpt["epoch"] + 1
        if cfg["epochs"] < start_epoch:
            epochs += ckpt['epoch']

        if cfg["amp"] and "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        del ckpt

    lf = lambda x: ((1 + math.cos(x * math.pi / epochs)) / 2) * (1 - hyp["lrf"]) + hyp["lrf"]  # cosine
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    scheduler.last_epoch = start_epoch

    train_dataset = LoadImagesAndLabels(train_path, imgsz_train, cfg["batch_size"],
                                        augment=True, hyp=hyp, rect=cfg["rect"],
                                        cache_images=cfg["cache_images"], single_cls=cfg["single_cls"])

    val_dataset = LoadImagesAndLabels(val_path, imgsz_test, cfg["batch_size"],
                                    hyp=hyp, rect=True, cache_images=cfg["cache_images"],
                                    single_cls=cfg["single_cls"])
    nw = min([os.cpu_count(), cfg["batch_size"] if cfg["batch_size"] > 1 else 0, 8])  # number of workers
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                batch_size=cfg["batch_size"],
                                                shuffle=not cfg["rect"],
                                                num_workers=nw,
                                                pin_memory=True,
                                                collate_fn=train_dataset.collate_fn)
    val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                batch_size=cfg["batch_size"],
                                                shuffle=False,
                                                num_workers=nw,
                                                pin_memory=True,
                                                collate_fn=val_dataset.collate_fn)

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
        if cfg["notest"] is False or epoch == epochs - 1:
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

                if cfg["savebest"] is False:
                    with open(results_file, 'r') as f:
                        save_files = {
                            'model': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'training_results': f.read(),
                            'epoch': epoch,
                            'best_map': best_map}
                        if cfg["amp"]:
                            save_files["scaler"] = scaler.state_dict()
                        torch.save(save_files, os.path.join(cfg["save_path"], cfg["save_name"] + f"-{epoch}.pth"))
                else:
                    if best_map == coco_mAP:
                        with open(results_file, 'r') as f:
                            save_files = {
                                'model': model.state_dict(),
                                'optimizer': optimizer.state_dict(),
                                'training_results': f.read(),
                                'epoch': epoch,
                                'best_map': best_map}
                            if cfg["amp"]:
                                save_files["scaler"] = scaler.state_dict()
                        torch.save(save_files, os.path.join(cfg["save_path"], cfg["save_name"] + f"-best-{epoch}.pth"))


if __name__ == "__main__":
    train()
