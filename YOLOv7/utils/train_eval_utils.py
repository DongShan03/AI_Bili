import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import random, math, time
import torch.nn.functional as F
from tqdm import tqdm
from YOLOv7.utils.utils import *
from YOLOv7.utils.coco_eval import CocoEvaluator
from YOLOv7.utils.coco_utils import get_coco_api_from_dataset
import YOLOv7.utils.distributed_utils as utils


def train_one_epoch(model, optimizer, data_loader,
                    computeLoss, computeLossOTA,
                    device, epoch, epochs,
                    accumulate, img_size,
                    grid_min, grid_max, gs,
                    multi_scale=False, warmup=False, scaler=None, use_OTA=False):
    model.train()

    lr_scheduler = None
    if epoch == 1 and warmup is True:  # 当训练第一轮（epoch=0）时，启用warmup训练方式，可理解为热身训练
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)
        accumulate = 1

    mloss = torch.zeros(4).to(device)  # mean losses
    now_lr = 0.
    nb = len(data_loader)  # number of batches
    # imgs: [batch_size, 3, img_size, img_size]
    # targets: [num_obj, 6] , that number 6 means -> (img_index, obj_index, x, y, w, h)
    # paths: list of img path
    pbar = tqdm(enumerate(data_loader), total=nb)
    for i, (imgs, targets, paths, _, _) in pbar:
        # ni 统计从epoch0开始的所有batch数
        ni = i + nb * epoch  # number integrated batches (since train start)
        imgs = imgs.to(device).float() / 255.0  # uint8 to float32, 0 - 255 to 0.0 - 1.0
        targets = targets.to(device)

        # Multi-Scale
        if multi_scale:
            # 每训练64张图片，就随机修改一次输入图片大小，
            # 由于label已转为相对坐标，故缩放图片不影响label的值
            if ni % accumulate == 0:  # adjust img_size (67% - 150%) every 1 batch
                # 在给定最大最小输入尺寸范围内随机选取一个size(size为32的整数倍)
                img_size = random.randrange(grid_min, grid_max + 1) * gs
            sf = img_size / max(imgs.shape[2:])  # scale factor

            # 如果图片最大边长不等于img_size, 则缩放图片，并将长和宽调整到32的整数倍
            if sf != 1:
                # gs: (pixels) grid size
                ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]  # new shape (stretched to 32-multiple)
                imgs = F.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

        with torch.amp.autocast("cuda", enabled=scaler is not None):
            pred = model(imgs, augment=False)  # forward
            if use_OTA:
                loss, loss_items = computeLossOTA(pred, targets.to(device), imgs)
            else:
                loss, loss_items = computeLoss(pred, targets.to(device))  # loss scaled by batch_size
        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            print("training image path: {}".format(",".join(paths)))
            sys.exit(1)

        # backward
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # optimize
        # 每训练64张图片更新一次权重
        if ni % accumulate == 0:
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

        mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
        now_lr = optimizer.param_groups[0]["lr"]
        mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
        s = ('Epochs: %g/%g ' + 'MEM: %4s ' + 'lr:%.6f ' + 'lbox:%8.4g ' + 'lobj:%8.4g ' + 'lcls:%8.4g ' + "loss:%8.4g") % \
            (epoch, epochs, mem, now_lr, *mloss)
        pbar.set_description(s)

        if ni % accumulate == 0 and lr_scheduler is not None:  # 第一轮使用warmup训练方式
            lr_scheduler.step()

    return mloss, now_lr


@torch.no_grad()
def evaluate(model, data_loader, coco=None, device=None):
    cpu_device = torch.device("cpu")
    model.eval()

    if coco is None:
        coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    for imgs, targets, paths, shapes, img_index in tqdm(data_loader, desc="Evaluating for coco evaluator: "):
        imgs = imgs.to(device).float() / 255.0  # uint8 to float32, 0 - 255 to 0.0 - 1.0
        # targets = targets.to(device)

        # 当使用CPU时，跳过GPU相关指令
        if device != torch.device("cpu"):
            torch.cuda.synchronize(device)

        model_time = time.time()
        pred = model(imgs)[0]  # only get inference result
        pred = non_max_suppression(pred, conf_thres=0.01, iou_thres=0.6)
        model_time = time.time() - model_time

        outputs = []
        for index, p in enumerate(pred):
            if p is None:
                p = torch.empty((0, 6), device=cpu_device)
                boxes = torch.empty((0, 4), device=cpu_device)
            else:
                # xmin, ymin, xmax, ymax
                boxes = p[:, :4]
                # shapes: (h0, w0), ((h / h0, w / w0), pad)
                # 将boxes信息还原回原图尺度，这样计算的mAP才是准确的
                boxes = scale_coords(imgs[index].shape[1:], boxes, shapes[index][0]).round()

            # 注意这里传入的boxes格式必须是xmin, ymin, xmax, ymax，且为绝对坐标
            info = {"boxes": boxes.to(cpu_device),
                    "labels": p[:, 5].to(device=cpu_device, dtype=torch.int64),
                    "scores": p[:, 4].to(cpu_device)}
            outputs.append(info)

        res = {img_id: output for img_id, output in zip(img_index, outputs)}

        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time

    # gather the stats from all processes
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()

    result_info = coco_evaluator.coco_eval[iou_types[0]].stats.tolist()  # numpy to list

    return result_info


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    return iou_types
