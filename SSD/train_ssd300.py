import os, sys
sys.path.append(os.path.dirname(__file__))
from cfg import cfg
import torch, datetime
from my_dataset import VOCDataset
from train_utils.coco_utils import get_coco_api_from_dataset
from train_utils import train_eval_utils

def create_model():
    model = cfg["model"]
    pre_ssd_path = os.path.join(cfg["save_path"], "resnet-SSD-origin.pt")
    if not os.path.exists(pre_ssd_path):
        raise FileNotFoundError(pre_ssd_path)
    pre_model_dict = torch.load(pre_ssd_path, map_location='cpu')
    pre_weights_dict = pre_model_dict["model"]

    # 删除类别预测器权重，注意，回归预测器的权重可以重用，因为不涉及num_classes
    del_conf_loc_dict = {}
    for k, v in pre_weights_dict.items():
        split_key = k.split(".")
        if "conf" in split_key:
            continue
        del_conf_loc_dict.update({k: v})
    missing_keys, unexpected_keys = model.load_state_dict(del_conf_loc_dict, strict=False)
    if len(missing_keys) != 0 or len(unexpected_keys) != 0:
        print("missing_keys: ", missing_keys)
        print("unexpected_keys: ", unexpected_keys)

    return model

def main():
    device = cfg["device"]
    print("Using {} device training.".format(device.type))
    if not os.path.exists(cfg["save_path"]):
        os.mkdir(cfg["save_path"])

    results_file = "results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    results_file = os.path.join(cfg["save_path"], results_file)
    data_transform = cfg["transform"]
    VOC_root = cfg["data_root"]

    train_dataset = VOCDataset(VOC_root, data_transform["train"], file_name="train.txt")
    batch_size = cfg["batch_size"]
    assert batch_size > 1, "batch size must be greater than 1"
     # 防止最后一个batch_size=1，如果最后一个batch_size=1就舍去
    drop_last = True if len(train_dataset) % batch_size == 1 else False
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using %g dataloader workers' % nw)
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=nw,
        collate_fn=train_dataset.collate_fn,
        drop_last=drop_last
    )

    val_dataset = VOCDataset(VOC_root, data_transform["val"], file_name="val.txt")
    val_data_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=nw,
        collate_fn=val_dataset.collate_fn
    )

    model = create_model()
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=cfg["lr"], momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.3)

    if cfg["resume"] != "":
        checkpoint = torch.load(cfg["resume"], map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        cfg["start_epoch"] = checkpoint["epoch"] + 1
        print("the training process from epoch{}...".format(cfg["start_epoch"]))

    train_loss = []
    learning_rate = []
    val_map = []

     # 提前加载验证集数据，以免每次验证时都要重新加载一次数据，节省时间
    val_data = get_coco_api_from_dataset(val_data_loader.dataset)
    for epoch in range(cfg["start_epoch"], cfg["epochs"]):
        mean_loss, lr = train_eval_utils.train_one_epoch(
            model=model, optimizer=optimizer,
            data_loader=train_data_loader,
            device=device, epoch=epoch,
            print_freq=50
        )
        train_loss.append(mean_loss)
        learning_rate.append(lr)
        lr_scheduler.step()
        coco_info = train_eval_utils.evaluate(
            model=model, data_loader=val_data_loader,
            device=device, data_set=val_data
        )

        with open(results_file, 'a') as f:
            result_info = [str(round(i, 4)) for i in coco_info + [mean_loss.item()]] + [str(round(lr, 6))]
            txt = "epoch:{} {}".format(epoch, '  '.join(result_info))
            f.write(txt + "\n")

        #! pascal mAP
        val_map.append(coco_info[1])

        # save weights
        save_files = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch}
        torch.save(save_files, os.path.join(cfg["save_path"], "ssd300-{}.pth".format(epoch)))


if __name__ == "__main__":
    main()
