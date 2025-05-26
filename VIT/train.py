import torch.optim as optim
import os, torch, math
from tensorboardX import SummaryWriter
import torch.optim.lr_scheduler as lr_scheduler
from cfg import cfg
from utils import train_one_epoch, evaluate, preprocess

def train(net_name, epochs=20, learn_rate=0.0002):
    writer = SummaryWriter(os.path.join(cfg['dir_root'], cfg["net_name"] + "_log"), comment=cfg['data_name'])
    train_loader, validate_loader = preprocess.train_loader, preprocess.validate_loader
    if not os.path.exists(os.path.dirname(cfg["structure"])):
        os.mkdir(os.path.dirname(cfg["structure"]))
    if os.path.exists(cfg["structure"]):
        print(net_name + "模型结构已存在！")
    else:
        print(net_name + "模型结构不存在！")
        with open(cfg["structure"], 'w', encoding='utf-8') as f:
            f.write(str(cfg["net"]))
        #! tensorboard 模型图
        writer.add_graph(cfg["net"], torch.zeros(cfg["batch_size"], 3, cfg["train_size"], cfg["train_size"]).to(cfg["device"]))
        print("模型结构绘制成功!")
    if not os.path.exists(os.path.dirname(cfg["save_path"])):
        os.mkdir(os.path.dirname(cfg["save_path"]))
    if os.path.exists(cfg["save_path"]):
        print(net_name + "_" + cfg["data_name"] + "模型存在!")
        # #! 迁移学习
        if cfg["transfer_learning"]:
            print("使用迁移学习,冻结除最后一层的所有参数!")
            pre_weight = torch.load(cfg["save_path"])
            pre_dict = {k : v for k, v in pre_weight.items() if "head" or "pre_logits" not in k}
            cfg["net"].load_state_dict(pre_dict, strict=False)
            for name, param in cfg["net"].named_parameters():
                if "head" not in name and "pre_logits" not in name:
                    param.requires_grad = False
        else:
            cfg["net"].load_state_dict(torch.load(cfg["save_path"]))
    else:
        print(net_name + "_" + cfg["data_name"] + "模型不存在!")
    print("开始训练!")
    #* 从这准备训练
    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    pg = [p for p in cfg["net"].parameters() if p.requires_grad]
    lf = lambda x: ((1 + math.cos(x * math.pi / epochs)) / 2) * (1 - learn_rate * 0.1) + learn_rate * 0.1
    optimizer = optim.SGD(pg, lr=learn_rate, momentum=0.9, weight_decay=5E-5)
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    best_acc = 0.0
    for epoch in range(epochs):
        # train
        train_loss, train_acc = train_one_epoch(model=cfg["net"],
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                device=cfg["device"],
                                                epoch=epoch)

        scheduler.step()

        # validate
        val_loss, val_acc = evaluate(model=cfg["net"],
                                    data_loader=validate_loader,
                                    device=cfg["device"],
                                    epoch=epoch)

        tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
        writer.add_scalar(tags[0], train_loss, epoch)
        writer.add_scalar(tags[1], train_acc, epoch)
        writer.add_scalar(tags[2], val_loss, epoch)
        writer.add_scalar(tags[3], val_acc, epoch)
        writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(cfg["net"].state_dict(), cfg["save_path"])
    writer.close()
    print("Finished Training!")

def main():
    train(cfg["net_name"], epochs=cfg["epochs"], learn_rate=cfg["learn_rate"])

if __name__ == "__main__":
    main()
