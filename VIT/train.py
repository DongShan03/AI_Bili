import torch.optim as optim
import torch.nn as nn
import os, time, torch
from tensorboardX import SummaryWriter
from cfg import cfg
from preprocess import get_data_loader

def train(net_name, epochs=20, learn_rate=0.0002):
    writer = SummaryWriter(os.path.join(cfg['dir_root'], cfg["net_name"] + "_log"), comment=cfg['data_name'])
    train_num, validate_num, train_loader, validate_loader = get_data_loader()
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
            pre_dict = {k : v for k, v in pre_weight.items() if "head" not in k}
            missing_keys, unexpected_keys = cfg["net"].load_state_dict(pre_dict, strict=False)
            for param in cfg["net"].parameters():
                param.requires_grad = False
            for param in cfg["net"].head.parameters():
                param.requires_grad = True
        else:
            cfg["net"].load_state_dict(torch.load(cfg["save_path"]))
    else:
        print(net_name + "_" + cfg["data_name"] + "模型不存在!")
    print("开始训练!")
    cfg["net"].train()
    #* 从这准备训练
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(cfg["net"].parameters(), lr=learn_rate)
    best_acc = 0.0

    for epoch in range(epochs):
        cfg["net"].train()
        running_loss = 0.0
        t1 = time.perf_counter()
        for step, data in enumerate(train_loader, start=0):
            images, labels = data
            optimizer.zero_grad()
            logits = cfg["net"](images.to(cfg["device"]))
            loss = loss_function(logits, labels.to(cfg["device"]))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            rate = (step + 1) / len(train_loader)
            a = "*" * int(rate * 50)
            b = "." * int((1 - rate) * 50)
            print("\rtrain loss: {:^3.0f}%[{}->{}]{:.3f}".format(int(rate*100), a, b, loss), end="")
        print()             #* print中\r表示回到行首
        print("Train time: %.3f s" % (time.perf_counter() - t1))

        cfg["net"].eval()
        acc = 0.0
        with torch.no_grad():
            for data_test in validate_loader:
                test_images, test_labels = data_test
                outputs = cfg["net"](test_images.to(cfg["device"]))
                predict_y = torch.max(outputs, 1)[1]
                acc += (predict_y == test_labels.to(cfg["device"])).sum().item()
            accurate_test = acc / validate_num
            train_loss = running_loss / step
            if accurate_test > best_acc:
                best_acc = accurate_test
                torch.save(cfg["net"].state_dict(), cfg["save_path"])
            writer.add_scalar("scalar/" + cfg['net_name'] + '_' + cfg['data_name'] + "test_accuracy", accurate_test, epoch)
            writer.add_scalar("scalar/" + cfg['net_name'] + '_' + cfg['data_name'] + "train_loss", train_loss, epoch)
            print('[epoch %d] train_loss: %.3f  test_accuracy: %.3f' %
                (epoch + 1, train_loss, accurate_test))
    writer.close()
    print("Finished Training!")

def main():
    train(cfg["net_name"], epochs=cfg["epochs"], learn_rate=cfg["learn_rate"])

if __name__ == "__main__":
    main()
