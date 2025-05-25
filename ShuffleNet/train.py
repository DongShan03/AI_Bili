import torch.optim as optim
import torch.nn as nn
import os, time, torch
from cfg import cfg
from preprocess import get_data_loader


def train(net_name, epochs=20, learn_rate=0.0002):
    train_num, validate_num, train_loader, validate_loader = get_data_loader()
    if os.path.exists(cfg["save_path"]):
        print(net_name + "模型已存在，继续训练！")
        # #! 迁移学习
        # pre_weight = torch.load(save_path)
        # pre_dict = {k : v for k, v in pre_weight.items() if "classifier" not in k}
        # missing_keys, unexpected_keys = net.load_state_dict(pre_dict, strict=False)
        # for param in net.features.parameters():
        #     param.requires_grad = False
        cfg["net"].load_state_dict(torch.load(cfg["save_path"]))
    else:
        print(net_name + "模型不存在")
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
            if accurate_test > best_acc:
                best_acc = accurate_test
                torch.save(cfg["net"].state_dict(), cfg["save_path"])
            print('[epoch %d] train_loss: %.3f  test_accuracy: %.3f' %
                (epoch + 1, running_loss / step, acc / validate_num))

    print("Finished Training!")

def main():
    train(cfg["net_name"], epochs=cfg["epochs"], learn_rate=cfg["learn_rate"])

if __name__ == "__main__":
    main()
