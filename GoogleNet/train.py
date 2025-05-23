import torch.optim as optim
import torch.nn as nn
from model import GoogleNet
import os, time, torch
from cfg import *
from preprocess import get_data_loader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train(net_name, epochs=20, learn_rate=0.0002):
    train_num, validate_num, train_loader, validate_loader = get_data_loader()
    if os.path.exists(save_path):
        print(net_name + "模型已存在，继续训练！")
        net = GoogleNet(num_classes=num_classes, aux_logits=True, init_weights=False).to(device)
        net.load_state_dict(torch.load(save_path))
    else:
        print(net_name + "模型不存在")
        net = GoogleNet(num_classes=num_classes, aux_logits=True, init_weights=True).to(device)
    net.train()
    #* 从这准备训练
    loss_function = nn.CrossEntropyLoss()
    # pata = list(net.parameters())
    optimizer = optim.Adam(net.parameters(), lr=learn_rate)
    best_acc = 0.0

    for epoch in range(epochs):
        net.train()
        running_loss = 0.0
        t1 = time.perf_counter()
        for step, data in enumerate(train_loader, start=0):
            images, labels = data
            optimizer.zero_grad()
            logits, aux_logits2, aux_logits1 = net(images.to(device))
            loss0 = loss_function(logits, labels.to(device))
            loss1 = loss_function(aux_logits1, labels.to(device))
            loss2 = loss_function(aux_logits2, labels.to(device))
            loss = loss0 + loss1 * 0.3 + loss2 * 0.3
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            rate = (step + 1) / len(train_loader)
            a = "*" * int(rate * 50)
            b = "." * int((1 - rate) * 50)
            print("\rtrain loss: {:^3.0f}%[{}->{}]{:.3f}".format(int(rate*100), a, b, loss), end="")
        print()             #* print中\r表示回到行首
        print("Train time: %.3f s" % (time.perf_counter() - t1))

        net.eval()
        acc = 0.0
        with torch.no_grad():
            for data_test in validate_loader:
                test_images, test_labels = data_test
                outputs = net(test_images.to(device))
                predict_y = torch.max(outputs, 1)[1]
                acc += (predict_y == test_labels.to(device)).sum().item()
            accurate_test = acc / validate_num
            if accurate_test > best_acc:
                best_acc = accurate_test
                torch.save(net.state_dict(), save_path)
            print('[epoch %d] train_loss: %.3f  test_accuracy: %.3f' %
                (epoch + 1, running_loss / step, acc / validate_num))

    print("Finished Training!")

def main():
    train(net_name, epochs=epochs, learn_rate=learn_rate)

if __name__ == "__main__":
    main()
