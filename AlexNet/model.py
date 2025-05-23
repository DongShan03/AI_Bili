import torch.nn as nn
import torch

class AlexNet(nn.Module):
    def __init__(self, num_classes=1000, init_weights=False):
        super().__init__()
        self.features = nn.Sequential(          #! [224, 224, 3]
            nn.Conv2d(3, 48, kernel_size=11, stride=4, padding=2),  #! [55, 55, 48]
            nn.ReLU(inplace=True),                  #? inplace=True 的作用增加计算量，使得模型承载力更大？
            nn.MaxPool2d(kernel_size=3, stride=2),                  #! [27, 27, 48]
            nn.Conv2d(48, 128, kernel_size=5, padding=2),           #! [27, 27, 128]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                  #! [13, 13, 128]
            nn.Conv2d(128, 192, kernel_size=3, padding=1),          #! [13, 13, 192]
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, padding=1),          #! [13, 13, 192]
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 128, kernel_size=3, padding=1),          #! [13, 13, 128]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                  #! [6, 6, 128]
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),                      #? Droupout常用于全连接层之间
            nn.Linear(6 * 6 * 128, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)   #* pytorch中数据为[batch, channel, height, weight]
                                            #* 这里flatten从1开始展开变为[batch, channel*height*weight]
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
