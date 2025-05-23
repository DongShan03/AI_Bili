import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x

class Inception(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        super().__init__()
        self.branch1 = BasicConv2d(in_channels, ch1x1, kernel_size=1)

        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, ch3x3red, kernel_size=1),
            BasicConv2d(ch3x3red, ch3x3, kernel_size=3, padding=1)
        )

        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, ch5x5red, kernel_size=1),
            BasicConv2d(ch5x5red, ch5x5, kernel_size=5, padding=2),
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_channels, pool_proj, kernel_size=1),
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs, 1)


class InceptionAux(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.average_pool = nn.AvgPool2d(kernel_size=5, stride=3)
        self.conv = BasicConv2d(in_channels, 128, kernel_size=1)    #! [batch, 128, 4, 4]
        self.fc1 = nn.Linear(128*4*4, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        #! aux1 : N x 512 x 14 x 14  aux2: N x 528 x 14 x 14
        x = self.average_pool(x)
        #! aux1 : N x 512 x 4 x 4  aux2: N x 528 x 4 x 4
        x = self.conv(x)
        #! N x 128 x 4 x 4
        x = torch.flatten(x, start_dim=1)
        x = F.dropout(x, 0.5, training=self.training)   #* self.training由model.train()控制
        x = F.relu(self.fc1(x), inplace=True)
        x = F.dropout(x, 0.5, training=self.training)
        x = self.fc2(x)
        return x

class GoogleNet(nn.Module):
    def __init__(self, num_classes=1000, aux_logits=True, init_weights=False):
        super().__init__()
        self.aux_logits = aux_logits
        #! N x 3 x 224 x 224
        self.conv1 = BasicConv2d(3, 64, kernel_size=7, stride=2, padding=3)
        #! N x 64 x 112 x 112
        self.maxpool1 = nn.MaxPool2d(3, stride=2, ceil_mode=True) #* ceil_mode=True向上取整 否则向下取整
        #! N x 64 x 56 x 56
        self.conv2 = BasicConv2d(64, 64, kernel_size=1)
        #! N x 64 x 56 x 56
        self.conv3 = BasicConv2d(64, 192, kernel_size=3, padding=1)
        #! N x 192 x 56 x 56
        self.maxpool2 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        #! N x 192 x 28 x 28

        self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32)
        #! N x 256 x 28 x 28
        self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64)
        #! N x 480 x 28 x 28
        self.maxpool3 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        #! N x 480 x 14 x 14

        self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64)
        #! N x (192+208+48+64) = 512 x 14 x 14
        self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64)
        #! N x (160+224+64+64) = 512 x 14 x 14
        self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64)
        #! N x (128+256+64+64) = 512 x 14 x 14
        self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64)
        #! N x (112+288+64+64) = 528 x 14 x 14
        self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128)
        #! N x (256+320+128+128) = 832 x 14 x 14
        self.maxpool4 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        #! N x 832 x 7 x 7

        self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128)
        #! N x (256+320+128+128) = 832 x 7 x 7
        self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128)
        #! N x (384+384+128+128) = 1024 x 7 x 7

        if self.aux_logits:
            #! input: N x 512 x 14 x 14 output: N x num_classes
            self.aux1 = InceptionAux(512, num_classes)  #* aux1接下inception4a后
            #! input: N x 528 x 14 x 14 output: N x num_classes
            self.aux2 = InceptionAux(528, num_classes)  #* aux2接在inception4d后

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))     #* 自适应平均池化
        #! N x 1024 x 1 x 1
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(1024, num_classes)
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.maxpool2(x)
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)
        x = self.inception4a(x)
        if self.training and self.aux_logits:
            aux1 = self.aux1(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        if self.training and self.aux_logits:
            aux2 = self.aux2(x)
        x = self.inception4e(x)
        x = self.maxpool4(x)
        x = self.inception5a(x)
        x = self.inception5b(x)
        x = self.avgpool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.dropout(x)
        x = self.fc(x)
        if self.training and self.aux_logits:
            return x, aux2, aux1
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
