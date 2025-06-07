import os, sys
sys.path.append(os.path.dirname(__file__))
from res50_backbone import resnet50
from utils import dboxes300_coco, Encoder, PostProcess, Loss
import torch
from torch import nn
from torch.jit.annotations import List


class backBone(nn.Module):
    def __init__(self, pretrain_path=None):
        super().__init__()
        net = resnet50()
        self.out_channels = [1024, 512, 512, 256, 256, 256]
        if pretrain_path is not None:
            net.load_state_dict(torch.load(pretrain_path))
        #* list(net.children())[:7] -> conv1 bn1 relu maxpool layer1 layer2 layer3
        self.feature_extractor = nn.Sequential(*list(net.children())[:7])
        conv4_block1 = self.feature_extractor[-1][0]
        conv4_block1.conv1.stride = (1, 1)
        conv4_block1.conv2.stride = (1, 1)
        conv4_block1.downsample[0].stride = (1, 1)

    def forward(self, x):
        return self.feature_extractor(x)


class SSD300(nn.Module):
    def __init__(self, backbone=None, num_classes=21):
        super().__init__()
        if backbone is None:
            raise Exception("backbone is None")
        if not hasattr(backbone, "out_channels"):
            raise Exception("backbone out_channels is None")
        self.feature_extractor = backbone
        self.num_classes = num_classes

        self._build_addition_features(self.feature_extractor.out_channels)
        self.num_defaults = [4, 6, 6, 6, 4, 4]
        location_extractor = []
        confidence_extractor = []

        for nd, oc in zip(self.num_defaults, self.feature_extractor.out_channels):
            """
                解释这里为什么是nd * 4
                首先oc对应特征层的通道数,nd是每个特征层的默认框数,而每个特征框需要四个参数
                比如输入特征层为38x38x1024,那么经过nn.Conv2d变为38x38x(nd x 4or6)

                confidence_extractor 对应的是分类置信度,num_classes是21=20+1
            """
            location_extractor.append(nn.Conv2d(oc, nd * 4, kernel_size=3, padding=1))
            confidence_extractor.append(nn.Conv2d(oc, nd * self.num_classes, kernel_size=3, padding=1))
        self.loc = nn.ModuleList(location_extractor)
        self.conf = nn.ModuleList(confidence_extractor)
        self._init_weights()

        default_boxes = dboxes300_coco()
        self.compute_loss = Loss(default_boxes)
        self.encoder = Encoder(default_boxes)
        self.postprocess = PostProcess(default_boxes)

    def _build_addition_features(self, input_channels: list):
        additional_blocks = []
        middle_channels = [256, 256, 128, 128, 128]
        for i, (input_channel, middle_channel, outchannel) in \
            enumerate(zip(input_channels[:-1], middle_channels, input_channels[1:])):
            padding, stride = (1, 2) if i < 3 else (0, 1)
            additional_blocks.append(
                nn.Sequential(
                    nn.Conv2d(input_channel, middle_channel, kernel_size=1,
                            stride=1, padding=0, bias=False),
                    nn.BatchNorm2d(middle_channel),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(middle_channel, outchannel, kernel_size=3,
                            stride=stride, padding=padding, bias=False),
                    nn.BatchNorm2d(outchannel),
                    nn.ReLU(inplace=True),
                )
            )
        self.additional_blocks = nn.ModuleList(additional_blocks)

    def _init_weights(self):
        layers = [*self.additional_blocks, *self.loc, *self.conf]
        for m in layers:
            for param in m.parameters():
                if param.dim() > 1:
                    nn.init.xavier_uniform_(param)

    def bbox_view(self, detection_features, loc, conf):
        locs, confs = [], []
        for feature, l, c in zip(detection_features, loc, conf):
            #! [batch, nd*4, feature_h, f_w] -> [batch, 4, nd*f_h*f_w]
            locs.append(l(feature).view(feature.size(0), 4, -1))
            #! [batch, nd*self.num_classes, f_h, f_w] -> [batch, self.num_classes, nd*f_h*f_w]
            confs.append(c(feature).view(feature.size(0), self.num_classes, -1))
        #! 38x38x4 + 19x19x6 + 10x10x6 + 5x5x6 + 3x3x4 + 1x1x4 = 8732
        #! 在dim=2上进行拼接 把各个特征层上的预测结果合并到一起
        #! locs -> [batch, 4, 8732]  confs -> [batch, self.num_classes, 8732]
        locs, confs = torch.cat(locs, 2).contiguous(), torch.cat(confs, 2).contiguous()
        return locs, confs

    def forward(self, images, targets=None):
        x = self.feature_extractor(images)
        #! detection_features用来存储各个特征层
        detection_features = torch.jit.annotate(List[torch.Tensor], [])
        detection_features.append(x)
        for layer in self.additional_blocks:
            x = layer(x)
            detection_features.append(x)
        locs, confs = self.bbox_view(detection_features, self.loc, self.conf)

        if self.training:
            if targets is None:
                raise ValueError("If training, targets should not be none")
            #! targets["boxes"] -> [batch, num_boxes, 4=(gx1, gy1, gx2, gy2)]   g**代表真实
            bboxes_out = targets["boxes"]
            #! bboxes_out -> [batch, 4=(gx1, gy1, gx2, gy2), num_boxes]
            bboxes_out = bboxes_out.transpose(1, 2).contiguous()
            #! locs -> [batch, 4=(px1, py1, px2, py2), 8732]    p**代表预测
            #! confs -> [batch, self.num_classes, 8732]
            #! labels_out -> [batch, -1]
            labels_out = targets["labels"]
            #* ploc, plabel, gloc, glabel
            loss = self.compute_loss(locs, confs, bboxes_out, labels_out)
            return {"total_losses": loss}
        results = self.postprocess(locs, confs)
        return results
