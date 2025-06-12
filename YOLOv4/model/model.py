import os, sys, math
import numpy as np
from pathlib import Path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from YOLOv4.model.layers import (MixConv2d, Swish, GAP, Silence, ScaleChannel, \
        ScaleSpatial, FeatureConcat, FeatureConcat2, FeatureConcat3, FeatureConcat_l, \
        YOLOLayer, WeightedFeatureFusion, Reorg, JDELayer, get_yolo_layers, \
        fuse_conv_and_bn, DeformConv2d)
from YOLOv4.utils.parse_config import parse_model_cfg
from YOLOv4.utils.image_transform import scale_img
import torch
import torch.nn as nn
import torch.nn.functional as F

def create_modules(module_defs, img_size, cfg):

    img_size = [img_size] * 2 if isinstance(img_size, int) else img_size
    _ = module_defs.pop(0)
    out_filters = [3]   #* 输入Channel
    module_list = nn.ModuleList()
    routs = []
    yolo_index = -1

    for i, mdef in enumerate(module_defs):
        modules = nn.Sequential()

        if mdef["type"] == "convolutional":
            bn = mdef["batch_normalize"]
            filters = mdef["filters"]
            ksize = mdef["size"]
            stride = mdef["stride"] if 'stride' in mdef else (mdef["stride_y"], mdef["stride_x"])
            if isinstance(ksize, int):
                modules.add_module(
                    "Conv2d", nn.Conv2d(in_channels=out_filters[-1],
                                        out_channels=filters,
                                        kernel_size=ksize,
                                        stride=stride,
                                        padding=ksize // 2 if mdef['pad'] else 0,
                                        groups=mdef["groups"] if 'groups' in mdef else 1,
                                        bias=not bn)
                )
            else:
                modules.add_module(
                    'MixConv2d', MixConv2d(in_channel=out_filters[-1],
                                        out_channel=filters,
                                        kernel_size=ksize,
                                        stride=stride,
                                        bias=not bn)
                )

            if bn:
                modules.add_module("BatchNorm2d", nn.BatchNorm2d(filters, momentum=0.03, eps=1E-4))
            else:
                #* 如果卷积操作不接bn层，意味着该层为yolo的predictor
                routs.append(i)

            if mdef["activation"] == "leaky":
                modules.add_module("activation", nn.LeakyReLU(0.1, inplace=True))
            elif mdef["activation"] == "swish":
                modules.add_module("activation", Swish())
            elif mdef["activation"] == "mish":
                modules.add_module("activation", nn.Mish())
            elif mdef['activation'] == 'emb':
                modules.add_module("activation", F.normalize())
            elif mdef['activation'] == 'logistic':
                modules.add_module('activation', nn.Sigmoid())
            elif mdef['activation'] == 'silu':
                modules.add_module('activation', nn.SiLU())

        elif mdef["type"] == "deformableconvolutional":
            bn = mdef["batch_normalize"]
            filters = mdef["filters"]
            ksize = mdef["size"]
            stride = mdef["stride"] if 'stride' in mdef else (mdef["stride_y"], mdef["stride_x"])
            if isinstance(ksize, int):
                modules.add_module(
                    "DeformConv2d", DeformConv2d(
                        in_channel=out_filters[-1],
                        out_channel=filters,
                        kernel_size=ksize,
                        padding=ksize // 2 if mdef['pad'] else 0,
                        stride=stride,
                        bias=not bn,
                        modulation=True
                    )
                )
            else:  # multiple-size conv
                modules.add_module(
                    'MixConv2d', MixConv2d(
                        in_channel=out_filters[-1],
                        out_channel=filters,
                        k=ksize,
                        stride=stride,
                        bias=not bn
                    )
                )
            if bn:
                modules.add_module("BatchNorm2d", nn.BatchNorm2d(filters, momentum=0.03, eps=1E-4))
            else:
                routs.append(i)

            if mdef["activation"] == "leaky":
                modules.add_module("activation", nn.LeakyReLU(0.1, inplace=True))
            elif mdef["activation"] == "swish":
                modules.add_module("activation", Swish())
            elif mdef["activation"] == "mish":
                modules.add_module("activation", nn.Mish())
            elif mdef['activation'] == 'silu':
                modules.add_module('activation', nn.SiLU())

        elif mdef["type"] == 'dropout':
            p = mdef["probability"]
            modules = nn.Dropout(p)

        elif mdef["type"] == 'avgpool':
            modules = GAP()

        elif mdef["type"] == 'silence':
            filters = out_filters[-1]
            modules = Silence()

        elif mdef["type"] == "scale_channels":
            layers = mdef["from"]
            filters = out_filters[-1]
            routs.extend([i + l if l < 0 else l for l in layers])
            modules = ScaleChannel(layers)

        elif mdef["type"] == "sam":
            layers = mdef["from"]
            filters = out_filters[-1]
            routs.extend([i + l if l < 0 else l for l in layers])
            modules = ScaleSpatial(layers)

        elif mdef['type'] == 'BatchNorm2d':
            filters = out_filters[-1]
            modules = nn.BatchNorm2d(filters, momentum=0.03, eps=1E-4)
            if i == 0 and filters == 3:
                modules.running_mean = torch.tensor([0.485, 0.456, 0.406])
                modules.running_var = torch.tensor([0.0524, 0.0502, 0.0506])

        elif mdef["type"] == "maxpool":
            ksize = mdef["size"]
            stride = mdef["stride"]
            maxpool = nn.MaxPool2d(kernel_size=ksize, stride=stride, padding=(ksize - 1) // 2)
            if ksize == 2 and stride == 1:   # yolov3-tiny
                modules.add_module("ZeroPad2d", nn.ZeroPad2d((0, 1, 0, 1)))
                modules.add_module("MaxPool2d", maxpool)
            else:
                modules = maxpool

        elif mdef["type"] == 'local_avgpool':
            ksize = mdef["size"]
            stride = mdef["stride"]
            avgpool = nn.AvgPool2d(kernel_size=ksize, stride=stride, padding=(ksize - 1) // 2)
            if ksize == 2 and stride == 1:   # yolov3-tiny
                modules.add_module("ZeroPad2d", nn.ZeroPad2d((0, 1, 0, 1)))
                modules.add_module("AvgPool2d", avgpool)
            else:
                modules = avgpool

        elif mdef["type"] == "upsample":
            modules = nn.Upsample(scale_factor=mdef["stride"])

        elif mdef["type"] == "route":
            layers = mdef["layers"]
            filters = sum([out_filters[l + 1 if l > 0 else l] for l in layers])
            routs.extend([i + l if l < 0 else l for l in layers])
            modules = FeatureConcat(layers)

        elif mdef["type"] == 'route2':
            layers = mdef["layers"]
            filters = sum([out_filters[l + 1 if l > 0 else l] for l in layers])
            routs.extend([i + l if l < 0 else l for l in layers])
            modules = FeatureConcat2(layers)

        elif mdef["type"] == 'route3':
            layers = mdef["layers"]
            filters = sum([out_filters[l + 1 if l > 0 else l] for l in layers])
            routs.extend([i + l if l < 0 else l for l in layers])
            modules = FeatureConcat3(layers)

        elif mdef["type"] == 'route_lhalf':
            layers = mdef["layers"]
            filters = sum([out_filters[l + 1 if l > 0 else l] for l in layers])
            routs.extend([i + l if l < 0 else l for l in layers])
            modules = FeatureConcat_l(layers)

        elif mdef["type"] == "shortcut":
            layers = mdef["from"]
            filters = out_filters[-1]
            routs.extend([i + l if l < 0 else l for l in layers])
            modules = WeightedFeatureFusion(layers=layers, weight='weights_type' in mdef)

        elif mdef['type'] == 'reorg3d':  # yolov3-spp-pan-scale
            pass

        elif mdef['type'] == 'reorg':  # yolov3-spp-pan-scale
            filters = 4 * out_filters[-1]
            modules.add_module('Reorg', Reorg())

        elif mdef["type"] == "yolo":
            yolo_index += 1
            stride = [8, 16, 32, 64, 128]
            if any(x in cfg for x in ['yolov4-tiny', 'fpn', 'yolov3']):
                stride = [32, 16, 8]
            layers = mdef["from"] if "from" in mdef else []
            modules = YOLOLayer(
                anchors=mdef["anchors"][mdef["mask"]],
                nc=mdef["classes"],
                img_size=img_size,
                yolo_index=yolo_index,
                layers=layers,
                stride=stride[yolo_index]
            )
            try:
                j = layers[yolo_index] if "from" in mdef else -1
                #* bias_ -> [255,]
                bias_ = module_list[j][0].bias
                #* bias -> [3, 85]
                bias = bias_[:modules.no * modules.na].view(modules.na, -1)
                bias.data[:, 4] += math.log(8 / (640 / stride[yolo_index]) ** 2)
                bias.data[:, 5:] += math.log(0.6 / (modules.nc - 0.99))
                module_list[j][0].bias = torch.nn.Parameter(bias_, requires_grad=bias_.requires_grad)
            except:
                print('WARNING: smart bias initialization failure.')

        elif mdef["type"] == 'jde':
            yolo_index += 1
            stride = [8, 16, 32, 64, 128]
            if any(x in cfg for x in ['yolov4-tiny', 'fpn', 'yolov3']):
                stride = [32, 16, 8]
            layers = mdef["from"] if "from" in mdef else []
            modules = JDELayer(
                anchors=mdef["anchors"][mdef["mask"]],
                nc=mdef["classes"],
                img_size=img_size,
                yolo_index=yolo_index,
                layers=layers,
                stride=stride[yolo_index]
            )
            try:
                j = layers[yolo_index] if 'from' in mdef else -1
                bias_ = module_list[j][0].bias  # shape(255,)
                bias = bias_[:modules.no * modules.na].view(modules.na, -1)  # shape(3,85)
                #bias[:, 4] += -4.5  # obj
                bias.data[:, 4] += math.log(8 / (640 / stride[yolo_index]) ** 2)  # obj (8 objects per 640 image)
                bias.data[:, 5:] += math.log(0.6 / (modules.nc - 0.99))  # cls (sigmoid(p) = 1/nc)
                module_list[j][0].bias = torch.nn.Parameter(bias_, requires_grad=bias_.requires_grad)
            except:
                print('WARNING: smart bias initialization failure.')

        else:
            print('Warning: Unrecognized Layer Type: ' + mdef['type'])

        module_list.append(modules)
        out_filters.append(filters)

    routs_binary = [False] * (i + 1)
    for i in routs:
        routs_binary[i] = True

    return module_list, routs_binary


class Darknet(nn.Module):
    #* cfg = "yolov4.cfg"
    def __init__(self, cfg:str="yolov4.cfg", img_size=(416, 416)):
        super().__init__()
        self.module_defs = parse_model_cfg(cfg_name=cfg)
        self.module_list, self.routs = create_modules(self.module_defs, img_size=img_size, cfg=cfg)
        self.yolo_layers = get_yolo_layers(self)
        if not self.training:
            self.fuse()
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            t = type(m)
            if t is nn.Conv2d:
                pass  # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif t is nn.BatchNorm2d:
                m.eps = 1e-3
                m.momentum = 0.03
            elif t in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6]:
                m.inplace = True

    def forward(self, x, augment=False):
        if not augment:
            return self.forward_once(x)
        else:
            #* img_size = (416, 416)
            img_size = x.shape[-2:]
            s = [0.83, 0.67]
            y = []
            for i, xi in enumerate(
                #* x1 = x, x2是在x通道3处反转(也就是左右翻转)通道并且缩放尺寸到原先的0.83倍,并填充到32的整数倍 x3同理
                #* x -> [B, inC, H, W]
                (x, scale_img(x.flip(3), s[0], same_shape=False), scale_img(x, s[1], same_shape=False))
            ):
                #* 所以这里每张图片进来都会变成3张 并且结果会放到y中
                y.append(self.forward_once(xi)[0])
            #* 这里是缩放回原先的尺寸
            y[1][..., :4] /= s[0]
            #* 这里是左右翻转回原先的样子
            y[1][..., 0] = img_size[1] - y[1][..., 0]
            y[2][..., :4] /= s[1]
            #* 拼接成[B, 3*outC, H, W]
            y = torch.cat(y, 1)
            return y, None

    def forward_once(self, x, augment=False):
        img_size = x.shape[-2:]
        yolo_out, out = [], []
        #* 和之前的一样
        if augment:
            nb = x.shape[0]
            s = [0.83, 0.67]
            x = torch.cat(
                (x, scale_img(x.flip(3), s[0]), scale_img(x, s[1])), 0
            )
        #* 遍历所有结构
        #* 对于部分结构需要之前的输出 所有会引进out然后根据索引找到对应的输出
        for i, module in enumerate(self.module_list):
            name = module.__class__.__name__
            if name in ['WeightedFeatureFusion', 'FeatureConcat', 'FeatureConcat2', 'FeatureConcat3', 'FeatureConcat_l', 'ScaleChannel', 'ScaleSpatial']:
                x = module(x, out)
            elif name == 'YOLOLayer':
                yolo_out.append(module(x, out))
            elif name == 'JDELayer':
                yolo_out.append(module(x, out))
            else:
                x = module(x)
            #* routs[i] = True 说明后续会有结构需要该层的输出
            out.append(x if self.routs[i] else [])

        if self.training:
            return yolo_out
        else:
            #* 非训练模式 也就是推理预测
            #* yolo_out 会返回元组 -> ([B, -1, 85], [B, num_anchors, (1 + 4 + num_classes), H, W])
            x, p = zip(*yolo_out)
            #* 将每个特征层输出的x拼接到一起
            #* x -> [B, outC1 + outC2 + outC3, 85] outC1 = num_anchors(3) * H1 * W1
            x = torch.cat(x, 1)
            if augment:
                #* 数据增强了的话反转回原来的样子
                x = torch.split(x, nb, dim=0)
                x[1][..., :4] /= s[0]
                x[1][..., 0] = img_size[1] - x[1][..., 0]
                x[2][..., :4] /= s[1]
                x = torch.cat(x, 1)
            #* x -> [B, outC1 + outC2 + outC3, 85]
            #* p -> [B, num_anchors, (1 + 4 + num_classes), H, W]
            return x, p

    def fuse(self):
        print('Fusing layers...')
        fused_list = nn.ModuleList()
        for a in list(self.children())[0]:
            if isinstance(a, nn.Sequential):
                for i, b in enumerate(a):
                    if isinstance(b, nn.modules.batchnorm.BatchNorm2d):
                        conv = a[i - 1]
                        fused = fuse_conv_and_bn(conv, b)
                        a = nn.Sequential(fused, *list(a.children())[i + 1:])
                        break
            fused_list.append(a)
        self.module_list = fused_list
        print('Fusing finish.')
