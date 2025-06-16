import torch
import os
import numpy as np
from copy import deepcopy
import math
import yaml
import torch.nn as nn
import torch.nn.functional as F

def read_yaml(yaml_file):
    if not yaml_file.endswith(".yaml"): yaml_file += ".yaml"
    yaml_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model_cfg", yaml_file)
    with open(yaml_file) as f:
        hyp = yaml.load(f, Loader=yaml.FullLoader)
    return hyp

def make_divisible(x, divisor):
    return math.ceil(x / divisor) * divisor

def check_anchor_order(m):
    """Checks and corrects anchor order against stride in YOLOv5 Detect() module if necessary."""
    a = m.anchors.prod(-1).mean(-1).view(-1)  # mean anchor area per output layer
    da = a[-1] - a[0]  # delta a
    ds = m.stride[-1] - m.stride[0]  # delta s
    if da and (da.sign() != ds.sign()):  # same order
        m.anchors[:] = m.anchors.flip(0)

def scale_img(img, ratio=1.0, same_shape=False, gs=32):  # img(16,3,256,416)
    """Scales an image tensor `img` of shape (bs,3,y,x) by `ratio`, optionally maintaining the original shape, padded to
    multiples of `gs`.
    """
    if ratio == 1.0:
        return img
    h, w = img.shape[2:]
    s = (int(h * ratio), int(w * ratio))  # new size
    img = F.interpolate(img, size=s, mode="bilinear", align_corners=False)  # resize
    if not same_shape:  # pad/crop img
        h, w = (math.ceil(x * ratio / gs) * gs for x in (h, w))
    return F.pad(img, [0, w - s[1], 0, h - s[0]], value=0.447)  # value = imagenet mean

def xyxy2xywh(x):
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y

def xywh2xyxy(x):
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y

def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False):
    box2 = box2.t()
    if x1y1x2y2:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    #* Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1
    union = (w1 * h1 + 1e-16) + w2 * h2 - inter

    iou = inter / union
    if GIoU or DIoU or CIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)
        if GIoU:
            c_area = cw * ch + 1e-16
            return iou - (c_area - union) / c_area
        elif DIoU or CIoU:
            #* 凸包面积的平方
            c2 = cw ** 2 + ch ** 2 + 1e-16
            #* 中心点的距离的平方
            rho2 = ((b2_x1 + b2_x2) - (b1_x1 + b1_x2)) ** 2 / 4 + \
                    ((b2_y1 + b2_y2) - (b1_y1 + b1_y2)) ** 2 / 4
            if DIoU:
                return iou - rho2 / c2

            elif CIoU:
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / (1 - iou + v)
                return iou - (rho2 / c2 + v * alpha)
    return iou


class ModelEMA:
    """ Model Exponential Moving Average from https://github.com/rwightman/pytorch-image-models
    Keep a moving average of everything in the model state_dict (parameters and buffers).
    This is intended to allow functionality like
    https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    A smoothed version of the weights is necessary for some training schemes to perform well.
    E.g. Google's hyper-params for training MNASNet, MobileNet-V3, EfficientNet, etc that use
    RMSprop with a short 2.4-3 epoch decay period and slow LR decay rate of .96-.99 requires EMA
    smoothing of weights to match results. Pay attention to the decay constant you are using
    relative to your update count per epoch.
    To keep EMA from using GPU resources, set device='cpu'. This will save a bit of memory but
    disable validation of the EMA weights. Validation will have to be done manually in a separate
    process, or after the training stops converging.
    This class is sensitive where it is initialized in the sequence of model init,
    GPU assignment and distributed training wrappers.
    I've tested with the sequence in my own train.py for torch.DataParallel, apex.DDP, and single-GPU.
    """

    def __init__(self, model, decay=0.9999, device=''):
        # make a copy of the model for accumulating moving average of weights
        self.ema = deepcopy(model)
        self.ema.eval()
        self.updates = 0  # number of EMA updates
        self.decay = lambda x: decay * (1 - math.exp(-x / 2000))  # decay exponential ramp (to help early epochs)
        self.device = device  # perform ema on different device from model if set
        if device:
            self.ema.to(device=device)
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        self.updates += 1
        d = self.decay(self.updates)
        with torch.no_grad():
            if type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel):
                msd, esd = model.module.state_dict(), self.ema.module.state_dict()
            else:
                msd, esd = model.state_dict(), self.ema.state_dict()

            for k, v in esd.items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1. - d) * msd[k].detach()

    def update_attr(self, model):
        # Assign attributes (which may change during training)
        for k in model.__dict__.keys():
            if not k.startswith('_'):
                setattr(self.ema, k, getattr(model, k))

def labels_to_class_weights(labels, nc=80):
    if labels[0] is None:
        return torch.Tensor()
    labels = np.concatenate(labels, 0)
    classes = labels[:, 0].astype(np.int32)
    weights = np.bincount(classes, minlength=nc)

    weights[weights == 0] = 1
    weights = 1 / weights
    weights /= weights.sum()
    return torch.from_numpy(weights)

def labels_to_image_weights(labels, nc=80, class_weights=np.ones(80)):
    n = len(labels)
    class_counts = np.array([np.bincount(labels[i][:, 0].astype(np.int32), minlength=nc) for i in range(n)])
    image_weights = (class_weights.reshape(1, nc) * class_counts).sum(1)
    return image_weights

def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    if ratio_pad is None:
        gain = max(img1_shape) / max(img0_shape)
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]
    coords[:, [1, 3]] -= pad[1]
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords

def clip_coords(boxes, img_shape):
    boxes[:, 0].clamp_(0, img_shape[1])
    boxes[:, 1].clamp_(0, img_shape[0])
    boxes[:, 2].clamp_(0, img_shape[1])
    boxes[:, 3].clamp_(0, img_shape[0])
