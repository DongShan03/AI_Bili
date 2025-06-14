import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from copy import deepcopy
import contextlib
from YOLOv5.utils.utils import read_yaml, make_divisible
from YOLOv5.model.layers import *
from YOLOv5.utils.autoanchor import check_anchor_order
from YOLOv5.utils.image_utils import scale_img

def create_yolov5_model(model_config, input_channel):
    #* anchor模板, 类别数, 深度倍数, 宽度倍数, 激活函数, 通道倍数
    anchors, nc, gd, gw, act, ch_mul = (
        model_config["anchors"],
        model_config["nc"],
        model_config["depth_multiple"],
        model_config["width_multiple"],
        model_config.get("activation"),
        model_config.get("channel_multiple"),
    )
    if act:
        Conv.default_act = eval(act)
    if not ch_mul:
        ch_mul = 8

    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors
    no = na * (nc + 5)

    layers, save, out_channel = [], [], input_channel[-1]
    #* [from, number, module, args]
    for i, (f, number, module, args) in enumerate(model_config["backbone"] + model_config["head"]):
        module = eval(module) if isinstance(module, str) else module
        for j, a in enumerate(args):
            with contextlib.suppress(NameError):
                args[j] = eval(a) if isinstance(a, str) else a

        number = max(round(number * gd), 1) if number > 1 else 1
        if module in (
            Conv,
            GhostConv,
            Bottleneck,
            GhostBottleneck,
            SPP,
            SPPF,
            DWConv,
            MixConv2d,
            Focus,
            CrossConv,
            BottleneckCSP,
            C3,
            C3TR,
            C3SPP,
            C3Ghost,
            nn.ConvTranspose2d,
            DWConvTranspose2d,
            C3x,
        ):
            in_channel, out_channel = input_channel[f], args[0]
            if out_channel != no:
                out_channel = make_divisible(out_channel * gw, ch_mul)
            args = [in_channel, out_channel, *args[1:]]
            if module in (BottleneckCSP, C3, C3TR, C3SPP, C3Ghost, C3x):
                args.insert(2, number)
                number = 1
        elif module is nn.BatchNorm2d:
            args = [input_channel[f]]
        elif module is Concat:
            out_channel = sum(input_channel[x] for x in f)
        elif module in {Detect, Segment}:
            args.append([input_channel[x] for x in f])
            if isinstance(args[1], int):
                args[1] = [list(range(args[1] * 2))] * len(f)
            if module is Segment:
                args[3] = make_divisible(args[3] * gw, ch_mul)
        elif module is Contract:
            out_channel = input_channel[f] * args[0] ** 2
        elif module is Expand:
            out_channel  = input_channel[f] // args[0] ** 2
        else:
            out_channel = input_channel[f]

        module_ = nn.Sequential(*(module(*args) for _ in range(number))) if number > 1 else module(*args)
        module_type = str(module)[8:-2].replace("__main__", "")
        np = sum(x.numel() for x in module_.parameters())
        module_.i, module_.f, module_.type, module_.np = i, f, module_type, np
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)
        layers.append(module_)
        if i == 0:
            input_channel = []
        input_channel.append(out_channel)
    return nn.Sequential(*layers), sorted(save)

class BaseModel(nn.Module):
    #* YOLOv5 base model
    def forward(self, x):
        if not self.training:
            self.fuse()
        return self._forward_once(x)

    def _forward_once(self, x):
        y, dt = [], []
        for m in self.model:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]

            x = m(x)
            y.append(x if m.i in self.save else None)

        return x

    def fuse(self):
        for m in self.model.modules():
            if isinstance(m, (Conv, DWConv)) and hasattr(m, "bn"):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, "bn")  # remove batchnorm
                m.forward = m.forward_fuse  # update forward
        return self

    def _apply(self, fn):
        self = super()._apply(fn)
        m = self.model[-1]
        if isinstance(m, (Detect, Segment)):
            m.stride = fn(m.stride)
            m.grid = list(map(fn, m.grid))
            if isinstance(m.anchor_grid, list):
                m.anchor_grid = list(map(fn, m.anchor_grid))
        return self

class DetectionModel(BaseModel):
    def __init__(self, cfg="yolov5l.yaml", input_channel=3, nclasses=None, anchors=None):
        super().__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg
        else:
            self.yaml = read_yaml(cfg)

        input_channel = self.yaml["ch"] = self.yaml.get("ch", input_channel)
        if nclasses and nclasses != self.yaml["nc"]:
            self.yaml["nc"] = nclasses
        if anchors:
            self.yaml["anchors"] = round(anchors)

        self.model, self.save = create_yolov5_model(deepcopy(self.yaml), [input_channel])
        self.names = [str(i) for i in range(self.yaml["nc"])]
        self.inplace = self.yaml.get("inplace", True)

        m = self.model[-1]  #* Detect()
        if isinstance(m, (Detect, Segment)):
            def _forward(x):
                return self.forward(x, augment=False)[0] if isinstance(m, Segment) else self.forward(x, augment=False)

            s = 256
            m.inplace = self.inplace
            m.stride = torch.tensor([s / x.shape[-2] for x in _forward(torch.zeros(1, input_channel, s, s))])
            check_anchor_order(m)
            m.anchors /= m.stride.view(-1, 1, 1)
            self.stride = m.stride
            self._initialize_biases()  # only run once

        # Init weights, biases
        initialize_weights(self)

    def forward(self, x, augment=False):
        if augment:
            #! 不可以启用
            return self._forward_augment(x)
        return self._forward_once(x)

    def _forward_augment(self, x):
        img_size = x.shape[-2:]
        s = [1, 0.83, 0.67]  #* scales
        f = [None, 3, None]  #* flips (2-ud, 3-lr)
        y = []
        for si, fi in zip(s, f):
            #* 这里图像大小改变了
            #! x -> [B, 3, 640, 640]
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            #! yi -> [B, 3, H, W, 85]
            yi = self._forward_once(xi)[0]
            yi = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)
        y = self._clip_augmented(y)
        return torch.cat(y, 1), None

    def _descale_pred(self, pred, flips, scale, img_size):
        if self.inplace:    #? 节省内存
            pred[..., :4] /= scale
            if flips == 2:
                pred[..., 1] = img_size[0] - pred[..., 1]
            elif flips == 3:
                pred[..., 0] = img_size[1] - pred[..., 0]
        else:
            x, y, wh = pred[..., 0:1] / scale, pred[..., 1:2] / scale, pred[..., 2:4] / scale
            if flips == 2:
                y = img_size[0] - y
            elif flips == 3:
                x = img_size[1] - x
            pred = torch.cat((x, y, wh, pred[..., 4:]), -1)
        return pred

    def _clip_augmented(self, y):
        #! 这里有bug 源码也跑不了 ?
        #* affecting first and last tensors based on grid points and layer counts
        #* nl of detect layer = 3 （P3, P4, P5）
        nl = self.model[-1].nl
        #* g = sum(4**0, 4**1, 4**2) = 21
        #* 每一次下采样后长宽变为1/2 故上一层的grid点数为下一层的4倍
        g = sum(4**x for x in range(nl))
        #* 排除了中间一个yi
        e = 1
        #* y[0] -> [B, 3, H, W, 85] i = 3 // 21 * sum(4**0) = 0 * 1 = 0
        i = (y[0].shape[1] // g) * sum(4**x for x in range(e))
        #* y[0] -> [B, 0, ny, nx]
        y[0] = y[0][:, :-i] #* large
        #* y[-1] -> [B, 3, H, W, 85] i = 3 // 21 * sum(4**2) = 0
        i = (y[-1].shape[1] // g) * sum(4 ** (nl - 1 - x) for x in range(e))
        #* y[-1] -> [B, 255 - 192, ny, nx]
        y[-1] = y[-1][:, i:]    #* small
        return y


    def _initialize_biases(self, cf=None):
        m = self.model[-1]
        #* Detect.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in channels)
        for mi, s in zip(m.m, m.stride):
            #* 255 -> 3, 85
            b = mi.bias.view(m.na, -1)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)
            b.data[:, 5: 5 + m.nc] += (
                math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())
            )
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

Model = DetectionModel

class SegmentationModel(DetectionModel):
    def __init__(self, cfg="yolov5s-seg.yaml", input_channel=3, nclasses=None, anchors=None):
        super().__init__(cfg, input_channel, nclasses, anchors)

class ClassificationModel(BaseModel):
    def __init__(self, cfg=None, model=None, nclasses=1000, cutoff=10):
        super().__init__()
        self._from_detection_model(model, nclasses, cutoff) if model is not None else self._from_yaml(cfg)

    def _from_detection_model(self, model, nc=1000, cutoff=10):
        model.model = model.model[:cutoff]  # backbone
        m = model.model[-1]  # last layer
        ch = m.conv.in_channels if hasattr(m, "conv") else m.cv1.conv.in_channels
        c = Classify(ch, nc)  # Classify()
        c.i, c.f, c.type = m.i, m.f, "models.common.Classify"  # index, from, type
        model.model[-1] = c  # replace
        self.model = model.model
        self.stride = model.stride
        self.save = []
        self.nc = nc

    def _from_yaml(self, cfg):
        """Creates a YOLOv5 classification model from a specified *.yaml configuration file."""
        model_config = read_yaml(cfg)
        self.model, _ = create_yolov5_model(deepcopy(cfg), [3])
        self.stride = self.model.stride
        self.save = []

if __name__ == "__main__":

    x = torch.randn(1, 3, 640, 640)
    model = DetectionModel()
    model(x)
