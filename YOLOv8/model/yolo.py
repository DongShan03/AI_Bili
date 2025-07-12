import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from YOLOv8.model.layers import *
from YOLOv8.utils.utils import *
from pathlib import Path
import logging
logger = logging.getLogger(__name__)
from copy import deepcopy

class DFL(nn.Module):
    def __init__(self, c1=16):
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, kernel_size=1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        #* batch_size, self.reg_max * 4, 8400
        b, c, a = x.shape
        #* batch_size, self.reg_max * 4, 8400 -> batch_size, self.reg_max, 4, 8400 -> b, 4, 8400
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)


class Detect(nn.Module):
    def __init__(self, nc: int = 80, ch: tuple = ()):
        super().__init__()
        self.nc = nc
        self.nl = len(ch)
        self.reg_max = 16   # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.no = nc + self.reg_max * 4 #* 每个anchor对用的输出
        self.stride = torch.zeros(self.nl)
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 1000))
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch
        )

        self.cv3 = nn.ModuleList(
            nn.Sequential(
                nn.Sequential(DWConv(x, x, 3), Conv(x, c3, 1)),
                nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)),
                nn.Conv2d(c3, self.nc, 1),
            )
            for x in ch
        )
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def forward(self, x):
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
            #* x[i] -> (B, nc + self.reg_max * 4, H, W)
        if self.training:
            return x
        y = self._inference(x)
        return (y, x)

    def _inference(self, x):
        #* shape -> B, C, H, W
        shape = x[0].shape
        #* for each layer: xi(B, C, H, W) -> (B, nc + self.reg_max * 4, H*W)
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)

        self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
        self.shape = shape
        box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)

        dbox = self.decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0)) * self.strides

        return torch.cat((dbox, cls.sigmoid()), 1)

    def decode_bboxes(self, bboxes, anchors, xywh):
        #* 将距离坐标转换为真实坐标
        return dist2bbox(bboxes, anchors, xywh, dim=1)

    def bias_init(self):
        m = self
        for a, b, s in zip(m.cv2, m.cv3, m.stride):
            a[-1].bias.data[:] = 1.0
            b[-1].bias.data[:m.nc] = math.log(5 / m.nc / (640 / s) ** 2)

    @staticmethod
    def postprocess(preds, max_det, nc=80):
        #* max_det (int): Maximum detections per image.
        #* preds -> (bs, num_anchors, 4 + nc)
        bs, anchors, _ = preds.shape
        #* boxes -> (bs, anchors, 4), scores -> (bs, anchors, nc)
        boxes, scores = preds.split([4, nc], dim=-1)
        #* index -> (bs, max_det, 1)
        index = scores.amax(dim=-1).topk(min(max_det, anchors))[1].unsqueeze(-1)
        #* index.repeat(1, 1, 4) -> (bs, max_det, 4)
        #* boxes -> (bs, anchors, 4) -> (bs, max_det, 4)
        boxes = boxes.gather(dim=1, index=index.repeat(1, 1, 4))
        #* scores -> (bs, max_det, nc)
        scores = scores.gather(dim=1, index=index.repeat(1, 1, nc))
        scores, index = scores.flatten(1).topk(min(max_det, anchors))
        #* i -> (bs, 1)
        i = torch.arange(bs)[..., None]
        return torch.cat([boxes[i, index // nc], scores[..., None], (index % nc)[..., None].float()], dim=-1)

class Model(nn.Module):
    def __init__(self, cfg="yolov8.yaml", phi="n", ch=3, nc=None):
        super().__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg
        else:
            import yaml
            file_root = Path(__file__).parent.parent
            cfg = file_root / "model_cfg" / cfg
            self.yaml_file = Path(cfg).name
            with open(cfg) as f:
                self.yaml = yaml.load(f, Loader=yaml.SafeLoader)

        ch = self.yaml['ch'] = self.yaml.get('ch', ch)
        dep_mul, wid_mul, max_channels = self.yaml["scales"][phi]
        #* input img -> (3, 640, 640)
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch], gd=dep_mul, gw=wid_mul, mc=max_channels)
        #* 物体分类的默认名称
        self.names = {i: f"{i}" for i in range(self.yaml["nc"])}  # default names dict
        # self.inplace = self.yaml.get("inplace", True)
        m = self.model[-1]
        if isinstance(m, Detect):
            s = 256
            # m.inplace = self.inplace
            def _forward(x):
                return self.forward(x)

            self.model.eval()
            m.training = True
            m.stride = torch.tensor([s / x.shape[-2] for x in _forward(torch.zeros(1, ch, s, s))])
            self.stride = m.stride
            self.model.train()
        else:
            self.stride = torch.Tensor([32])

        initialize_weights(self)

    def forward(self, x, augment=False):
        if augment:
            img_size = x.shape[-2:]
            s = [1, 0.83, 0.67]
            f = [None, 3, None] #* 反转 (2-上下, 3-左右)
            y = []
            for si, fi in zip(s, f):
                xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
                yi = self.forward_once(xi)[0]
                yi[..., :4] /= si
                if fi == 2:
                    #* 当缩放大小为0.83时必定会左右反转
                    yi[..., 1] = img_size[0] - yi[..., 1]
                elif fi == 3:
                    yi[..., 0] = img_size[1] - yi[..., 0]
                y.append(yi)
            return torch.cat(y, 1), None
        else:
            return self.forward_once(x)

    def forward_once(self, x):
        y = []
        for m in self.model:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            y.append(x if m.i in self.save else None)
        return x

    def fuse(self):
        print("Fusing layers...")
        for m in self.model.modules():
            if type(m) is Conv and hasattr(m, "bn"):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)
                delattr(m, 'bn')
                m.forward = m.fuseforward
        return self

#* model_dict, input_channels(3)
def parse_model(d, ch, gd, gw, mc):
    nc = d["nc"]
    no = nc + 16 * 4
    # anchors, nc = d["anchors"], d["nc"]
    # na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors
    # no = na * (nc + 5)  #* number of outputs = anchors * (classes + 5)

    layers, save, c2 = [], [], ch[-1]
    #* from, number, module, arguments
    for i, (f, n, m, args) in enumerate(d["backbone"] + d["head"]):
        m = eval(m) if isinstance(m, str) else m
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a
            except:
                pass
        n = max(round(n * gd), 1) if n > 1 else n #* depth gain
        if m in [nn.Conv2d, Conv, DWConv, GhostConv, SPPF, C2f]:
            c1, c2 = ch[f], args[0]
            if c2 != no:
                c2 = min(make_divisible(c2 * gw, 8), mc)

            args = [c1, c2, *args[1:]]
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum([ch[x] for x in f])
        elif m in [Detect]:
            args.append([ch[x] for x in f])
        else:
            c2 = ch[f]

        m_ = nn.Sequential(
            *[m(*args) for _ in range(n)]
        ) if n > 1 else m(*args)
        #* t -> model.name
        t = str(m)[8:-2].replace("__main__.", "")
        np = sum([x.numel() for x in m_.parameters()])
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  #* attach index, 'from' index, type, number params
        # logger.info('%3s%18s%3s%10.0f  %-40s%-30s' % (i, f, n, np, t, args))  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)
