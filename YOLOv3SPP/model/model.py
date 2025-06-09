import os, sys, math
project_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(os.path.dirname(project_dir))
from YOLOv3SPP.utils.parse_config import parse_model_cfg
import torch
import torch.nn as nn

net_cfg_path = os.path.join(project_dir, "net_cfg", "yolov3-spp.cfg")

class FeatureConcat(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = layers
        self.multiple = len(layers) > 1

    #! x指的是原始输入，但在融合前已经将需要融合的特征矩阵放在了layers中，因此不需要使用x
    def forward(self, x, outputs):
        return torch.cat([outputs[i] for i in self.layers], dim=1) \
            if self.multiple else outputs[self.layers[0]]


class WeightedFeatureFusion(nn.Module):
    def __init__(self, layers, weight=False):
        super().__init__()
        self.layers = layers
        self.weight = weight
        self.n = len(layers) + 1
        if weight:
            #! 融合权重
            self.w = nn.Parameter(torch.ones(self.n), requires_grad=True)

    def forward(self, x, outputs):
        if self.weight:
            w = torch.sigmoid(self.w) * (2 / self.n)
            x = x * w[0]
        #! input_channel
        nx = x.shape[1]
        for i in range(self.n - 1):
            a = outputs[self.layers[i]] * w[i + 1] if self.weight else outputs[self.layers[i]]
            na = a.shape[1]
            #! 调整channel
            if nx == na:
                x = x + a
            elif nx > na:
                x[ :, :na] = x[ :, :na] + a
            else:
                x = x + a[ :, :nx]

        return x

class YOLOLayer(nn.Module):
    def __init__(self, anchors, nc, img_size, stride):
        super().__init__()
        self.anchors = torch.tensor(anchors)
        self.stride = stride
        self.na = len(anchors)
        #* num of classes
        self.nc = nc
        #* num of output = x, y, w, h, conf, cls(nc)
        self.no = nc + 5
        self.nx, self.ny, self.ng = 0, 0, (0, 0)
        #* 将anchors大小缩放到grid尺度
        self.anchor_vec = self.anchors / self.stride
        #! [batch_size, na, grid_h, grid_w, wh]
        self.anchor_wh = self.anchor_vec.view(1, self.na, 1, 1, 2)
        self.grid = None

    def create_grids(self, ng=(13, 13), device="cpu"):
        """
        更新grids信息并生成新的grids参数
        :param ng: 特征图大小
        :param device:
        :return:
        """
        self.nx, self.ny = ng
        self.ng = torch.tensor(ng, dtype=torch.float)

        # build xy offsets 构建每个cell处的anchor的xy偏移量(在feature map上的)
        if not self.training:  # 训练模式不需要回归到最终预测boxes
            yv, xv = torch.meshgrid([torch.arange(self.ny, device=device),
                                     torch.arange(self.nx, device=device)], indexing='ij')
            # batch_size, na, grid_h, grid_w, wh
            self.grid = torch.stack((xv, yv), 2).view((1, 1, self.ny, self.nx, 2)).float()

        if self.anchor_vec.device != device:
            self.anchor_vec = self.anchor_vec.to(device)
            self.anchor_wh = self.anchor_wh.to(device)

    def forward(self, x):
        bs, _, ny, nx = x.shape
        if (self.nx, self.ny) != (nx, ny) or self.grid is None:
            self.create_grids((nx, ny), x.device)
        #! x -> [batch, 255, 13, 13] -> view[batch, 3, 85, 13, 13]
        #! -> [batch, anchor=3, grid_h=13, grid_w=13, xywh + obj + classes=85]
        x = x.view(bs, self.na, self.no, self.ny, self.nx).permute(0, 1, 3, 4, 2).contiguous()
        if self.training:
            return x
        else:
            io = x.clone()
            #* 将回归参数映射到特征图上
            io[..., :2] = torch.sigmoid(io[..., :2]) + self.grid
            io[..., 2:4] = torch.exp(io[..., 2:4]) * self.anchor_wh
            #* 换算到原图尺寸
            io[..., :4] *= self.stride
            torch.sigmoid_(io[..., 4:])
            return io.view(bs, -1, self.no), x #! [1, 3, 13, 13, 85] -> [1, 507, 85]



def create_modules(model_cfg: list, img_size):
    model_cfg.pop(0)
    output_filters = [3]
    module_list = nn.ModuleList()
    #* 统计哪些特征层的输出会在后续被使用
    routs = []
    yolo_index = -1

    for i, mdef in enumerate(model_cfg):
        modules = nn.Sequential()
        if mdef["type"] == "convolutional":
            bn = mdef["batch_normalize"]
            filters = mdef["filters"]
            ksize = mdef["size"]
            stride = mdef["stride"]
            padding = ksize // 2 if mdef["pad"] else 0
            act = mdef["activation"]
            if isinstance(ksize, int):
                modules.add_module(
                    "Conv2d", nn.Conv2d(in_channels=output_filters[-1],
                                        out_channels=filters, kernel_size=ksize,
                                        stride=stride, padding=padding, bias=not bn)
                )
            else:
                raise ValueError("Conv2d kernel size must be an integer.")
            if bn:
                modules.add_module("BatchNorm2d", nn.BatchNorm2d(filters))
            else:
                #* 如果卷积操作不接bn层，意味着该层为yolo的predictor
                routs.append(i)

            if act == "leaky":
                modules.add_module("activation", nn.LeakyReLU(0.1, inplace=True))
            else:
                pass
        elif mdef["type"] == "BatchNorm2d":
            modules.add_module("BatchNorm2d", nn.BatchNorm2d(output_filters[-1]))
        elif mdef['type'] == "maxpool":
            ksize = mdef["size"]
            stride = mdef["stride"]
            modules.add_module("maxpool", nn.MaxPool2d(kernel_size=ksize, stride=stride, padding=(ksize - 1) // 2))
        elif mdef["type"] == "upsample":
            modules.add_module("upsample", nn.Upsample(scale_factor=mdef["stride"]))
        elif mdef["type"] == "route":
            layers = mdef["layers"]
            filters = sum([output_filters[l + 1 if l > 0 else l] for l in layers])
            routs.extend([i + l if l < 0 else l for l in layers])
            modules = FeatureConcat(layers=layers)
        elif mdef["type"] == "shortcut":
            layers = mdef["from"]
            filters = output_filters[-1]
            routs.append(i + layers[0])
            modules = WeightedFeatureFusion(layers=layers, weight="weights_type" in mdef)
        elif mdef["type"] == "yolo":
            yolo_index += 1
            #* 预测特征层对应原图的缩放比例
            stride = [32, 16, 8]
            modules = YOLOLayer(
                anchors=mdef["anchors"][mdef["mask"]],
                nc=mdef["classes"],
                img_size=img_size,
                stride=stride[yolo_index]
            )
            try:
                j = -1
                bias_ = module_list[j][0].bias
                #* YOLOLayer的bias初始化
                bias = bias_.view(modules.na, -1)
                with torch.no_grad():
                    bias[:, 4] += -4.5
                    bias[:, 5:] += math.log(0.6 / (modules.nc - 0.99))
                module_list[j][0].bias = torch.nn.Parameter(bias.view(-1), requires_grad=True)
            except Exception as e:
                print('WARNING: smart bias initialization failure.', e)
        else:
            print("Unknown Layer Type: " + mdef["type"])
        module_list.append(modules)
        output_filters.append(filters)

    routs_binary = [False] * len(model_cfg)
    #! 记录哪些层在后续会被用到
    for i in routs:
        routs_binary[i] = True
    return module_list, routs_binary

def get_yolo_layers(self):
    return [i for i, m in enumerate(self.module_list) if m.__class__.__name__ == "YOLOLayer"]

class Yolov3SPP(nn.Module):
    def __init__(self, cfg_path=net_cfg_path, img_size=(416, 416), verbose=False):
        super().__init__()
        #* input_size只在导出ONNX模型时有用
        self.input_size = [img_size] * 2 if isinstance(img_size, int) else img_size
        self.verbose = verbose
        self.model_cfg = parse_model_cfg(cfg_path)
        #! 这里原先是img_size,改为input_size
        self.module_list, self.routes = create_modules(self.model_cfg, self.input_size)
        self.yolo_layers = get_yolo_layers(self)

    def forward(self, x, verbose=False):
        return self.forward_once(x, verbose)

    def forward_once(self, x, verbose=False):
        yolo_out, out = [], []
        if verbose:
            print("0", x.shape)
            str = ''

        for i, module in enumerate(self.module_list):
            name = module.__class__.__name__
            if name in ["WeightedFeatureFusion", "FeatureConcat"]:
                if verbose:
                    l = [i - 1] + module.layers
                    sh = [list(x.shape)] + [list(out[i].shape) for i in module.layers]
                    str_out = ' >> ' + " + ".join(['layer %g %s '% x for x in zip(l, sh)])
                x = module(x, out)
            elif name == "YOLOLayer":
                yolo_out.append(module(x))
            else:
                x = module(x)

            out.append(x if self.routes[i] else [])
            if verbose:
                print("%g/%g %s - "%(i, len(self.module_list), name), list(x.shape), str_out)
                str_out = ""
        if self.training:
            return yolo_out
        else:
            #! x -> tuple([1, 768, 25], [1, 3072, 25], [1, 12288, 25])
            #! p -> tuple([1, 3, 16, 16, 25], [1, 3, 32, 32, 25], [1, 3, 64, 64, 25])
            x, p = zip(*yolo_out)
            x = torch.cat(x, 1)
            #! x -> [1, 16128, 85]
            return x, p


if __name__ == '__main__':
    model = Yolov3SPP()
    model.eval()
    x = torch.randn(1, 3, 512, 512)
    x, p = model(x)
    print(x.shape)
