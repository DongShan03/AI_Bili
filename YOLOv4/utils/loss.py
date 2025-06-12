import torch, sys, os
import torch.nn as nn
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from YOLOv4.utils.utils import is_parallel, bbox_iou

def smooth_BCE(eps=0.1):
    #* return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps

class BCEBlurWithLogitsLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=0.05):
        super().__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()

class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss

def build_targets(p, targets, model):
    nt = targets.shape[0]   # number of targets
    tcls, tbox, indices, anch = [], [], [], []
    gain = torch.ones(6, device=targets.device).long()
    #* 需要考虑来自四个方向上的anchor
    off = torch.tensor([[1, 0], [0, 1], [-1, 0], [0, -1]], device=targets.device).float()
    #* offset
    g = 0.5
    multi_gpu = is_parallel(model)
    for i, j in enumerate(model.module.yolo_layers if multi_gpu else model.yolo_layers):
        # get number of grid points and anchor vec for this yolo layer
        anchors = model.module.module_list[j].anchor_vec if multi_gpu else model.module_list[j].anchor_vec
        device = anchors.device
        #! p[i] -> [B, C, H, W]第i个特征层的输出
        #! gain[2:] -> [W, H, W, H]
        gain[2:] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]

        #* 将目标匹配到anchor模板
        #* targets -> [nt, 6] gain -> [1, 1, W, H, W, H]
        #* t -> [nt, 6]
        a, t, offsets = [], targets * gain, 0
        if nt:
            na = anchors.shape[0]   #* 3
            at = torch.arange(na).view(na, -1).repeat(1, nt).to(device)    #* [3, nt]
            #* t[None, :, 4:6] -> [1, nt, 2(W, H)]
            #* anchors[:, None] -> [3, 1, 2]        anchor内只存放宽高
            r = (t[None, :, 4:6] / anchors[:, None]).to(device)
            #* r -> [3, nt, 2(W/Wa, H/Ha)]
            j = torch.tensor(torch.max(r, 1. / r).max(2)[0] < model.hyp['anchor_t'], dtype=torch.bool).to(device)
            #* j -> [3, nt] bool
            #* t.repeat(na, 1, 1) -> [3, nt, 6]
            #* a -> [j.num] t -> [j.num, 6]
            a, t = at[j], t.repeat(na, 1, 1)[j]
            #* gxy -> [j.num, 2(W, H)]
            gxy = t[:, 2:4]
            z = torch.zeros_like(gxy)
            #* j, k, l, m 筛选出周围四个格子中符合条件的正样本
            j, k = ((gxy % 1. < g) & (gxy > 1.)).T
            l, m = ((gxy % 1. > (1 - g)) & (gxy < (gain[[2, 3]] - 1.))).T
            a, t = torch.cat((a, a[j], a[k], a[l], a[m]), 0), torch.cat((t, t[j], t[k], t[l], t[m]), 0)
            offsets = torch.cat((z, z[j] + off[0], z[k] + off[1], z[l] + off[2], z[m] + off[3]), 0) * g
        #* b -> image, c -> class
        b, c = t[:, :2].long().T
        gxy = t[:, 2:4]
        gwh = t[:, 4:6]
        gij = (gxy - offsets).long()
        gi, gj = gij.T

        indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))
        tbox.append(torch.cat((gxy - gij, gwh), 1))
        anch.append(anchors[a])
        tcls.append(c)
    return tcls, tbox, indices, anch


#* predictions, targets, model
def compute_loss(p, targets, model):
    device = targets.device
    lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
    tcls, tbox, indices, anchors = build_targets(p, targets, model)
    h = model.hyp

    BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([h["cls_pw"]])).to(device)
    BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([h["obj_pw"]])).to(device)

    cp, cn = smooth_BCE(eps=0.0)

    g = h["fl_gamma"]
    if g > 0:
        BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

    nt = 0
    no = len(p)     #* number of outputs
    balance = [4.0, 1.0, 0.4] if no == 3 else [4.0, 1.0, 0.4, 0.1]
    balance = [4.0, 1.0, 0.5, 0.4, 0.1] if no == 5 else balance
    #* i -> 输出层索引 pi -> 输出
    for i, pi in enumerate(p):
        b, a, gj, gi = indices[i]
        tobj = torch.zeros_like(pi[..., 0], device=device)

        n = b.shape[0]  #* 目标数量
        if n:
            nt += n
            ps = pi[b, a, gj, gi]

            pxy = ps[:, :2].sigmoid() * 2. - 0.5
            pwh =  (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
            pbox = torch.cat((pxy, pwh), 1).to(device)

            iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)
            lbox += (1 - iou).mean()

            tobj[b, a, gj, gi] = (1.0 - model.gr) + model.gr * iou.detach().clamp(0).type(tobj.dtype)

            if model.nc > 1:
                t = torch.full_like(ps[:, 5:], cn, device=device)
                t[range(n), tcls[i]] = cp
                lcls += BCEcls(ps[:, 5:], t)  #* cls loss

        lobj += BCEobj(pi[..., 4], tobj) * balance[i]  #* obj loss

    s = 3 / no
    lbox *= h["box"] * s
    lobj *= h["obj"] * s * (1.4 if no >= 4 else 1.)
    lcls *= h["cls"] * s
    bs = tobj.shape[0]  # batch size

    return {
        "box_loss": lbox * bs,
        "obj_loss": lobj * bs,
        "class_loss": lcls * bs
    }
