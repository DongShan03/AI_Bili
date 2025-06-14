import torch, sys, os
import torch.nn as nn
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from YOLOv5.utils.utils import de_parallel, bbox_iou

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

class QFocalLoss(nn.Module):
    #* 用来解决正负样本不平衡的问题
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred_prob = torch.sigmoid(pred)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss

class ComputeLoss:
    sort_obj_iou = False
    def __init__(self, model, hyp, autobalance=True):
        device = next(model.parameters()).device  # get model device

        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([hyp['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([hyp['obj_pw']], device=device))

        self.cp, self.cn = smooth_BCE(eps=hyp.get("label_smoothing", 0.0))
        # Focal loss
        # g = hyp['fl_gamma']  # focal loss gamma
        # if g > 0:
        #     BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        #* QFocal loss
        g = hyp['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = QFocalLoss(BCEcls, gamma=g), QFocalLoss(BCEobj, gamma=g)

        m = de_parallel(model).model[-1]
        self.balance = {3: [4.0, 1.0, 0.4]}.get(m.nl, [4.0, 1.0, 0.25, 0.06, 0.02])
        self.ssi = list(m.stride).index(16) if autobalance else 0   #* 去除值为16的index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, 1.0, hyp, autobalance
        self.na = m.na
        self.nc = m.nc
        self.nl = m.nl
        self.anchors = m.anchors
        self.device = device

    def __call__(self, p, targets):
        lcls = torch.zeros(1, device=self.device)
        lobj = torch.zeros(1, device=self.device)
        lbox = torch.zeros(1, device=self.device)

        tcls, tbox, indices, anchors = self.build_targets(p, targets)

        for i, pi in enumerate(p):
            b, a, gj, gi = indices[i]   #* [5*na*nt]
            tobj = torch.zeros(pi.shape[:4], dtype=torch.float64, device=self.device)

            n = b.shape[0]
            if n:
                pxy, pwh, _, pcls = pi[b, a, gi, gj].split((2, 2, 1, self.nc), 1)

                pxy = pxy.sigmoid() * 2 - 0.5
                pwh = (pwh.sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)
                iou = bbox_iou(pbox, tbox[i], CIoU=True).squeeze()
                lbox += (1.0 - iou).mean()

                iou = iou.detach().clamp(0).type(tobj.dtype)
                if self.sort_obj_iou:
                    j = iou.argsort()
                    b, a, gj, gi, iou = b[j], a[j], gj[j], gi[j], iou[j]
                if self.gr < 1:
                    iou = (1.0 - self.gr) + self.gr * iou
                tobj[b, a, gj, gi] = iou

                if self.nc > 1:
                    t = torch.full_like(pcls, self.cn, device=self.device)
                    t[range(n), tcls[i]] = self.cp
                    lcls += self.BCEcls(pcls, t)

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]

            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]

        lbox *= self.hyp["box"]
        lobj *= self.hyp["obj"]
        lcls *= self.hyp["cls"]
        bs = tobj.shape[0] * 9
        return {
            "box_loss": lbox * bs,
            "obj_loss": lobj * bs,
            "class_loss": lcls * bs
        }
    def build_targets(self, p, targets):
        na, nt = self.na, targets.shape[0]
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=self.device)
        #* ai -> [na, 1] -> [na, nt]
        ai = torch.arange(na, device=self.device).float().view(na, 1).repeat(1, nt)
        #* target[0] -> (image,class,x,y,w,h)
        #* targets -> [nt, 6] -> repeat -> [na, nt, 6]
        #* ai -> [na, nt, 1]
        #* targets -> [na, nt, 7]
        targets = torch.cat((targets.repeat(na, 1, 1), ai[..., None]), 2)   # append anchor indices

        g = 0.5 #* grid offset
        #* off -> [5, 2]
        off = torch.tensor([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1]], device=self.device).float() * g
        for i in range(self.nl):
            #* anchors -> [2(Wa, Ha]
            anchors, shape = self.anchors[i], p[i].shape
            gain[2:6] = torch.tensor(shape)[[3, 2, 3, 2]]  # xyxy gain
            #* targets -> [na, nt, 7] gain -> list=[1, 1, W, H, W, H]
            #* t -> [na, nt, 7]
            t = targets * gain
            if nt:
                #* t[..., 4:6] -> [na, nt, (W, H)] / [2, 1] -> [na, nt, (W/Wa, H/Ha)]
                r = t[..., 4:6] / anchors[:, None]
                #* j -> [na, nt]
                j = torch.max(r, 1 / r).max(2)[0] < self.hyp["anchor_t"]    #* hyp["anchor_t"] = 4.0
                #* t -> [na*nt, 7]
                t = t[j]

                # Offsets
                #* gain[2, 3] from p[ist layer].shape means the shape of grid
                gxy = t[:, 2:4]  # grid xy
                #* gxi 表示左右翻转后的gtbox坐标
                gxi = gain[[2, 3]] - gxy  # inverse
                #* 筛选符合条件的anchor
                #* gxy -> [na*nt, (W/Wa, H/Ha)] gxy.T -> [2, na*nt]
                #* j, k, l, m -> [na*nt]
                j, k = ((gxy % 1 < g) & (gxy > 1)).T
                l, m = ((gxi % 1 < g) & (gxi > 1)).T
                #* j -> [5, na*nt]
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                #* t -> [na*nt, 7] -> [5, na*nt, 7] -> [5*na*nt, 7]
                t = t.repeat((5, 1, 1))[j]
                #* torch.zeros_like(gxy)[None] -> [1, na*nt, 2] -> [5, na*nt, 2]
                #* off[:, None] -> [5, 1, 2] -> [5, na*nt, 2]
                #* offsets -> [5*na*nt, 2]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            #* bc -> [5*na*nt, 2] gxy -> [5*na*nt, 2] gwh -> [5*na*nt, 2] a -> [5*na*nt, 1]
            bc, gxy, gwh, a = t.chunk(4, 1)
            #* b -> [5*na*nt], c -> [5*na*nt]
            a, (b, c) = a.long().view(-1), bc.long().T
            #* [5*na*nt, 2] - [5*na*nt, 2] -> [5*na*nt, 2]
            gij = (gxy - offsets).long()
            #* gi -> [5*na*nt]
            gi, gj = gij.T

            indices.append((b, a, gj.clamp_(0, shape[2] - 1), gi.clamp_(0, shape[3] - 1)))  # image, anchor, grid)
            tbox.append(torch.cat((gxy - gij, gwh), 1))
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class

        return tcls, tbox, indices, anch
