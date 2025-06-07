import itertools

import torch
import torch.nn.functional as F
from torch.jit.annotations import Tuple, List
from torch import nn, Tensor
import numpy as np

def box_area(boxes):
    return(boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

def calc_iou_tensor(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou

def nms(boxes, scores, iou_threshold):
    # type: (Tensor, Tensor, float) -> Tensor
    return torch.ops.torchvision.nms(boxes, scores, iou_threshold)

def batched_nms(boxes, scores, idxs, iou_threshold):
    # type: (Tensor, Tensor, Tensor, float) -> Tensor
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)

    # 获取所有boxes中最大的坐标值（xmin, ymin, xmax, ymax）
    max_coordinate = boxes.max()

    # 为每一个类别生成一个很大的偏移量
    # 这里的to只是让生成tensor的dytpe和device与boxes保持一致
    offsets = idxs.to(boxes) * (max_coordinate + 1)
    # boxes加上对应层的偏移量后，保证不同类别之间boxes不会有重合的现象
    boxes_for_nms = boxes + offsets[:, None]
    keep = nms(boxes_for_nms, scores, iou_threshold)
    return keep


class Encoder(object):
    def __init__(self, dboxes):
        self.dboxes = dboxes(order="ltrb")
        self.dboxes_xywh = dboxes(order="xywh").unsqueeze(0)
        self.nboxes = self.dboxes.size(0)    #! default boxes的数量
        self.scale_xy = dboxes.scale_xy
        self.scale_wh = dboxes.scale_wh

    def encode(self, bboxes_in, labels_in, criteria=0.5):
        #! 计算每个GT(bboxes_in)与default box的iou
        #! ious -> [nboxes, 8732]
        ious = calc_iou_tensor(bboxes_in, self.dboxes)
        #! 寻找每个default box匹配到的最大IoU
        #! best_dbox_ious -> [8732, ]
        best_dbox_ious, best_dbox_idx = ious.max(dim=0)
        #! 寻找每个GT匹配到的最大IoU
        #! best_bbox_idx -> [nboxes, ]
        best_bbox_ious, best_bbox_idx = ious.max(dim=1)

        #* 将每个GT匹配到的最佳default box设置为正样本
        best_dbox_ious.index_fill_(0, best_bbox_idx, 2.0) # dim, index, value
        #* 将相应default box匹配最大IOU的GT索引进行替换 torch.arange(0, nboxes)
        idx = torch.arange(0, best_bbox_idx.size(0), dtype=torch.int64)
        #! best_bbox_idx[idx] -> 每个GT匹配到的最好的default box
        best_dbox_idx[best_bbox_idx[idx]] = idx

        # filter IoU > 0.5
        # 寻找与GT iou大于0.5的default box,对应论文中Matching strategy的第二条(这里包括了第一条匹配到的信息)
        masks = best_dbox_ious > criteria
        labels_out = torch.zeros(self.nboxes, dtype=torch.int64)
        labels_out[masks] = labels_in[best_dbox_idx[masks]]

        bboxes_out = self.dboxes.clone()
        bboxes_out[masks, :] = bboxes_in[best_dbox_idx[masks], :]

        x = 0.5 * (bboxes_out[:, 0] + bboxes_out[:, 2])
        y = 0.5 * (bboxes_out[:, 1] + bboxes_out[:, 3])
        w = bboxes_out[:, 2] - bboxes_out[:, 0]
        h = bboxes_out[:, 3] - bboxes_out[:, 1]
        bboxes_out[:, 0] = x
        bboxes_out[:, 1] = y
        bboxes_out[:, 2] = w
        bboxes_out[:, 3] = h

        return bboxes_out, labels_out

    #* 将box格式从xywh转换回ltrb, 将预测目标score通过softmax处理
    def scale_back_batch(self, bboxes_in, scores_in):
        if bboxes_in.device == torch.device("cpu"):
            self.dboxes = self.dboxes.cpu()
            self.dboxes_xywh = self.dboxes_xywh.cpu()
        else:
            self.dboxes = self.dboxes.cuda()
            self.dboxes_xywh = self.dboxes_xywh.cuda()

        bboxes_in = bboxes_in.permute(0, 2, 1)
        scores_in = scores_in.permute(0, 2, 1)
        #* 预测的回归参数
        bboxes_in[:, :, :2] = self.scale_xy * bboxes_in[:, :, :2]
        bboxes_in[:, :, 2:] = self.scale_wh * bboxes_in[:, :, 2:]

        #* 将预测的回归参数叠加到default box上得到最终的预测边界框
        bboxes_in[:, :, :2] = bboxes_in[:, :, :2] * self.dboxes_xywh[:, :, 2:] + self.dboxes_xywh[:, :, :2]
        bboxes_in[:, :, 2:] = bboxes_in[:, :, 2:].exp() * self.dboxes_xywh[:, :, 2:]

        l = bboxes_in[:, :, 0] - 0.5 * bboxes_in[:, :, 2]
        t = bboxes_in[:, :, 1] - 0.5 * bboxes_in[:, :, 3]
        r = bboxes_in[:, :, 0] + 0.5 * bboxes_in[:, :, 2]
        b = bboxes_in[:, :, 1] + 0.5 * bboxes_in[:, :, 3]

        bboxes_in[:, :, 0] = l
        bboxes_in[:, :, 1] = t
        bboxes_in[:, :, 2] = r
        bboxes_in[:, :, 3] = b
        return bboxes_in, F.softmax(scores_in, dim=-1)

    def decode_batch(self, bboxes_in, scores_in, criteria=0.45, max_output=200):
        bboxes, probs = self.scale_back_batch(bboxes_in, scores_in)

        outputs = []
        for bbox, prob in zip(bboxes.split(1, 0), probs.split(1, 0)):
            bbox = bbox.squeeze(0)
            prob = prob.squeeze(0)
            outputs.append(self.decode_single_new(bbox, prob, criteria, max_output))

        return outputs

    def decode_single_new(self, bboxes_in, scores_in, criteria, num_output=200):
        device = bboxes_in.device
        num_classes = scores_in.shape[-1]

        #* 对越界的bbox进行裁剪
        bboxes_in = bboxes_in.clamp(min=0, max=1)
        #! [8732, 4] -> [8732, 21, 4]
        bboxes_in = bboxes_in.repeat(1, num_classes).reshape(scores_in.shape[0], -1, 4)

        labels = torch.arange(num_classes, device=device)
        labels = labels.view(1, -1).expand_as(scores_in)

        bboxes_in = bboxes_in[:, 1:, :]
        scores_in = scores_in[:, 1:]
        labels = labels[:, 1:]

        bboxes_in = bboxes_in.reshape(-1, 4)
        scores_in = scores_in.reshape(-1)
        labels = labels.reshape(-1)
        #! 移除低概率目标，self.scores_thresh=0.05
        inds = torch.nonzero(scores_in > 0.05, as_tuple=False).squeeze(dim=1)
        bboxes_in, scores_in, labels = bboxes_in[inds], scores_in[inds], labels[inds]

        ws, hs = bboxes_in[:, 2] - bboxes_in[:, 0], bboxes_in[:, 3] - bboxes_in[:, 1]
        keep = (ws >= 0.1 / 300) & (hs >= 0.1 / 300)
        keep = keep.nonzero(as_tuple=False).squeeze(1)
        bboxes_in, scores_in, labels = bboxes_in[keep], scores_in[keep], labels[keep]

        #* nms
        keep = batched_nms(bboxes_in, scores_in, labels, iou_threshold=criteria)
        #* 只保留前num_output个
        keep = keep[:num_output]
        bboxes_out = bboxes_in[keep, :]
        labels_out = labels[keep]
        scores_out = scores_in[keep]

        return bboxes_out, labels_out, scores_out

class Loss(nn.Module):
    def __init__(self, dboxes):
        super().__init__()
        self.scale_xy = 1.0 / dboxes.scale_xy   #* 10
        self.scale_wh = 1.0 / dboxes.scale_wh   #* 5
        self.location_loss = nn.SmoothL1Loss(reduction="none")
        #! dboxes(order='xywh') -> [8732, 4]
        #! transpose(0, 1) -> [4, 8732]
        #! unsqueeze(0) -> [1, 4, 8732] 为batch提前准备
        self.dboxes = nn.Parameter(dboxes(order='xywh').transpose(0, 1).unsqueeze(0), requires_grad=False)
        self.confidence_loss = nn.CrossEntropyLoss(reduction="none")

    def _location_vec(self, loc):
        gxy = self.scale_xy * (loc[:, :2, :] - self.dboxes[:, :2, :]) / self.dboxes[:, 2:, ]  #* [batch, 2, 8732]
        gwh = self.scale_wh * (loc[:, 2:, :] / self.dboxes[:, 2:, ]).log()  #* [batch, 2, 8732]
        return torch.cat((gxy, gwh), dim=1).contiguous()

    def forward(self, ploc, plabel, gloc, glabel):
        #* ploc -> [batch, 4, 8732]
        #* plabel -> [batch, num_classes, 8732]
        #? gloc -> [batch, 4, 8732]
        #? glabel -> [batch, 8732]
        mask = glabel > 0
        #* 计算一个batch中每张照片正样本的数量
        #* pos_num -> [batch, N]
        pos_num = mask.sum(dim=1)
        #* 计算gt的location参数 [batch, 4, 8732]
        vec_gd = self._location_vec(gloc)
        #* loc_loss -> [batch, 8732]
        loc_loss = self.location_loss(ploc, vec_gd).sum(dim=1)
        #* loc_loss -> [batch, 1]
        loc_loss = (mask.float() * loc_loss).sum(dim=1)

        con = self.confidence_loss(plabel, glabel)

        con_neg = con.clone()
        #* 排除正样本
        con_neg[mask] = torch.tensor(0.0)
        _, con_idx = con_neg.sort(dim=1, descending=True)
        _, con_rank = con_idx.sort(dim=1)
        #! 负样本的个数是正样本的三倍
        neg_num = torch.clamp(pos_num * 3, max=mask.size(1)).unsqueeze(-1)
        neg_mask = con_rank < neg_num
        #* confidence最终loss使用选取的正样本loss+选取的负样本loss
        con_loss = (con * (mask.float() + neg_mask.float())).sum(dim=1)
        total_loss = loc_loss + con_loss
        #! 避免gtbox为0的情况
        #* 统计一个batch中的每张图像中是否存在正样本
        num_mask = (pos_num > 0).float()
        #* 防止出现分母为零的情况
        pos_num = pos_num.float().clamp(min=1e-6)
        #* 只计算存在正样本的图像损失
        ret = (total_loss * num_mask / pos_num).mean(dim=0)
        return ret

class DefaultBoxes:
    def __init__(self, fig_size,    #* 输入网络的图像大小
                feature_sizes,            #* 每个特征层的尺寸
                steps,                  #* 每个特征层的cell对应原图的步长
                scales,                 #* 每个特征层上default boxes的尺寸[21, 45, 99, 153, 207, 261, 315]
                aspect_ratios,          #* 每个特征层上default boxes的宽高比
                scale_xy=0.1,
                scale_wh=0.2
                ):
        self.fig_size = fig_size    #* 300
        self.feature_sizes = feature_sizes    #* [38, 19, 10, 5, 3, 1]
        self.scale_xy_ = scale_xy
        self.scale_wh_ = scale_wh
        #* [8, 16, 32, 64, 100, 300]
        self.steps = steps
        #* [21, 45, 99, 153, 207, 261, 315]
        self.scales = scales

        fk = fig_size / np.array(steps)
        self.aspect_ratios = aspect_ratios

        self.default_boxes = []
        for idx, feat_size in enumerate(self.feature_sizes):
            sk1 = self.scales[idx] / fig_size
            sk2 = self.scales[idx + 1] / fig_size
            sk3 = np.sqrt(sk1 * sk2)
            all_sizes = [(sk1, sk2), (sk3, sk3)]
            for alpha in aspect_ratios[idx]:
                w, h = sk1 * np.sqrt(alpha), sk2 / np.sqrt(alpha)
                all_sizes.append((w, h))
                all_sizes.append((h, w))
            for w, h in all_sizes:
                for i, j in itertools.product(range(feat_size), repeat=2):
                    cx, cy = (j + 0.5) / fk[idx], (i + 0.5) / fk[idx]
                    self.default_boxes.append((cx, cy, w, h))
        self.dboxes = torch.tensor(self.default_boxes, dtype=torch.float32)
        self.dboxes.clamp_(min=0, max=1)

        self.dboxes_ltrb = self.dboxes.clone()
        #! 将(cx, cy, w, h)转化为(xmin, ymin, xmax, ymax)->(l, t, r, b)
        self.dboxes_ltrb[:, 0] = self.dboxes[:, 0] - self.dboxes[:, 2] * 0.5
        self.dboxes_ltrb[:, 1] = self.dboxes[:, 1] - self.dboxes[:, 3] * 0.5
        self.dboxes_ltrb[:, 2] = self.dboxes[:, 0] + self.dboxes[:, 2] * 0.5
        self.dboxes_ltrb[:, 3] = self.dboxes[:, 1] + self.dboxes[:, 3] * 0.5

    #* @property 把方法当作属性调用
    @property
    def scale_xy(self):
        return self.scale_xy_

    @property
    def scale_wh(self):
        return self.scale_wh_

    def __call__(self, order="ltrb"):
        if order == "ltrb":
            return self.dboxes_ltrb
        else:
            return self.dboxes

def dboxes300_coco():
    figsize = 300
    feature_sizes = [38, 19, 10, 5, 3, 1]
    steps = [8, 16, 32, 64, 100, 300]
    scales = [21, 45, 99, 153, 207, 261, 315]
    aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
    dboxes = DefaultBoxes(figsize, feature_sizes, steps, scales, aspect_ratios)
    return dboxes


class PostProcess(nn.Module):
    def __init__(self, dboxes):
        super().__init__()
        self.dboxes_xywh = nn.Parameter(dboxes(order="xywh").unsqueeze(0), requires_grad=False)
        self.scale_xy = dboxes.scale_xy
        self.scale_wh = dboxes.scale_wh
        self.criteria = 0.5
        self.max_output = 100

    def scale_back_batch(self, bboxes_in, scores_in):
        """
            将预测的boxes回归参数转化为实际预测坐标
            将box从xywh转化为ltrb
            softmax处理
        """
        #! [batch, 4, 8732] -> [batch, 8732, 4]
        bboxes_in = bboxes_in.permute(0, 2, 1)
        scores_in = scores_in.permute(0, 2, 1)
        bboxes_in[..., :2] = bboxes_in[..., :2] * self.scale_xy
        bboxes_in[..., 2:] = bboxes_in[..., 2:] * self.scale_wh

        bboxes_in[:, :, :2] = bboxes_in[:, :, :2] * self.dboxes_xywh[:, :, 2:] + self.dboxes_xywh[:, :, :2]
        bboxes_in[:, :, 2:] = bboxes_in[:, :, 2:].exp() * self.dboxes_xywh[:, :, 2:]

        l = bboxes_in[:, :, 0] - 0.5 * bboxes_in[:, :, 2]
        t = bboxes_in[:, :, 1] - 0.5 * bboxes_in[:, :, 3]
        r = bboxes_in[:, :, 0] + 0.5 * bboxes_in[:, :, 2]
        b = bboxes_in[:, :, 1] + 0.5 * bboxes_in[:, :, 3]

        bboxes_in[:, :, 0] = l
        bboxes_in[:, :, 1] = t
        bboxes_in[:, :, 2] = r
        bboxes_in[:, :, 3] = b
        #! scores_in -> [batch, 8732, label_num=21]
        #! scores_out -> [batch, 8732, label_num=21]
        return bboxes_in, F.softmax(scores_in, dim=-1)

    def decode_single_new(self, bboxes_in, scores_in, criteria, num_output=200):
        device = bboxes_in.device
        num_classes = scores_in.shape[-1]

        #* 对越界的bbox进行裁剪
        bboxes_in = bboxes_in.clamp(min=0, max=1)
        #! [8732, 4] -> [8732, 21, 4]
        bboxes_in = bboxes_in.repeat(1, num_classes).reshape(scores_in.shape[0], -1, 4)

        labels = torch.arange(num_classes, device=device)
        labels = labels.view(1, -1).expand_as(scores_in)
        #* 移除背景概率信息
        bboxes_in = bboxes_in[:, 1:, :]
        scores_in = scores_in[:, 1:]
        labels = labels[:, 1:]
        #! [8732, 20, 4] -> [8732*20, 4]
        bboxes_in = bboxes_in.reshape(-1, 4)
        scores_in = scores_in.reshape(-1)
        labels = labels.reshape(-1)
        #! 移除低概率目标，self.scores_thresh=0.05
        inds = torch.nonzero(scores_in > 0.05, as_tuple=False).squeeze(dim=1)
        bboxes_in, scores_in, labels = bboxes_in[inds, :], scores_in[inds], labels[inds]

        ws, hs = bboxes_in[:, 2] - bboxes_in[:, 0], bboxes_in[:, 3] - bboxes_in[:, 1]
        keep = (ws >= 0.1 / 300) & (hs >= 0.1 / 300)
        keep = keep.nonzero(as_tuple=False).squeeze(1)
        bboxes_in, scores_in, labels = bboxes_in[keep], scores_in[keep], labels[keep]

        #* nms
        keep = batched_nms(bboxes_in, scores_in, labels, iou_threshold=criteria)
        #* 只保留前num_output个
        keep = keep[:num_output]
        bboxes_out = bboxes_in[keep, :]
        labels_out = labels[keep]
        scores_out = scores_in[keep]

        return bboxes_out, labels_out, scores_out

    def forward(self, bboxes_in, scores_in):
        bboxes, probs = self.scale_back_batch(bboxes_in, scores_in)

        outputs = torch.jit.annotate(List[Tuple[Tensor, Tensor]], [])
        #* 对每张图片
        for bbox, prob in zip(bboxes.split(1, 0), probs.split(1, 0)):   #! split_size, split_dim
            bbox = bbox.squeeze(0)
            prob = prob.squeeze(0)
            outputs.append(self.decode_single_new(bbox, prob, self.criteria, self.max_output))

        return outputs
