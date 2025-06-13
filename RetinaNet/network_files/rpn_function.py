import torch, os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.jit.annotations import List, Tuple, Optional, Dict
from RetinaNet.network_files.image_list import ImageList
import RetinaNet.network_files.det_utils as det_utils
import RetinaNet.network_files.boxes as box_ops
class AnchorsGenerator(nn.Module):
    __annotations__ = {
        "cell_anchors": Optional[List[Tensor]],
        "_cache": Dict[str, List[Tensor]]
    }
    def __init__(self, sizes=(128, 256, 512), aspect_ratios=(0.5, 1.0, 2.0)):
        super().__init__()
        if not isinstance(sizes[0], (list, tuple)):
            sizes = tuple((s,) for s in sizes)
        if not isinstance(aspect_ratios[0], (list, tuple)):
            aspect_ratios = (aspect_ratios,) * len(sizes)
        assert len(sizes) == len(aspect_ratios)
        self.sizes = sizes
        self.aspect_ratios = aspect_ratios
        self.cell_anchors = None
        self._cache = {}

    def num_anchors_per_location(self):
        # 计算每个预测特征层上每个滑动窗口的预测目标数
        return [len(s) * len(a) for s, a in zip(self.sizes, self.aspect_ratios)]

    def generate_anchors(self, scales, aspect_ratios, dtype=torch.float32, device="cpu"):
        # type: (List[int], List[float], int, torch.device) -> Tensor
        scales = torch.as_tensor(scales, dtype=dtype, device=device)
        aspect_ratios = torch.as_tensor(aspect_ratios, dtype=dtype, device=device)
        h_ratio = torch.sqrt(aspect_ratios)
        w_ratio = 1 / h_ratio

        ws = (w_ratio[:, None] * scales[None, :]).view(-1)
        hs = (h_ratio[:, None] * scales[None, :]).view(-1)
        #! 生成的anchor以（0, 0）为中心
        base_anchors = torch.stack([-ws, -hs, ws, hs], dim=1) / 2
        return base_anchors.round()

    def set_cell_anchors(self, dtype, device):
        # type: (torch.dtype, torch.device) -> None
        if self.cell_anchors is not None:
            cell_anchors = self.cell_anchors
            assert cell_anchors is not None
            if cell_anchors[0].device == device:
                return
        cell_anchors = [
            self.generate_anchors(size, aspect_ratio, dtype, device) \
            for size, aspect_ratio in zip(self.sizes, self.aspect_ratios)
        ]
        self.cell_anchors = cell_anchors

    def grid_anchors(self, grid_sizes, strides):
        # type: (List[List[int]], List[List[Tensor]]) -> List[Tensor]
        anchors = []
        cell_anchors = self.cell_anchors
        assert cell_anchors is not None

        for size, stride, base_anchors in zip(grid_sizes, strides, cell_anchors):
            grid_height, grid_width = size
            stride_height, stride_width = stride
            device = base_anchors.device

            shifts_x = torch.arange(0, grid_width, dtype=torch.float32, device=device) * stride_width
            shifts_y = torch.arange(0, grid_height, dtype=torch.float32, device=device) * stride_height

            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing='ij')
            shift_x = shift_x.reshape(-1)
            shift_y = shift_y.reshape(-1)

            shifts = torch.stack((shift_x, shift_y, shift_x, shift_y), dim=1)

            shifts_anchors = shifts.view(-1, 1, 4) + base_anchors.view(1, -1, 4)
            anchors.append(shifts_anchors.reshape(-1, 4))
        return anchors

    def cached_grid_anchors(self, grid_sizes, strides):
        # type: (List[List[int]], List[List[Tensor]]) -> List[Tensor]
        key = str(grid_sizes) + str(strides)
        if key in self._cache:
            return self._cache[key]
        anchors = self.grid_anchors(grid_sizes, strides)
        self._cache[key] = anchors
        return anchors

    def forward(self, image_list, feature_maps):
        # type: (ImageList, List[Tensor]) -> List[Tensor]
        #! 获取每个预测特征图的尺度
        grid_sizes = list([feature_map.shape[-2:] for feature_map in feature_maps])
        image_size = image_list.tensors.shape[-2:]
        dtype, device = feature_maps[0].dtype, feature_maps[0].device
        #* 计算特征层的一格相当于原图的多少格
        strides = [[torch.tensor(image_size[0] / g[0], dtype=torch.int64, device=device),
                    torch.tensor(image_size[1] / g[1], dtype=torch.int64, device=device)]
                    for g in grid_sizes]
        #* 根据提供的sizes和aspect_ratios生成anchors
        self.set_cell_anchors(dtype, device)
        #* 读取所有anchors坐标信息，返回list，对应每一个特征图映射回原图的anchor坐标信息
        anchors_over_all_feature_map = self.cached_grid_anchors(grid_sizes, strides)
        anchors = torch.jit.annotate(List[List[torch.Tensor]], [])
        #* 遍历一个batch中的所有图像
        for i, (image_height, image_width) in enumerate(image_list.image_sizes):
            anchors_in_image = []
            #* 遍历每张特征图映射回原图的anchor坐标信息
            for anchors_per_feature_map in anchors_over_all_feature_map:
                anchors_in_image.append(anchors_per_feature_map)
            anchors.append(anchors_in_image)
        #* 将每一张图像的所有特征层的anchor坐标信息合并
        #* anchors为list，每一个元素为一张图像的所有特征层的anchor
        anchors = [torch.cat(anchors_per_image) for anchors_per_image in anchors]
        self._cache.clear()
        return anchors

class RPNHead(nn.Module):
    def __init__(self, in_channels, num_anchors):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.cls_logits = nn.Conv2d(in_channels, num_anchors, kernel_size=1, stride=1)
        self.bbox_pred = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=1, stride=1)

        for layer in self.children():
            if isinstance(layer, nn.Conv2d):
                torch.nn.init.normal_(layer.weight, mean=0, std=0.01)
                torch.nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        # type: (List[Tensor]) -> Tuple[List[Tensor], List[Tensor]]
        logits = []
        bbox_reg = []
        for _, feature in enumerate(x):
            t = F.relu(self.conv(feature))
            logits.append(self.cls_logits(t))
            bbox_reg.append(self.bbox_pred(t))
        return logits, bbox_reg


class RegionProposalNetwork(nn.Module):
    __annotations__ = {
        "box_coder": det_utils.BoxCoder,
        "proposal_matcher": det_utils.Matcher,
        "fg_bg_sampler": det_utils.BalancedPositiveNegativeSampler,
        "pre_nms_top_n": Dict[str, int],
        "post_nms_top_n": Dict[str, int]
    }
    def __init__(self, anchor_generator, head,
                fg_iou_thresh, bg_iou_thresh,
                batch_size_per_image,
                positive_fraction,
                pre_nms_top_n, post_nms_top_n,
                nms_thresh):
        super().__init__()
        self.anchor_generator = anchor_generator
        self.head = head
        self.box_coder = det_utils.BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))

        self.box_similarity = box_ops.box_iou
        self.proposal_matcher = det_utils.Matcher(
            #! 当iou大于fg_iou_thresh时，认为是正样本，小于bg_iou_thresh时，认为是负样本
            fg_iou_thresh, bg_iou_thresh,
            allow_low_quality_matches=True
        )
        self.fg_bg_sampler = det_utils.BalancedPositiveNegativeSampler(
            #! 正样本的比例 256, 0.5
            batch_size_per_image, positive_fraction
        )
        self._pre_nms_top_n = pre_nms_top_n
        self._post_nms_top_n = post_nms_top_n
        self.nms_thresh = nms_thresh
        self.min_size = 1e-3

    def pre_nms_top_n(self):
        if self.training:
            return self._pre_nms_top_n['training']
        return self._pre_nms_top_n['testing']

    def post_nms_top_n(self):
        if self.training:
            return self._post_nms_top_n['training']
        return self._post_nms_top_n['testing']

    def _get_top_n_idx(self, objectness, num_anchors_per_level):
        # type: (Tensor, List[int]) -> Tensor
        #! 获取每张预测特征图上预测概率排前pre_nms_top_n的anchors索引值
        r = [] # 记录每个预测特征层上预测目标概率前pre_nms_top_n的索引信息
        offset = 0
        # 遍历每个预测特征层上的预测目标概率信息
        for ob in objectness.split(num_anchors_per_level, 1):
            # 预测特征层上的预测的anchors个数
            num_anchors = ob.shape[1]
            pre_nms_top_n = min(num_anchors, self.pre_nms_top_n())
            _, top_n_idx = ob.topk(pre_nms_top_n, dim=1)
            r.append(top_n_idx + offset)
            offset += num_anchors
        return torch.cat(r, 1)

    def filter_proposals(self, proposals, objectness, image_shapes, num_anchors_per_level):
        # type: (Tensor, Tensor, List[Tuple[int, int]], List[int]) -> Tuple[List[Tensor], List[Tensor]]
        """
            筛除小boxes框，nms处理，根据预测概率获取前post_nms_top_n个目标
            Args:
                proposals: 预测的bbox坐标
                objectness: 预测的目标概率
                image_shapes: batch中每张图片的size信息
            num_anchors_per_level: 每个预测特征层上预测anchors的数目
        """
        num_images = proposals.shape[0]
        device = proposals.device

        objectness = objectness.detach().reshape(num_images, -1)

        levels = [
            torch.full((n, ), idx, dtype=torch.int64, device=device)
            for idx, n in enumerate(num_anchors_per_level)
        ]
        levels = torch.cat(levels, 0)
        levels = levels.reshape(1, -1).expand_as(objectness)

        top_n_idx = self._get_top_n_idx(objectness, num_anchors_per_level)
        image_range = torch.arange(num_images, device=device)
        #! batch_idx -> [Batch_size, 1]
        batch_idx = image_range[:, None]

        #! 这里得到排序后的anchors
        objectness = objectness[batch_idx, top_n_idx]
        levels = levels[batch_idx, top_n_idx]
        proposals = proposals[batch_idx, top_n_idx]

        final_boxes = []
        final_scores = []
        #* 遍历每张图像的相关信息
        for boxes, scores, lvl, img_shape in zip(proposals, objectness, levels, image_shapes):
            #* 将越界的box边界调整到图像边界
            boxes = box_ops.clip_boxes_to_image(boxes, img_shape)
            keep = box_ops.remove_small_boxes(boxes, self.min_size)
            boxes, scores, lvl = boxes[keep], scores[keep], lvl[keep]
            keep = box_ops.batched_nms(boxes, scores, lvl, self.nms_thresh)
            keep = keep[:self.post_nms_top_n()]
            boxes, scores = boxes[keep], scores[keep]
            final_boxes.append(boxes)
            final_scores.append(scores)
        return final_boxes, final_scores

    def assign_targets_to_anchors(self, anchors, targets):
        # type: (List[Tensor], List[Dict[str, Tensor]]) -> Tuple[List[Tensor], List[Tensor]]
        labels = []
        matched_gt_boxes = []
        for anchors_per_image, targets_per_image in zip(anchors, targets):
            gt_boxes = targets_per_image['boxes']
            if gt_boxes.numel() == 0:
                # Background image
                device = anchors_per_image.device
                matched_gt_box_per_image = torch.zeros(anchors_per_image.shape, dtype=torch.float32, device=device)
                labels_per_image = torch.zeros((anchors_per_image.shape[0], ), dtype=torch.float32, device=device)
            else:
                match_equality_matrix = box_ops.box_iou(gt_boxes, anchors_per_image)
                matched_idx = self.proposal_matcher(match_equality_matrix)
                matched_gt_box_per_image = gt_boxes[matched_idx.clamp(min=0)]
                labels_per_image = matched_idx >= 0
                #! 这里相当于把iou>=fg_iou_thresh的设置为1.0
                labels_per_image = labels_per_image.to(dtype=torch.float32)
                # 背景
                bg_indices = matched_idx == self.proposal_matcher.BELOW_LOW_THRESHOLD
                labels_per_image[bg_indices] = 0.0
                # 丢弃
                discard_indices = matched_idx == self.proposal_matcher.BETWEEN_THRESHOLDS
                labels_per_image[discard_indices] = -1.0

            labels.append(labels_per_image)
            matched_gt_boxes.append(matched_gt_box_per_image)
        return labels, matched_gt_boxes

    def compute_loss(self, objectness, pred_bbox_deltas, labels, regression_targets):
        # type: (Tensor, Tensor, List[Tensor], List[Tensor]) -> Tuple[Tensor, Tensor]
        #! 计算RPN损失，包括类别损失（前景与背景），bbox regression损失
        # 按照给定的batch_size_per_image, positive_fraction选择正负样本
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
        sampled_pos_inds = torch.nonzero(torch.cat(sampled_pos_inds, dim=0)).squeeze(1)
        sampled_neg_inds = torch.nonzero(torch.cat(sampled_neg_inds, dim=0)).squeeze(1)
        #! 正负样本拼接
        sampled_inds = torch.cat([sampled_pos_inds, sampled_neg_inds], dim=0)
        objectness = objectness.flatten()

        labels = torch.cat(labels, dim=0)
        regression_targets = torch.cat(regression_targets, dim=0)

        box_loss = det_utils.smooth_l1_loss(
            pred_bbox_deltas[sampled_pos_inds],
            regression_targets[sampled_pos_inds],
            beta=1 / 9,
            size_average=False,
        ) / (sampled_pos_inds.numel() + 1e-6)

        objectness_loss = F.binary_cross_entropy_with_logits(
            objectness[sampled_inds], labels[sampled_inds]
        )

        return objectness_loss, box_loss


    def forward(self, images, features, targets=None):
        # type: (ImageList, Dict[str, Tensor], Optional[List[Dict[str, Tensor]]]) -> Tuple[List[Tensor], Dict[str, Tensor]]
        # RPN does not need to compute the roi losses
        features = list(features.values())
        objectness, pred_bbox_deltas = self.head(features)
        #! anchors -> [Batch_size, anchor_num, 4]
        anchors = self.anchor_generator(images, features)
        # batch_size
        num_images = len(anchors)

        num_anchors_per_level_shape_tensors = [o[0].shape for o in objectness]
        #! 每个特征层的anchor数量
        num_anchors_per_level = [s[0] * s[1] * s[2] for s in num_anchors_per_level_shape_tensors]
        objectness, pred_bbox_deltas = concat_box_prediction_layers(objectness, pred_bbox_deltas)
        #* 将预测到的bbox regression参数应用到anchors得到最终的bbox坐标
        proposals = self.box_coder.decode(pred_bbox_deltas.detach(), anchors)
        proposals = proposals.view(num_images, -1, 4)

        boxes, scores = self.filter_proposals(proposals, objectness, images.image_sizes, num_anchors_per_level)
        losses = {}
        if self.training:
            assert targets is not None
            labels, matched_gt_boxes = self.assign_targets_to_anchors(anchors, targets)
            regression_targets = self.box_coder.encode(matched_gt_boxes, anchors)
            loss_objectness, loss_rpn_box_reg = self.compute_loss(
                objectness, pred_bbox_deltas, labels, regression_targets
            )
            losses = {
                "loss_objectness": loss_objectness,
                "loss_rpn_box_reg": loss_rpn_box_reg,
            }
        return boxes, losses


def permute_and_flatten(x, N, A, C, H, W):
    #! [N, Ax(C or 4), H, W]
    x = x.view(N, -1 , C , H, W)
    #! [N, A, C, H, W]
    x = x.permute(0, 3, 4, 1, 2)
    #! [N, H, W, A, C]
    x = x.reshape(N, -1, C)
    #! [N, H*W*A, C]
    return x


def concat_box_prediction_layers(objectness, pred_bbox_deltas):
    box_cls_flattened = []
    box_regression_flattened = []

    for box_cls_per_level, box_regression_per_level in zip(
        objectness, pred_bbox_deltas
    ):
        #! [batch_size, anchor_num_per_position * classes_num, H, W]
        #! 计算 RPN的proposal的时候，classes_num=1
        N, AxC, H, W = box_cls_per_level.shape
        #! box_regression_per_level -> [B, anchor_num_per_position * 4, H,  W]
        Ax4 = box_regression_per_level.shape[1]
        #! anchor_num_per_position = A
        A = Ax4 // 4
        C = AxC // A

        box_cls_per_level = permute_and_flatten(box_cls_per_level, N, A, C, H, W)
        #! box_cls_per_level -> [B, H*W*A, classes_num]
        box_cls_flattened.append(box_cls_per_level)

        #! [batch_size, anchor_num_per_position *  4, H, W] -> [B, H*W*A, 4]
        box_regression_per_level = permute_and_flatten(box_regression_per_level, N, A, 4, H, W)
        box_regression_flattened.append(box_regression_per_level)

    box_cls = torch.cat(box_cls_flattened, dim=1).flatten(0, -2) #start_dim, end_dim
    box_regression = torch.cat(box_regression_flattened, dim=1).reshape(-1, 4)

    return box_cls, box_regression
