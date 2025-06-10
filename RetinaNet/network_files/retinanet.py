import math, warnings, os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from collections import OrderedDict
from typing import Dict, List, Tuple, Optional, Union

import torch
from torch import nn, Tensor

from RetinaNet.network_files import det_utils
from RetinaNet.network_files.rpn_function import AnchorsGenerator
import RetinaNet.network_files.boxes as box_ops
from RetinaNet.network_files.losses import sigmoid_focal_loss
from RetinaNet.network_files.transform import GeneralizedRCNNTransform


def _sum(x: List[Tensor]) -> Tensor:
    res = x[0]
    for i in x[1:]:
        res += i
    return res

class RetinaNetClassificationHead(nn.Module):
    def __init__(self, in_channels, num_anchors, num_classes, prior_prob=0.01):
        super().__init__()

        #* class subnet是由四个3x3的卷积层(激活函数为ReLU) + 一个3x3的卷积层(分类器)
        conv = []
        for _ in range(4):
            conv.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1))
            conv.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*conv)

        self.cls_logits = nn.Conv2d(in_channels, num_anchors * num_classes, kernel_size=3, stride=1, padding=1)

        for layer in self.conv.children():
            if isinstance(layer, nn.Conv2d):
                torch.nn.init.normal_(layer.weight, mean=0, std=0.01)
                torch.nn.init.constant_(layer.bias, 0)

        torch.nn.init.normal_(self.cls_logits.weight, std=0.01)
        torch.nn.init.constant_(self.cls_logits.bias, -math.log((1 - prior_prob) / prior_prob))

        self.num_anchors = num_anchors
        self.num_classes = num_classes

        self.BETWEEN_THRESHOLDS = det_utils.Matcher.BETWEEN_THRESHOLDS

    def compute_loss(self, targets: List[Dict[str, Tensor]],
                     head_outputs: Dict[str, Tensor],
                     matched_idx: List[Tensor]) -> Tensor:
        losses = []
        cls_logits = head_outputs["cls_logits"]
        for targets_per_img, cls_logits_per_img, matched_idxs_per_img in zip(targets, cls_logits, matched_idx):
            #* 找出所有前景目标
            foreground_idx_per_img = torch.ge(matched_idxs_per_img, 0)
            num_foreground = foreground_idx_per_img.sum()

            gt_classes_target = torch.zeros_like(cls_logits_per_img)
            gt_classes_target[foreground_idx_per_img, targets_per_img["labels"][matched_idxs_per_img[foreground_idx_per_img]]] = 1.0

            #* 忽略iou在[0.4, 0.5)之间的anchors
            valid_idxs_per_img = torch.ne(matched_idxs_per_img, self.BETWEEN_THRESHOLDS)

            losses.append(sigmoid_focal_loss(
                cls_logits_per_img[valid_idxs_per_img],
                gt_classes_target[valid_idxs_per_img],
                reduction="sum"
            ) / max(1, num_foreground))

        return _sum(losses) / len(targets)

    def forward(self, x:Tensor) -> Tensor:
        all_cls_logits = []
        #* 遍历每个特征层
        for features in x:
            cls_logits = self.conv(features)
            cls_logits = self.cls_logits(cls_logits)

            N, _, H, W = cls_logits.shape
            #! view: [batch, num_anchors * num_classes, H, W] -> [batch, num_anchors, num_classes, H, W]
            cls_logits = cls_logits.view(N, -1, self.num_classes, H, W)
            #! permute: [B, A, C, H, W] -> [B, A, H, W, C]
            cls_logits = cls_logits.permute(0, 1, 3, 4, 2).contiguous()
            #! view: [B, A, H, W, C] -> [B, AHW, C]
            cls_logits = cls_logits.view(N, -1, self.num_classes)

            all_cls_logits.append(cls_logits)
        return torch.cat(all_cls_logits, dim=1)


class RetinaNetRegressionHead(nn.Module):
    __annotations__ = {
        'box_coder': det_utils.BoxCoder,
    }

    def __init__(self, in_channels, num_anchors):
        super().__init__()

        #* box subnet是由四个3x3的卷积层(激活函数为ReLU) + 一个3x3的卷积层(边界框回归器)
        conv = []
        for _ in range(4):
            conv.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1))
            conv.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*conv)

        self.bbox_reg = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=3, stride=1, padding=1)

        for layer in self.conv.children():
            if isinstance(layer, nn.Conv2d):
                torch.nn.init.normal_(layer.weight, std=0.01)
                torch.nn.init.zeros_(layer.bias)

        self.bbox_coder = det_utils.BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))

    def compute_loss(self, targets: List[Dict[str, Tensor]], head_outputs:Dict[str, Tensor],
                     anchors: List[Tensor], matched_idx: List[Tensor]) -> Tensor:
        losses = []
        bbox_regression = head_outputs["bbox_regression"]

        for targets_per_img, bbox_regression_per_img, anchors_per_img, matched_idxs_per_img in zip(targets, bbox_regression, anchors, matched_idx):
            #* 找出所有前景目标
            foreground_idx_per_img = torch.ge(matched_idxs_per_img, 0)
            num_foreground = foreground_idx_per_img.numel()

            #* 只选择前景目标对应的box
            matched_idxs_per_img = targets_per_img["boxes"][matched_idxs_per_img[foreground_idx_per_img]]
            bbox_regression_per_img = bbox_regression_per_img[foreground_idx_per_img, :]
            anchors_per_img = anchors_per_img[foreground_idx_per_img, :]

            targets_regression = self.bbox_coder.encode_single(matched_idxs_per_img, anchors_per_img)
            losses.append(
                torch.nn.functional.l1_loss(
                    bbox_regression_per_img,
                    targets_regression,
                    reduction="sum"
                ) / max(1, num_foreground)
            )
        return _sum(losses) / max(1, len(targets))

    def forward(self, x: List[Tensor]) -> Tensor:
        all_bbox_regression = []

        for features in x:
            bbox_regression = self.conv(features)
            bbox_regression = self.bbox_reg(bbox_regression)

            N, _, H, W = bbox_regression.shape
            #! view: [B, Anchor_num * 4, H, W] -> [B, A, 4, H, W]
            bbox_regression = bbox_regression.view(N, -1, 4, H, W)
            #! permute: [B, A, 4, H, W] -> [B, A, H, W, 4]
            bbox_regression = bbox_regression.permute(0, 1, 3, 4, 2).contiguous()
            #! view: [B, A, H, W, 4] -> [B, AHW, 4]
            bbox_regression = bbox_regression.view(N, -1, 4)

            all_bbox_regression.append(bbox_regression)

        return torch.cat(all_bbox_regression, dim=1)

class RetinaNetHead(nn.Module):
    def __init__(self, in_channels, num_anchors, num_classes):
        super().__init__()
        self.classification_head = RetinaNetClassificationHead(in_channels, num_anchors, num_classes)
        self.regression_head = RetinaNetRegressionHead(in_channels, num_anchors)

    def compute_loss(self, targets: List[Dict[str, Tensor]], head_outputs: Dict[str, Tensor],
                    anchors: List[Tensor], matched_idx: List[Tensor]):
        return {
            "classification": self.classification_head.compute_loss(targets, head_outputs, anchors, matched_idx),
            "bbox_regression": self.regression_head.compute_loss(targets, head_outputs, anchors, matched_idx),
        }

    def forward(self, x):
        return {
            "cls_logits": self.classification_head(x),
            "bbox_regression": self.regression_head(x),
        }


class RetinaNet(nn.Module):
    __annotations__ = {
        'box_coder': det_utils.BoxCoder,
        'proposal_matcher': det_utils.Matcher,
    }
    def __init__(self, backbone, num_classes, min_size=800, max_size=1333,
                image_mean=None, image_std=None, anchor_generator=None,
                head=None, proposal_matcher=None, score_thresh=0.05,
                nms_thresh=0.5, detection_per_img=100,
                fg_iou_thresh=0.5, bg_iou_thresh=0.4,
                topk_candidates=1000):
        super().__init__()
        if not hasattr(backbone, "out_channels"):
            raise ValueError(
                "backbone should contain an attribute out_channels "
                "specifying the number of output channels (assumed to be the "
                "same for all the levels)"
            )
        self.backbone = backbone
        assert isinstance(anchor_generator, (AnchorsGenerator, type(None)))
        if anchor_generator is None:
            anchor_size = tuple((x, int(x * 2 ** (1.0 / 3)), int(x * 2 ** (2.0 / 3))) \
                                for x in [32, 64, 128, 256, 512])
            aspect_ratio = ((0.5, 1.0, 2.0),) * len(anchor_size)
            anchor_generator = AnchorsGenerator(anchor_size, aspect_ratio)

        self.anchor_generator = anchor_generator

        if head is None:
            head = RetinaNetHead(
                backbone.out_channels,
                anchor_generator.num_anchors_per_location()[0],
                num_classes
            )

        self.head = head

        if proposal_matcher is None:
            proposal_matcher = det_utils.Matcher(
                fg_iou_thresh, bg_iou_thresh,
                allow_low_quality_matches=True
            )
        self.proposal_matcher = proposal_matcher
        self.box_coder = det_utils.BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))

        if image_mean is None:
            image_mean = [0.485, 0.456, 0.406]
        if image_std is None:
            image_std = [0.229, 0.224, 0.225]

        self.transform = GeneralizedRCNNTransform(min_size, max_size, image_mean, image_std)

        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.detection_per_img = detection_per_img

        self.topk_candidates = topk_candidates

        self._has_warned = False

    @torch.jit.unused
    def eager_outputs(self, losses, detections):
        if self.training:
            return losses
        return detections

    def compute_loss(self, targets, head_outputs, anchors, matched_idx):
        matched_idx = []
        for anchors_per_img, targets_per_img in zip(anchors, targets):
            if targets_per_img["boxes"].numel() == 0:
                matched_idx.append(torch.full((anchors_per_img.size(0), ), -1, dtype=torch.int64, device=anchors_per_img.device))
                continue

            match_quality_matrix = box_ops.box_iou(targets_per_img["boxes"], anchors_per_img)
            matched_idx.append(self.proposal_matcher(match_quality_matrix))
        return self.head.compute_loss(targets, head_outputs, anchors, matched_idx)

    def postprocess_detections(self, head_output, anchors, image_shapes):
        # type: (Dict[str, List[Tensor]], List[List[Tensor]], List[Tuple[int, int]]) -> List[Dict[str, Tensor]]
        class_logits = head_output["cls_logits"]
        box_regression = head_output["bbox_regression"]

        num_img = len(image_shapes)

        detections: List[Dict[str, Tensor]] = []

        for index in range(num_img):
            box_regression_per_img = [br[index] for br in box_regression]
            logits_per_img = [cl[index] for cl in class_logits]
            anchors_per_img, image_shape = anchors[index], image_shapes[index]

            img_boxes = []
            img_scores = []
            img_labels = []

            for box_regression_per_level, logits_per_level, anchors_per_level in \
                    zip(box_regression_per_img, logits_per_img, anchors_per_img):
                num_classes = logits_per_level.shape[-1]

                # remove low scoring boxes
                # 移除低概率的目标
                scores_per_level = torch.sigmoid(logits_per_level).flatten()
                keep_idxs = torch.gt(scores_per_level, self.score_thresh)  # gt: >
                scores_per_level = scores_per_level[keep_idxs]
                topk_idxs = torch.where(keep_idxs)[0]

                # keep only topk scoring predictions
                # 在每个level上只取前topk个目标
                num_topk = min(self.topk_candidates, topk_idxs.size(0))
                scores_per_level, idxs = scores_per_level.topk(num_topk)
                topk_idxs = topk_idxs[idxs]

                anchor_idxs = topk_idxs // num_classes
                labels_per_level = topk_idxs % num_classes

                boxes_per_level = self.box_coder.decode_single(box_regression_per_level[anchor_idxs],
                                                               anchors_per_level[anchor_idxs])
                boxes_per_level = box_ops.clip_boxes_to_image(boxes_per_level, image_shape)

                img_boxes.append(boxes_per_level)
                img_scores.append(scores_per_level)
                img_labels.append(labels_per_level)

            img_boxes = torch.cat(img_boxes, dim=0)
            img_scores = torch.cat(img_scores, dim=0)
            img_labels = torch.cat(img_labels, dim=0)

            # non-maximum suppression
            keep = box_ops.batched_nms(img_boxes, img_scores, img_labels, self.nms_thresh)
            keep = keep[:self.detections_per_img]

            detections.append({
                "boxes": img_boxes[keep],
                "scores": img_scores[keep],
                "labels": img_labels[keep]
            })

        return detections

    def forward(self, images, targets=None):
        # type: (List[Tensor], Optional[List[Dict[str, Tensor]]]) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")

        if self.training:
            assert targets is not None
            # check targets info
            for target in targets:
                boxes = target["boxes"]
                if isinstance(boxes, torch.Tensor):
                    if len(boxes.shape) != 2 or boxes.shape[-1] != 4:
                        raise ValueError("Expected target boxes to be a tensor"
                                         "of shape [N, 4], got {:}.".format(boxes.shape))
                else:
                    raise ValueError("Expected target boxes to be of type "
                                     "Tensor, got {:}.".format(type(boxes)))

        # get the original images sizes
        original_img_sizes: List[Tuple[int, int]] = []
        for img in images:
            val = img.shape[-2:]
            assert len(val) == 2
            original_img_sizes.append((val[0], val[1]))  # h, w

        # transform the input
        images, targets = self.transform(images, targets)

        # Check for degenerate boxes
        # TODO: Move this to a function
        if targets is not None:
            for target_idx, target in enumerate(targets):
                boxes = target["boxes"]
                degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
                if degenerate_boxes.any():
                    # print the first degenerate box
                    bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                    degen_bb: List[float] = boxes[bb_idx].tolist()
                    raise ValueError("All bounding boxes should have positive height and width."
                                     " Found invalid box {} for target at index {}."
                                     .format(degen_bb, target_idx))

        # get the features from the backbone
        features = self.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])

        features = list(features.values())

        # compute the retinanet heads outputs using the features
        head_outputs = self.head(features)

        # create the set of anchors
        anchors = self.anchor_generator(images, features)

        losses = {}
        detections: List[Dict[str, Tensor]] = []
        if self.training:
            assert targets is not None
            losses = self.compute_loss(targets, head_outputs, anchors)
        else:
            # recover level sizes
            num_anchors_per_level = [x.size(2) * x.size(3) for x in features]
            HW = 0
            for v in num_anchors_per_level:
                HW += v
            HWA = head_outputs["cls_logits"].size(1)
            A = HWA // HW
            num_anchors_per_level = [hw * A for hw in num_anchors_per_level]

            # split outputs per level
            split_head_outputs: Dict[str, List[Tensor]] = {}
            for k in head_outputs:
                split_head_outputs[k] = list(head_outputs[k].split(num_anchors_per_level, dim=1))
            split_anchors = [list(a.split(num_anchors_per_level)) for a in anchors]

            # compute the detections
            detections = self.postprocess_detections(split_head_outputs, split_anchors, images.image_sizes)
            detections = self.transform.postprocess(detections, images.image_sizes, original_img_sizes)

        if torch.jit.is_scripting():
            if not self._has_warned:
                warnings.warn("RetinaNet always returns a (Losses, Detections) tuple in scripting")
                self._has_warned = True
            return losses, detections
        return self.eager_outputs(losses, detections)
