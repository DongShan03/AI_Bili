import torch
import torch.nn as nn
import torch.nn.functional as F
import det_utils
import boxes as box_ops
from torch import Tensor
from torch.jit.annotations import List, Tuple, Optional, Dict

def fastrcnn_loss(class_logits, box_regression, labels, regression_targets):
    # type: (Tensor, Tensor, List[Tensor], List[Tensor]) -> Tuple[Tensor, Tensor]
    labels = torch.cat(labels, dim=0)
    regression_targets = torch.cat(regression_targets, dim=0)
    #* 计算类别损失
    classification_loss = F.cross_entropy(class_logits, labels)
    #* 返回标签类别大于0的索引
    sampled_pos_inds_subset = torch.where(torch.gt(labels, 0))[0]
    labels_pos = labels[sampled_pos_inds_subset]

    N, num_classes = class_logits.shape
    box_regression = box_regression.reshape(N, -1, 4)
    box_loss = det_utils.smooth_l1_loss(
        box_regression[sampled_pos_inds_subset, labels_pos],
        regression_targets[sampled_pos_inds_subset],
        beta=1/9, size_average=False
    ) / labels.numel()

    return classification_loss, box_loss


class RoIHeads(nn.Module):
    __annotations__ = {
        'box_coder': det_utils.BoxCoder,
        'proposal_matcher': det_utils.Matcher,
        'fg_bg_sampler': det_utils.BalancedPositiveNegativeSampler,
    }
    def __init__(self,
                box_roi_pool,   # Multi-scale RoIAlign pooling
                box_head,       # TwoMLPHead
                box_predictor,  # FastRCNNPredictor
                # Faster R-CNN training
                fg_iou_thresh, bg_iou_thresh,  # default: 0.5, 0.5
                batch_size_per_image, positive_fraction,  # default: 512, 0.25
                bbox_reg_weights,  # None
                # Faster R-CNN inference
                score_thresh,        # default: 0.05
                nms_thresh,          # default: 0.5
                detection_per_img):  # default: 100
        super().__init__()
        self.box_similarity = box_ops.box_iou
        self.proposal_matcher = det_utils.Matcher(
            fg_iou_thresh, bg_iou_thresh,
            allow_low_quality_matches=True
        )
        self.fg_bg_sampler = det_utils.BalancedPositiveNegativeSampler(
            batch_size_per_image, positive_fraction
        )
        if bbox_reg_weights is None:
            bbox_reg_weights = (10., 10., 5., 5.)
        self.box_coder = det_utils.BoxCoder(bbox_reg_weights)
        self.box_roi_pool = box_roi_pool
        self.box_head = box_head
        self.box_predictor = box_predictor

        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.detection_per_img = detection_per_img

    def add_gt_proposals(self, proposals, gt_boxes):
        # type: (List[Tensor], List[Tensor]) -> List[Tensor]
        proposals = [
            torch.cat((proposal, gt_boxe))
            for proposal, gt_boxe in zip(proposals, gt_boxes)
        ]
        return proposals

    def assign_targets_to_proposals(self, proposals, gt_boxes, gt_labels):
        # type: (List[Tensor], List[Tensor], List[Tensor]) -> Tuple[List[Tensor], List[Tensor]]
        #* 为每个proposal匹配对应的gt_box，并划分到正负样本中
        matched_idxs = []
        labels = []
        #* 遍历每张图像的proposals, gt_boxes, gt_labels信息
        for proposals_per_image, gt_boxes_per_image, gt_labels_per_image in zip(proposals, gt_boxes, gt_labels):
            if gt_boxes_per_image.numel() == 0: #* 如果当前图像没有gt_box,为背景
                # Background image
                device = proposals_per_image.device
                clamped_matched_idxs_per_image  = torch.zeros(
                    (proposals_per_image.shape[0],), dtype=torch.int64, device=device
                )
                labels_per_image = torch.zeros(
                    (proposals_per_image.shape[0],), dtype=torch.int64, device=device
                )
            else:
                # 计算proposal与每个gt_box的iou重合度
                match_quality_matrix = box_ops.box_iou(gt_boxes_per_image, proposals_per_image)
                # 计算proposal与每个gt_box匹配的iou最大值，并记录索引，
                # iou < low_threshold索引值为 -1， low_threshold <= iou < high_threshold索引值为 -2
                matched_idxs_per_image = self.proposal_matcher(match_quality_matrix)
                # 限制最小值，防止匹配标签时出现越界的情况
                # 注意-1, -2对应的gt索引会调整到0,获取的标签类别为第0个gt的类别（实际上并不是）,后续会进一步处理
                clamped_matched_idxs_per_image = matched_idxs_per_image.clamp(min=0)
                # 获取proposal匹配到的gt对应标签
                labels_per_image = gt_labels_per_image[clamped_matched_idxs_per_image]
                labels_per_image = labels_per_image.to(dtype=torch.int64)

                # 将gt索引为-1的类别设置为0，即背景，负样本
                bg_inds = matched_idxs_per_image == self.proposal_matcher.BELOW_LOW_THRESHOLD  # -1
                labels_per_image[bg_inds] = 0

                # 将gt索引为-2的类别设置为-1, 即废弃样本
                ignore_inds = matched_idxs_per_image == self.proposal_matcher.BETWEEN_THRESHOLDS  # -2
                labels_per_image[ignore_inds] = -1  # -1 is ignored by sampler

            matched_idxs.append(clamped_matched_idxs_per_image)
            labels.append(labels_per_image)
        return matched_idxs, labels

    def subsample(self, labels):
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
        sampled_inds = []
        # 遍历每张图片的正负样本索引
        for img_idx, (pos_inds_img, neg_inds_img) in enumerate(zip(sampled_pos_inds, sampled_neg_inds)):
            # 记录所有采集样本索引（包括正样本和负样本）
            img_sampled_inds = torch.where(pos_inds_img | neg_inds_img)[0]
            sampled_inds.append(img_sampled_inds)
        return sampled_inds

    def check_targets(self, targets):
        # type: (Optional[List[Dict[str, Tensor]]]) -> None
        assert targets is not None
        assert all(["boxes" in t for t in targets])
        assert all(["labels" in t for t in targets])

    def select_training_samples(self,
                                proposals,  # type: List[Tensor]
                                targets     # type: Optional[List[Dict[str, Tensor]]]
                                ):
        # type: (...) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]
        #! 划分正负样本，统计对应gt的标签以及边界框回归信息
        #! list元素个数为batch_size
        self.check_targets(targets)
        assert targets is not None
        dtype = proposals[0].dtype
        device = proposals[0].device

        gt_boxes = [t["boxes"].to(dtype) for t in targets]
        gt_labels = [t["labels"] for t in targets]
        #! gt_boxes: [batch_size, num_gt, 4]
        #! gt_labels: [batch_size, num_gt]
        #* 将gt_boxes拼接到proposal后面,增加正样本的数量
        proposals = self.add_gt_proposals(proposals, gt_boxes)

        #* 为每个proposal匹配对应的gt_box，并划分到正负样本中
        matched_idxs, labels = self.assign_targets_to_proposals(proposals, gt_boxes, gt_labels)
        #* 按给定数量和比例采样正负样本
        sampled_inds = self.subsample(labels)
        matched_gt_boxes = []
        num_images = len(proposals)
        for img_id in range(num_images):
            img_sampled_inds = sampled_inds[img_id]
            proposals[img_id] = proposals[img_id][img_sampled_inds]
            labels[img_id] = labels[img_id][img_sampled_inds]
            matched_idxs[img_id] = matched_idxs[img_id][img_sampled_inds]

            gt_boxes_in_image = gt_boxes[img_id]
            if gt_boxes_in_image.numel() == 0:
                gt_boxes_in_image = torch.zeros((1, 4), dtype=dtype, device=device)
            matched_gt_boxes.append(gt_boxes_in_image[matched_idxs[img_id]])
        regression_targets = self.box_coder.encode(matched_gt_boxes, proposals)
        return proposals, labels, regression_targets

    def postprocess_detections(self, class_logits, box_regression, proposals, image_shapes):
        #  type: (Tensor, Tensor, List[Tensor], List[Tuple[int, int]]) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]
        """
            (1)根据proposal以及预测的回归参数计算出最终bbox坐标
            (2)对预测类别结果进行softmax处理
            (3)裁剪预测的boxes信息，将越界的坐标调整到图片边界上
            (4)移除所有背景信息
            (5)移除低概率目标
            (6)移除小尺寸目标
            (7)执行nms处理，并按scores进行排序
            (8)根据scores排序返回前topk个目标
        """
        device = class_logits.device
        num_classes = class_logits.shape[-1]
        #* 获取每张图片的预测bbox数量
        boxes_per_image = [boxes_in_image.shape[0] for boxes_in_image in proposals]
        #* 根据proposal以及预测的回归参数计算出最终的bbox坐标
        pred_boxes = self.box_coder.decode(box_regression, proposals)
        #* 对预测类别结果进行softmax处理
        pred_scores = F.softmax(class_logits, -1)
        #* 根据每张图像的预测bbox数量划分解惑
        pred_boxes_list = pred_boxes.split(boxes_per_image, 0)
        pred_scores_list = pred_scores.split(boxes_per_image, 0)

        all_boxes = []
        all_scores = []
        all_labels = []
        for boxes, scores, image_shape in zip(pred_boxes_list, pred_scores_list, image_shapes):
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)
            labels = torch.arange(num_classes, device=device)
            labels = labels.view(1, -1).expand_as(scores)

            # 移除索引为0的类别 即背景
            boxes, scores, labels = boxes[:, 1:], scores[:, 1:], labels[:, 1:]

            boxes, scores, labels = boxes.reshape(-1, 4), scores.reshape(-1), labels.reshape(-1)

            #* 移除低概率目标
            inds = torch.where(torch.gt(scores, self.score_thresh))[0]
            boxes, scores, labels = boxes[inds], scores[inds], labels[inds]

            #* 移除小目标
            keep = box_ops.remove_small_boxes(boxes, min_size=1.)
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            #* nms处理
            keep = box_ops.batched_nms(boxes, scores, labels, self.nms_thresh)
            #* 返回Topk
            keep = keep[:self.detection_per_img]
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)

        return all_boxes, all_scores, all_labels

    def forward(self, features, proposals, image_shapes, targets=None):
        if targets is not None:
            for t in targets:
                floating_point_types = (torch.float, torch.double, torch.half)
                assert t["boxes"].dtype in floating_point_types
                assert t["labels"].dtype == torch.int64
        if self.training:
            proposals, labels, regression_targets = \
                self.select_training_samples(proposals, targets)
        else:
            labels = None
            regression_targets = None

        #! box_features_shape: [num_proposals, channel, height, width]
        box_features = self.box_roi_pool(features, proposals, image_shapes)
        #! box_features_shape: [num_proposals, representation_size]
        box_features = self.box_head(box_features)

        class_logits, box_regression = self.box_predictor(box_features)

        result = torch.jit.annotate(List[Dict[str, torch.Tensor]], [])
        losses = {}
        if self.training:
            assert labels is not None and regression_targets is not None
            loss_classifier, loss_box_reg = fastrcnn_loss(
                class_logits, box_regression, labels, regression_targets
            )
            losses = {
                "loss_classifier": loss_classifier,
                "loss_box_reg": loss_box_reg
            }
        else:
            boxes, scores, labels = self.postprocess_detections(class_logits, box_regression, proposals, image_shapes)
            num_images = len(proposals)
            for i in range(num_images):
                result.append(
                    {
                        "boxes": boxes[i],
                        "labels": labels[i],
                        "scores": scores[i],
                    }
                )

        return result, losses
