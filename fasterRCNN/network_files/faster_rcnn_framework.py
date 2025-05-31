import sys, os
sys.path.append(os.path.dirname(__file__))
from rpn_function import AnchorsGenerator, RPNHead, RegionProposalNetwork
from roi_head import RoIHeads
from collections import OrderedDict
from torch import nn as nn
from torch.jit.annotations import Tuple, Dict, List, Optional
from torchvision.ops import MultiScaleRoIAlign
from torch import Tensor
import torch.nn.functional as F
import warnings, torch
from transform import GeneralizedRCNNTransform

class FasterRCNNBase(nn.Module):
    def __init__(self, backbone, rpn, roi_heads, transform):
        super().__init__()
        self.transform = transform
        self.backbone = backbone
        self.rpn = rpn
        self.roi_heads = roi_heads
        self._has_warned = False

    @torch.jit.unused
    def eager_outputs(self, losses, detections):
        # type: (Dict[str, Tensor], List[Dict[str, Tensor]]) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]
        if self.training:
            return losses
        return detections

    def forward(self, images, targets):
         # type: (List[Tensor], Optional[List[Dict[str, Tensor]]]) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        if self.training:
            assert targets is not None
            for target in targets:
                boxes = target["boxes"]
                if isinstance(boxes, torch.Tensor):
                    if len(boxes.shape) != 2 or boxes.shape[-1] != 4:
                        raise ValueError(f"Expected ground-truth boxes to be a tensor, got {boxes}")
                else:
                    raise ValueError(f"Expected ground-truth boxes to be a tensor, got {type(boxes)}")
        original_image_sizes: List[Tuple[int, int]] = []
        for img in images:
            val = img.shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))

        images, targets = self.transform(images, targets)
        features = self.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])
        proposals, proposal_losses = self.rpn(images, features, targets)
        #! 将RPN生成的标注信息输入faster RCNN后半部分
        detections, detector_losses = self.roi_heads(features, proposals, images.image_sizes, targets)
        detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)
        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        if torch.jit.is_scripting():
            if not self._has_warned:
                warnings.warn("Scripted FasterRCNN always returns a (Losses, Detections) tuple")
                self._has_warned = True
            return losses, detections
        else:
            return self.eager_outputs(losses, detections)

class TwoMLPHead(nn.Module):
    def __init__(self, in_channels, representation_size):
        super().__init__()
        self.fc6 = nn.Linear(in_channels, representation_size)
        self.fc7 = nn.Linear(representation_size, representation_size)

    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        return x


class FastRCNNPredictor(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.cls_score = nn.Linear(in_channels, num_classes)
        self.bbox_pred = nn.Linear(in_channels, num_classes * 4)

    def forward(self, x):
        if x.dim() == 4:
            assert list(x.shape[2:]) == [1, 1]
        x = x.flatten(start_dim=1)
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)
        return scores, bbox_deltas


class FasterRCNN(FasterRCNNBase):
    def __init__(self, backbone, num_classes=None,
                min_size=800, max_size=1333,
                image_mean=None, image_std=None,
                #rpn_parameter
                rpn_anchor_generator=None,
                rpn_head=None,
                rpn_pre_nms_top_n_train=2000, rpn_pre_nms_top_n_test=1000,        #* rpn中在nms处理前保留的proposal数
                rpn_post_nms_top_n_train=2000, rpn_post_nms_top_n_test=1000,       #* rpn中在nms处理后保留的proposal数
                rpn_nms_thresh=0.7,
                rpn_fg_iou_thresh=0.7, rpn_bg_iou_thresh=0.3,                       #* rpn中正负样本的iou阈值
                rpn_batch_size_per_image=256, rpn_positive_fraction=0.5,      #* 每张图片的样本数量和正样本的比例
                #roi_head_parameter
                box_roi_pool=None, box_head=None, box_predictor=None,
                box_score_thresh=0.05,
                box_nms_thresh=0.5,                                         #* nms阈值
                box_detections_per_img=100,                                 #* 对预测结果根据scores取前100
                box_fg_iou_thresh=0.5, box_bg_iou_thresh=0.5,               #* fasterRCNN计算正负样本的阈值
                box_batch_size_per_image=512,                               #* fasterRCNN计算误差时的样本数量
                box_positive_fraction=0.25,
                bbox_reg_weights=None,
                ):
        if not hasattr(backbone, "out_channels"):
            raise ValueError(
                "backbone should contain an attribute out_channels ")
        assert isinstance(rpn_anchor_generator, (AnchorsGenerator, type(None)))
        assert isinstance(box_roi_pool, (MultiScaleRoIAlign, type(None)))

        if num_classes is not None:
            if box_predictor is not None:
                raise ValueError("num_classes should be None when box_predictor "
                                "is specified")
        else:
            if box_predictor is None:
                raise ValueError("num_classes should not be None when box_predictor "
                                "is not specified")
        out_channels = backbone.out_channels

        if rpn_anchor_generator is None:
            anchor_size = ((32,), (64,), (128,), (256,), (512,))
            aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_size)
            rpn_anchor_generator = AnchorsGenerator(
                anchor_size, aspect_ratios
            )

        if rpn_head is None:
            rpn_head = RPNHead(
                out_channels, rpn_anchor_generator.num_anchors_per_location()[0]
            )

        rpn_pre_nms_top_n = dict(training=rpn_pre_nms_top_n_train, testing=rpn_pre_nms_top_n_test)
        rpn_post_nms_top_n = dict(training=rpn_post_nms_top_n_train, testing=rpn_post_nms_top_n_test)
        rpn = RegionProposalNetwork(
            rpn_anchor_generator, rpn_head,
            rpn_fg_iou_thresh, rpn_bg_iou_thresh,
            rpn_batch_size_per_image, rpn_positive_fraction,
            rpn_pre_nms_top_n, rpn_post_nms_top_n,
            rpn_nms_thresh
        )

        if box_roi_pool is None:
            box_roi_pool = MultiScaleRoIAlign(
                featmap_names = ["0", "1", "2", "3"],
                output_size = [7, 7],
                sampling_ratio = 2
            )

        if box_head is None:
            resolution = box_roi_pool.output_size[0]
            representation_size = 1024
            box_head = TwoMLPHead(
                out_channels * resolution ** 2,
                representation_size
            )

        if box_predictor is None:
            representation_size = 1024
            box_predictor = FastRCNNPredictor(
                representation_size,
                num_classes
            )

        roi_heads = RoIHeads(
            box_roi_pool, box_head, box_predictor,
            box_fg_iou_thresh, box_bg_iou_thresh,
            box_batch_size_per_image, box_positive_fraction,
            bbox_reg_weights,
            box_score_thresh, box_nms_thresh, box_detections_per_img
        )

        if image_mean is None:
            image_mean = [0.485, 0.456, 0.406]
        if image_std is None:
            image_std = [0.229, 0.224, 0.225]

        transform = GeneralizedRCNNTransform(min_size, max_size, image_mean, image_std)

        super().__init__(backbone, rpn, roi_heads, transform)
