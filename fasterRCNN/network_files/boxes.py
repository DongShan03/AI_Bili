import torch
from torch import Tensor
from torch.jit.annotations import List, Tuple, Dict
import torchvision

def nms(boxes, scores, iou_threshold):
    # type: (Tensor, Tensor, float) -> Tensor
    """
    Parameters
    ----------
    boxes : Tensor[N, 4])
        boxes to perform NMS on. They
        are expected to be in (x1, y1, x2, y2) format
    scores : Tensor[N]
        scores for each one of the boxes
    iou_threshold : float
        discards all overlapping
        boxes with IoU > iou_threshold

    Returns
    -------
    keep : Tensor
        int64 tensor with the indices
        of the elements that have been kept
        by NMS, sorted in decreasing order of scores
    """
    return torchvision.ops.nms(boxes, scores, iou_threshold)

def batched_nms(boxes, scores, idxs, iou_threshold):
    # type: (Tensor, Tensor, Tensor, float) -> Tensor
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)
    # 获取所有boxes中最大的坐标值（xmin, ymin, xmax, ymax）
    max_coordinate = boxes.max()

    # 为每一层生成一个很大的偏移量
    # 这里的to只是让生成tensor的dytpe和device与boxes保持一致
    offsets = idxs.to(boxes) * (max_coordinate + 1)
    # boxes加上对应层的偏移量后，保证不同类别/层之间boxes不会有重合的现象
    boxes_for_nms = boxes + offsets[:, None]
    keep = nms(boxes_for_nms, scores, iou_threshold)
    return keep

def remove_small_boxes(boxes, min_size):
    # type: (Tensor, float) -> Tensor
    #! 移除宽高小于指定阈值的索引
    ws, hs = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]  # 预测boxes的宽和高
    # keep = (ws >= min_size) & (hs >= min_size)  # 当满足宽，高都大于给定阈值时为True
    keep = torch.logical_and(torch.ge(ws, min_size), torch.ge(hs, min_size))
    # nonzero(): Returns a tensor containing the indices of all non-zero elements of input
    # keep = keep.nonzero().squeeze(1)
    keep = torch.where(keep)[0]
    return keep

def clip_boxes_to_image(boxes, size):
    # type: (Tensor, Tuple[int, int]) -> Tensor
    dim = boxes.dim()
    boxes_x = boxes[..., 0::2]
    boxes_y = boxes[..., 1::2]
    height, width = size

    boxes_x = boxes_x.clamp(min=0, max=width)
    boxes_y = boxes_y.clamp(min=0, max=height)

    clipped_boxes = torch.stack((boxes_x, boxes_y), dim=dim)
    return clipped_boxes.reshape(boxes.shape)


def box_area(boxes):
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def box_iou(boxes1, boxes2):
    #! boxes1:(N,4) boxes2:(M,4)
    #! area1:(N,) area2:(M,)
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)
    #! boxes1[:, None, :2] -> (N, 1, 2)
    #! boxes2[:, 2:] -> (M, 2)
    #! lt -> (N, M, 2)
    #! wh -> (N, M, 2)
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    #! inter -> (N, M)
    inter = wh[:, :, 0] * wh[:, :, 1]
    #! area1[:, None] -> (N, 1)
    iou = inter / (area1[:, None] + area2 - inter)
    return iou
