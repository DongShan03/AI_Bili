import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import torch.nn.functional as F
import math, random
import albumentations as A
import numpy as np
import cv2
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from YOLOv5.utils.utils import *

def scale_img(img, ratio=1.0, same_shape=False, gs=32):  # img(16,3,256,416)
    if ratio == 1.0:
        return img
    h, w = img.shape[2:]
    s = (int(h * ratio), int(w * ratio))
    img = F.interpolate(img, size=s, mode="bilinear", align_corners=False)
    if not same_shape:
        h, w = (math.ceil(x * ratio / gs) * gs for x in (h, w))
    return F.pad(img, [0, w - s[1], 0, h - s[0]], value=0.447)

IMAGENET_MEAN = 0.485, 0.456, 0.406  # RGB mean
IMAGENET_STD = 0.229, 0.224, 0.225  # RGB standard deviation

class Albumentations:
    def __init__(self, size=640):
        self.transform = None
        T = [
            A.RandomResizedCrop(height=size, width=size, scale=(0.8, 1.0), ratio=(0.9, 0.11), p=0.0),
            A.Blur(p=0.01),
            A.MedianBlur(p=0.01),
            A.ToGray(p=0.01),
            A.CLAHE(p=0.01),
            A.RandomBrightnessContrast(p=0.1),
            A.RandomGamma(p=0.05),
            A.ImageCompression(quality_lower=75, quality_upper=95, p=0.1),
        ]
        self.transform = A.Compose(T, bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]))

    def __call__(self, im, labels, p=1.0):
        if self.transform and random.random() < p:
            new = self.transform(image=im, bboxes=labels[:, 1:], class_labels=labels[:, 0])
            im, labels = new["image"], np.array([[c, *box] for c, box in zip(new["class_labels"], new["bboxes"])])
        return im, labels


def normalize(x, mean=IMAGENET_MEAN, std=IMAGENET_STD, inplace=False):
    return TF.normalize(x, mean, std=std, inplace=inplace)

def denormalize(x, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    for i in range(3):
        x[:, i] = x[:, i] * std[i] + mean[i]
    return x

def augment_hsv(img, h_gain=0.5, s_gain=0.5, v_gain=0.5):
    r = np.random.uniform(-1, 1, 3) * [h_gain, s_gain, v_gain] + 1
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    dtype = img.dtype

    x = np.arange(0, 256, dtype=np.int16)
    lut_hue = ((x * r[0]) % 100).astype(dtype)
    lue_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lue_val = np.clip(x * r[2], 0, 255).astype(dtype)

    img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lue_sat), cv2.LUT(val, lue_val))).astype(dtype)
    cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)

def hist_equalize(im, clahe=True, bgr=False):
    #* 均衡图像直方图
    yuv = cv2.cvtColor(im, cv2.COLOR_BGR2YUV if bgr else cv2.COLOR_RGB2YUV)
    if clahe:
        c = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        yuv[:, :, 0] = c.apply(yuv[:, :, 0])
    else:
        yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB if bgr else cv2.COLOR_YUV2RGB)

def replicate(im, labels):
    #* 复制图像中一半的小物体标签，用于数据扩增
    h, w = im.shape[:2]
    boxes = labels[1:].astype(int)
    x1, y1, x2, y2 = boxes.T
    s = ((x2 - x1) + (y2 - y1)) * 0.5  # area
    for i in s.argsort()[:, round(s.size * 0.5)]:
        x1b, y1b, x2b, y2b = boxes[i]
        bh, bw = y2b - y1b, x2b - x1b
        yc, xc = int(random.uniform(0, h - bh)), int(random.uniform(0, w - bw))  # offset x, y
        x1a, y1a, x2a, y2a = [xc, yc, xc + bw, yc + bh]
        im[y1a:y2a, x1a:x2a] = im[y1b:y2b, x1b:x2b]
        labels = np.append(labels, [[labels[i, 0], x1a, y1a, x2a, y2a]], axis=0)

    return im, labels

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False,
            scaleup=True, auto_size=32):
    #* 输入的img是由cv2读取的 是BGR格式
    #* shape -> h, w
    shape = img.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    #* 计算缩放因子 使得图片最长边可以放入新尺寸
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)

    ratio = r, r
    #* 未填充的缩放后的图片尺寸
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    #* 需要填充的像素值
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    if auto:
        #* 计算余数
        dw, dh = np.mod(dw, auto_size), np.mod(dh, auto_size)
    elif scaleFill:
        #* 这里直接把图片变形缩放到填满尺寸
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]

    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, ratio, (dw, dh)

def random_perspective(img, targets=(), segments=(), degrees=10, translate=.1, scale=.1, shear=10, perspective=0.0, border=(0, 0)):
    height = img.shape[0] + border[0] * 2
    width = img.shape[1] + border[1] * 2
    #* [h, w, c]

    #* 中心
    C = np.eye(3)
    C[0, 2] = -img.shape[1] / 2
    C[1, 2] = -img.shape[0] / 2

    #* 透视
    P = np.eye(3)
    P[2, 0] = random.uniform(-perspective, perspective)
    P[2, 1] = random.uniform(-perspective, perspective)

    #* 旋转缩放
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    s = random.uniform(1 - scale, 1 + scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    #* 错切
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)

    #* 平移
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height

    #* 计算总变换矩阵
    M = T @ S @ R @ P @ C
    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():
        if perspective:
            img = cv2.warpPerspective(img, M, dsize=(width, height), borderValue=(114, 114, 114))
        else:
            img = cv2.warpAffine(img, M[:2], dsize=(width, height), borderValue=(114, 114, 114))

    #* 标签坐标转换
    n = len(targets)
    if n:
        use_segments = any(x.any() for x in segments) and len(segments) == n
        new = np.zeros((n, 4))
        if use_segments:
            segments = resample_segments(segments)
            for i, segment in enumerate(segments):
                xy = np.ones((len(segment), 3))
                xy[:, :2] = segment
                xy = xy @ M.T  # transform
                xy = xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]

                new[i] = segment2box(xy, width, height)
        else:
            #! xy -> [num_targets * 4(x1y1x2y2), 3]
            xy = np.ones((n * 4, 3))
            #! 提取真实的横纵坐标到第2维度，reshape后第二维度存放的是每个bbox的四个角点坐标
            xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
            xy = xy @ M.T  # transform
            #* 如果有透视变换
            if perspective:
                #* 对于透视变换，需要除以z坐标
                xy = (xy[:, :2] / xy[:, 2:3]).reshape(n, 8)
            else:
                xy = xy[:, :2].reshape(n, 8)

            #* 提取映射后的四个点的横坐标和纵坐标
            x = xy[:, [0, 2, 4, 6]]
            y = xy[:, [1, 3, 5, 7]]
            xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T
            #* 限制在图像范围内
            new[:, [0, 2]] = xy[:, [0, 2]].clip(0, width)
            new[:, [1, 3]] = xy[:, [1, 3]].clip(0, height)

        #* 筛选符合条件的候选框
        i = box_candidates(box1=targets[:, 1:5].T * s, box2=new.T, area_thr=0.01 if use_segments else 0.10)
        targets = targets[i]
        targets[:, 1:5] = new[i]

    return img, targets

def box_candidates(box1, box2, wh_thr=2, ar_chr=20, area_thr=0.1):
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    ar = np.maximum(w2 / (h2 + 1e-16), h2 / (w2 + 1e-16))  # aspect ratio
    return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + 1e-16) > area_thr) & (ar < ar_chr)

def copy_paste(im, labels, segments, p=0.5):
    #* Applies Copy-Paste augmentation by flipping and merging segments and labels on an image.
    #* 通过反转 合并分割语义和标签
    n = len(segments)
    if p and n:
        h, w, c = im.shape
        im_new = np.zeros(im.shape, np.uint8)
        for j in random.sample(range(n), k=round(p * n)):
            l, s = labels[j], segments[j]
            #* 新的box由label水平翻转得到
            box = w - l[3], l[2], w - l[1], l[4]

            ioa = bbox_ioa(box, labels[:, 1:5])
            if (ioa < 0.30).all():
                #* 允许和已有标签的IOA小于0.3
                labels = np.concatenate((labels, [[l[0], *box]]), 0)
                segments.append(np.concatenate((w - s[:, 0:1], s[:, 1:2]), 1))
                cv2.drawContours(im_new, [segments[j].astype(np.int32)], -1, (1, 1, 1), cv2.FILLED)

        result = cv2.flip(im, 1)
        i = cv2.flip(im_new, 1).astype(bool)
        im[i] = result[i]
    return im, labels, segments

def mixup(im, labels, im2, labels2):
    r = np.random.beta(32.0, 32.0)  # mixup ratio, alpha=beta=32.0
    im = (im * r + im2 * (1 - r)).astype(np.uint8)
    labels = np.concatenate((labels, labels2), 0)
    return im, labels

def cutout(im, labels, p=0.5):
    if random.random() < p:
        #* 用随机尺寸的遮挡放在原图中，同时筛选出和遮挡交集比较小的label
        h, w = im.shape[:2]
        scales = [0.5] * 1 + [0.25] * 2 + [0.125] * 4 + [0.0625] * 8 + [0.03125] * 16  # image size fraction
        for s in scales:
            mask_h = random.randint(1, int(h * s))
            mask_w = random.randint(1, int(w * s))

            xmin = max(0, random.randint(0, w) - mask_w // 2)
            ymin = max(0, random.randint(0, h) - mask_h // 2)

            xmax = min(w, xmin + mask_w)
            ymax = min(h, ymin + mask_h)

            #* 用随机颜色填充蒙版
            im[ymin:ymax, xmin:xmax] = [random.randint(64, 191) for _ in range(3)]

            if len(labels) and s > 0.03:
                box = np.array([xmin, ymin, xmax, ymax], dtype=np.float32)
                ioa = bbox_ioa(box, xywhn2xyxy(labels[:, 1:5], w, h))
                labels = labels[ioa < 0.60]  # remove >60% obscured labels

    return im, labels

def classify_transforms(size=224):
    return T.Compose([
        CenterCrop(size),
        ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])

def classify_albumentations(
    augment=True,
    size=224,
    scale=(0.08, 1.0),
    ratio=(0.75, 1.0 / 0.75),  # 0.75, 1.33
    hflip=0.5,
    vflip=0.0,
    jitter=0.4,
    mean=IMAGENET_MEAN,
    std=IMAGENET_STD,
):
    if augment:
        T = [A.RandomResizedCrop(size, size, scale=scale, ratio=ratio)]
        if hflip > 0:
            T += [A.HorizontalFlip(p=hflip)]
        if vflip > 0:
            T += [A.VerticalFlip(p=vflip)]
        if jitter > 0:
            color_jitter = (float(jitter),) * 3
            #* brightness, contrast, saturation, 0 hue
            T += [A.ColorJitter(*color_jitter, 0)]
    else:
        T = [A.SmallestMaxSize(max_size=size), A.CenterCrop(height=size, width=size)]
    T += [A.Normalize(mean=mean, std=std), A.pytorch.ToTensorV2()]
    return A.Compose(T)

class CenterCrop:
    def __init__(self, size=640):
        """Initializes CenterCrop for image preprocessing, accepting single int or tuple for size, defaults to 640."""
        super().__init__()
        self.h, self.w = (size, size) if isinstance(size, int) else size

    def __call__(self, im):
        imh, imw = im.shape[:2]
        m = min(imh, imw)  # min dimension
        top, left = (imh - m) // 2, (imw - m) // 2
        return cv2.resize(im[top : top + m, left : left + m], (self.w, self.h), interpolation=cv2.INTER_LINEAR)


class ToTensor:

    def __init__(self, half=False):
        super().__init__()
        self.half = half

    def __call__(self, im):
        im = np.ascontiguousarray(im.transpose((2, 0, 1))[::-1])  # HWC to CHW -> BGR to RGB -> contiguous
        im = torch.from_numpy(im)  # to torch
        im = im.half() if self.half else im.float()  # uint8 to fp16/32
        im /= 255.0  # 0-255 to 0.0-1.0
        return im
