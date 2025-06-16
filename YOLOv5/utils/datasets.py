import glob
import math
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import random
import shutil
import time
from pathlib import Path
from threading import Thread

import cv2
import numpy as np
import torch
from PIL import Image, ExifTags
from torch.utils.data import Dataset
from tqdm import tqdm
from YOLOv5.utils.utils import xyxy2xywh, xywh2xyxy

img_formats = ['.bmp', '.jpg', '.jpeg', '.png', '.tif', '.dng']
vid_formats = ['.mov', '.avi', '.mp4']

# Get orientation exif tag
for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == 'Orientation':
        break

def exif_size(img):
    # Returns exif-corrected PIL size
    s = img.size  # (width, height)
    try:
        rotation = dict(img._getexif().items())[orientation]
        if rotation == 6:  # rotation 270
            s = (s[1], s[0])
        elif rotation == 8:  # rotation 90
            s = (s[1], s[0])
    except:
        pass

    return s

class LoadImagesAndLabels(Dataset):  # for training/testing
    def __init__(self, path, img_size=416, batch_size=16, augment=False, hyp=None, rect=False, image_weights=False,
                cache_images=False, single_cls=False, pad=0.0):
        path = str(Path(path)) + os.sep + "images"
        try:
            # parent = str(Path(path).parent) + os.sep
            if os.path.isfile(path):
                with open(path, 'r') as f:
                    self.img_files = [x.replace('/', os.sep) for x in f.read().splitlines() if len(x)]
            elif os.path.isdir(path):
                f = glob.glob(path + os.sep + '*.*')
            else:
                raise Exception('%s does not exist' % path)
            self.img_files = [x.replace("/", os.sep) for x in f if os.path.splitext(x)[-1].lower() in img_formats]
        except:
            raise Exception('Error loading data from %s' % (path))

        n = len(self.img_files)
        assert n > 0, 'No images found in %s' % (path)

        bi = np.floor(np.arange(n) / batch_size).astype(np.int32)
        nb = bi[-1] + 1

        self.n = n
        self.batch = bi
        self.img_size = img_size
        self.augment = augment
        self.hyp = hyp
        self.image_weights = image_weights
        self.rect = False if image_weights else rect
        self.mosaic = self.augment and not self.rect

        self.label_files = [x.replace("images", "labels").replace(os.path.splitext(x)[-1], ".txt") for x in self.img_files]
        if self.rect:
            if os.path.isdir(path):
                sp = path + ".shapes"
            else:
                sp = path.replace(".txt", "") + ".shapes"
            try:
                with open(sp, 'r') as f:
                    s = [x.split() for x in f.read().splitlines()]
                    assert len(s) == n, "Shapefile out of sync"
            except:
                s = [exif_size(Image.open(f)) for f in tqdm(self.img_files, desc="Reading image shapes")]
                np.savetxt(sp, s, fmt='%g')  # save for next time

            s = np.array(s, dtype=np.float64)
            ar = s[:, 1] / s[:, 0]
            irect = ar.argsort()
            self.img_files = [self.img_files[i] for i in irect]
            self.label_files = [self.label_files[i] for i in irect]
            self.shapes = s[irect]
            ar = ar[irect]

            shapes = [[1, 1]] * nb
            for i in range(nb):
                ari = ar[bi == i]
                mini, maxi = ari.min(), ari.max()
                if maxi < 1:
                    shapes[i] = [maxi, 1]
                elif mini > 1:
                    shapes[i] = [1, 1 / mini]

            self.batch_shapes = np.ceil(np.array(shapes) * img_size / 32. + pad).astype(np.int32) * 32

        self.imgs = [None] * n
        self.labels = [torch.zeros((0, 5), dtype=torch.float32)] * n
        nm, nf, ne, ns, nd = 0, 0, 0, 0, 0
        labels_path = str(Path(self.label_files[0]).parent) + ".npy"
        labels_loaded = False
        if os.path.isfile(labels_path):
            s = labels_path
            x = torch.load(labels_path, weights_only=False)
            if len(x) == n:
                self.labels = x
                labels_loaded = True
        else:
            s = path.replace("images", "labels")
        pbar = tqdm(self.label_files, desc="Scanning images", total=len(self.label_files))
        for i, file in enumerate(pbar):
            if labels_loaded:
                l = self.labels[i]
            else:
                try:
                    with open(file, 'r') as f:
                        l = np.array([x.split() for x in f.read().splitlines()], dtype=np.float32)
                except:
                    nm += 1
                    continue

            if l.shape[0]:
                assert l.shape[1] == 5, "> 5 label columns: %s" % file
                assert (l >= 0).all(), "negative labels: %s" % file
                assert (l[:, 1:] <= 1).all(), "non-normalized or out of bounds coordinate labels: %s" % file
                if np.unique(l, axis=0).shape[0] < l.shape[0]:
                    nd += 1
                if single_cls:
                    l[:, 0] = 0
                self.labels[i] = l
                nf += 1

            else:
                ne += 1
            pbar.desc = 'Caching labels %s (%g found, %g missing, %g empty, %g duplicate, for %g images)' % (os.sep.join(s.split(os.sep)[-2:]), nf, nm, ne, nd, n)

        assert nf > 0, 'No labels found in %s' % s

        if not labels_loaded:
            print("Saving labels to %s for faster loading next time" % os.sep.join(labels_path.split(os.sep)[-2:]))
            torch.save(self.labels, labels_path)

        if cache_images:
            gb = 0
            pbar = tqdm(range(len(self.img_files)), desc="Caching images")
            self.img_hw0, self.img_hw = [None] * n, [None] * n
            for i in pbar:
                self.imgs[i], self.img_hw0[i], self.img_hw[i] = load_image(self, i)
                gb += self.imgs[i].nbytes
                pbar.desc = 'Caching images (%.1fGB)' % (gb / 1E9)

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        if self.image_weights:
            index = self.indices[index]
        hyp = self.hyp
        if self.mosaic:
            img, labels = load_mosaic(self, index)
            shapes = None
        else:
            img, (h0, w0), (h, w) = load_image(self, index)
            shape = self.batch_shapes[self.batch[index]] if self.rect else self.img_size  # final letterboxed shape
            img, ratio, pad = letterbox(img, shape, auto=False, scaleup=self.augment)   #* 等比缩放
            shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling

            labels = []
            x = self.labels[index]
            if x.size > 0:
                labels = x.copy()
                labels[:, 1] = ratio[0] * w * (x[:, 1] - x[:, 3] / 2) + pad[0]
                labels[:, 2] = ratio[1] * h * (x[:, 2] - x[:, 4] / 2) + pad[1]
                labels[:, 3] = ratio[0] * w * (x[:, 1] + x[:, 3] / 2) + pad[0]
                labels[:, 4] = ratio[1] * h * (x[:, 2] + x[:, 4] / 2) + pad[1]

        if self.augment:
            if not self.mosaic:
                img, labels = random_affine(img, labels, degrees=hyp['degrees'],
                                            translate=hyp['translate'],
                                            scale=hyp['scale'],
                                            shear=hyp['shear'])
            augment_hsv(img, h_gain=hyp['hsv_h'], s_gain=hyp['hsv_s'], v_gain=hyp['hsv_v'])

            # Apply cutouts
            if random.random() < 0.4:
                labels = cutout(img, labels)

        nL = len(labels)
        if nL:
            labels[:, 1:5] = xyxy2xywh(labels[:, 1:5])  # convert xyxy to xywh
            labels[:, [2, 4]] /= img.shape[0]  # normalized height 0-1
            labels[:, [1, 3]] /= img.shape[1]  # normalized width 0-1

        if self.augment:
            lr_flip = True
            if lr_flip and random.random() < 0.5:
                img = np.fliplr(img)
                if nL:
                    labels[:, 1] = 1 - labels[:, 1]
            ud_flip = False
            if ud_flip and random.random() < 0.5:
                img = np.flipud(img)
                if nL:
                    labels[:, 2] = 1 - labels[:, 2]

        labels_out = torch.zeros((nL, 6))
        if nL:
            labels_out[:, 1:] = torch.from_numpy(labels)

        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        return torch.from_numpy(img), labels_out, self.img_files[index], shapes, index

    def coco_index(self, index):
        #* 该方法是专门为cocotools统计标签信息准备，不对图像和标签作任何处理
        o_shapes = self.shapes[index][::-1] #* wh to hw

        x = self.labels[index]
        labels = x.copy()
        return torch.from_numpy(labels), o_shapes

    @staticmethod
    def collate_fn(batch):
        img, label, patch, shapes, index = zip(*batch)
        for i, l in enumerate(label):
            l[:, 0] = i
        return torch.stack(img, 0), torch.cat(label, 0), patch, shapes, index

def load_image(self, index):
    img = self.imgs[index]
    if img is None:
        path = self.img_files[index]
        img = cv2.imread(path)  # BGR
        assert img is not None, "Image Not Found " + path
        h0, w0 = img.shape[:2]  # orig hw
        r = self.img_size / max(h0, w0)  # resize image to img_size
        if r != 1.0:
            interp = cv2.INTER_AREA if r < 1 and not self.augment else cv2.INTER_LINEAR
            img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=interp)
        return img, (h0, w0), img.shape[:2]
    else:
        return self.imgs[index], self.img_hw0[index], self.img_hw[index]

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

def letterbox(img, new_shape=(416, 416), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    shape = img.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)

    ratio = r, r
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    if auto:
        dw, dh = np.mod(dw, 64), np.mod(dh, 64)
    elif scaleFill:
        dw, dh = 0, 0
        new_unpad = new_shape
        ratio = new_shape[0] / shape[1], new_shape[1] / shape[0]

    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, ratio, (dw, dh)

def load_mosaic(self, index):
    labels4 = []
    s = self.img_size
    xc, yc = [int(random.uniform(s * 0.5, s * 1.5)) for _ in range(2)]  # mosaic center x, y
    indices = [index] + [random.randint(0, len(self.labels) - 1) for _ in range(3)]  # 3 additional image indices
    for i, index in enumerate(indices):
        img, _, (h, w) = load_image(self, index)

        if i == 0:  # top left
            img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
            x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
        elif i == 1:  # top right
            x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
        elif i == 2:  # bottom left
            x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, max(xc, w), min(y2a - y1a, h)
        elif i == 3:  # bottom right
            x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

        img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
        padw = x1a - x1b
        padh = y1a - y1b

        x = self.labels[index]
        labels = x.copy()
        if x.size > 0:
            labels[:, 1] = w * (x[:, 1] - x[:, 3] / 2) + padw
            labels[:, 2] = h * (x[:, 2] - x[:, 4] / 2) + padh
            labels[:, 3] = w * (x[:, 1] + x[:, 3] / 2) + padw
            labels[:, 4] = h * (x[:, 2] + x[:, 4] / 2) + padh
        labels4.append(labels)

    if len(labels4):
        labels4 = np.concatenate(labels4, 0)
        np.clip(labels4[:, 1:], 0, 2 * s, out=labels4[:, 1:])  # use with random_affine

    img4, labels4 = random_affine(img4, labels4, degrees=self.hyp['degrees'], translate=self.hyp['translate'],
                                scales=self.hyp['scale'], shear=self.hyp['shear'], border=-s//2)
    return img4, labels4

def random_affine(img, targets=(), degrees=10, translate=.1, scale=.1, shear=10, border=0):

    height = img.shape[0] + border * 2
    width = img.shape[1] + border * 2

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    s = random.uniform(1 - scale, 1 + scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(img.shape[1] / 2, img.shape[0] / 2), scale=s)

    # Translation
    T = np.eye(3)
    T[0, 2] = random.uniform(-translate, translate) * img.shape[0] + border  # x translation (pixels)
    T[1, 2] = random.uniform(-translate, translate) * img.shape[1] + border  # y translation (pixels)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Combined rotation matrix
    M = S @ T @ R  # ORDER IS IMPORTANT HERE!!
    if (border != 0) or (M != np.eye(3)).any():  # image changed
        img = cv2.warpAffine(img, M[:2], dsize=(width, height), flags=cv2.INTER_LINEAR, borderValue=(114, 114, 114))

    # Transform label coordinates
    n = len(targets)
    if n:
        # warp points
        xy = np.ones((n * 4, 3))
        xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
        xy = (xy @ M.T)[:, :2].reshape(n, 8)

        # create new boxes
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

        # reject warped points outside of image
        xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, width)
        xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, height)
        w = xy[:, 2] - xy[:, 0]
        h = xy[:, 3] - xy[:, 1]
        area = w * h
        area0 = (targets[:, 3] - targets[:, 1]) * (targets[:, 4] - targets[:, 2])
        ar = np.maximum(w / (h + 1e-16), h / (w + 1e-16))  # aspect ratio
        i = (w > 4) & (h > 4) & (area / (area0 * s + 1e-16) > 0.2) & (ar < 10)

        targets = targets[i]
        targets[:, 1:5] = xy[i]

    return img, targets

def cutout(img, labels):
    h, w = img.shape[:2]

    def bbox_ioa(box1, box2):
        box2 = box2.transpose()
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]

        inter_area = (np.minimum(b1_x2, b2_x2) - np.maximum(b1_x1, b2_x1)).clip(0) * \
                    (np.minimum(b1_y2, b2_y2) - np.maximum(b1_y1, b2_y1)).clip(0)

        box2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1) + 1e-16

        return inter_area / box2_area  # i/o area

    scales = [0.5] * 1 + [0.25] * 2 + [0.125] * 4 + [0.0625] * 8 + [0.03125] * 16  # image size fraction
    for s in scales:
        mask_h = random.randint(1, int(h * s))
        mask_w = random.randint(1, int(w * s))

        xmin = max(0, random.randint(0, w) - mask_w // 2)
        ymin = max(0, random.randint(0, h) - mask_h // 2)
        xmax = min(w, xmin + mask_w)
        ymax = min(h, ymin + mask_h)

        img[ymin:ymax, xmin:xmax] = [random.randint(64, 191) for _ in range(3)]

        if len(labels) and s > 0.03:
            box = np.array([xmin, ymin, xmax, ymax], dtype=np.float32)
            ioa = bbox_ioa(box, labels[:, 1:5])
            labels = labels[ioa < 0.60]  # remove >60% obscured labels

    return labels
