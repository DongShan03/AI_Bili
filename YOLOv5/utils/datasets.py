import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from YOLOv5.utils.image_utils import Albumentations
from YOLOv5.utils.utils import segments2boxes, xyxy2xywh, xywhn2xyxy
from YOLOv5.utils.image_utils import *
from YOLOv5.utils.distributed_utils import torch_distributed_zero_first
import contextlib
import glob
import hashlib
import json
import math
import random
import shutil
import time
from itertools import repeat
from multiprocessing.pool import Pool, ThreadPool
from pathlib import Path
from threading import Thread
import numpy as np
import psutil
import torch
import torch.nn.functional as F
import torchvision
import yaml
import cv2
from PIL import ExifTags, Image, ImageOps
from torch.utils.data import DataLoader, Dataset, dataloader, distributed
from tqdm import tqdm


IMG_FORMATS = "bmp", "dng", "jpeg", "jpg", "mpo", "png", "tif", "tiff", "webp", "pfm"  # include image suffixes
VID_FORMATS = "asf", "avi", "gif", "m4v", "mkv", "mov", "mp4", "mpeg", "mpg", "ts", "wmv"  # include video suffixes

# Get orientation exif tag
for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == "Orientation":
        break

def get_hash(paths):
    """Generates a single SHA256 hash for a list of file or directory paths by combining their sizes and paths."""
    size = sum(os.path.getsize(p) for p in paths if os.path.exists(p))  # sizes
    h = hashlib.sha256(str(size).encode())  # hash sizes
    h.update("".join(paths).encode())  # hash paths
    return h.hexdigest()  # return hash

def exif_size(img):
    """Returns corrected PIL image size (width, height) considering EXIF orientation."""
    s = img.size  # (width, height)
    with contextlib.suppress(Exception):
        rotation = dict(img._getexif().items())[orientation]
        if rotation in [6, 8]:  # rotation 270 or 90
            s = (s[1], s[0])
    return s

def exif_transpose(image):
    exif = image.getexif()
    orientation = exif.get(0x0112, 1)
    if orientation > 1:
        method = {
            2: Image.FLIP_LEFT_RIGHT,
            3: Image.ROTATE_180,
            4: Image.FLIP_TOP_BOTTOM,
            5: Image.TRANSPOSE,
            6: Image.ROTATE_270,
            7: Image.TRANSVERSE,
            8: Image.ROTATE_90,
        }.get(orientation)
        if method is not None:
            image = image.transpose(method)
            del exif[0x0112]
            image.info["exif"] = exif.tobytes()
    return image

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def img2label_paths(img_paths):
    sa, sb = f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}"  # /images/, /labels/ substrings
    return [sb.join(x.rsplit(sa, 1)).rsplit(".", 1)[0] + ".txt" for x in img_paths]

class  _RepeatSampler(object):
    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)

class InfiniteDataLoader(torch.utils.data.dataloader.DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(self, *args, **kwargs)
        object.__setattr__(self, 'batch_sampler',  _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)

class LoadImagesAndLabels(Dataset):
    cache_version = 0.6
    rand_interp_method = [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4]

    def __init__(self, path, img_size=640, batch_size=16, augment=False, hyp=None,
                rect=False, image_weights=False, cache_images=False, single_cls=False,
                stride=32, pad=0.0, min_items=0, prefix="", rank=-1, seed=0):
        self.img_size = img_size
        self.augment = augment
        self.hyp = hyp
        self.image_weights = image_weights
        self.rect = False if image_weights else rect
        self.mosaic = self.augment and not self.rect  # load 4 images at a time into a mosaic (only during training)
        self.mosaic_border = [-img_size // 2, -img_size // 2]
        self.stride = stride
        self.path = path
        self.albumentations = Albumentations() if augment else None
        try:
            f = []
            for p in path if isinstance(path, list) else [path]:
                p = Path(p)
                if p.is_dir():
                    f += glob.glob(str(p / '**' / '*.*'), recursive=True)
                elif p.is_file():
                    with open(p, 'r') as t:
                        t = t.read().splitlines()
                        parent = str(p.parent) + os.sep
                        f += [x.replace('./', parent) if x.startswith('./') else x for x in t]
                else:
                    raise Exception('%s does not exist' % p)
            self.img_files = sorted([x.replace('/', os.sep) for x in f if x.split('.')[-1].lower() in IMG_FORMATS])
            assert self.img_files, 'No images found'
        except Exception as e:
            raise Exception('Error loading data from %s. %s' % (path, e))

        self.label_files = img2label_paths(self.img_files)
        cache_path = str(Path(self.label_files[0]).parent) + '.cache5'
        if os.path.isfile(cache_path):
            cache, exists = np.load(cache_path, allow_pickle=True).item(), True  # load dict
            assert cache["version"] == self.cache_version  # matches current version
            assert cache["hash"] == get_hash(self.label_files + self.img_files)  # identical hash
        else:
            cache, exists = self.cache_labels(cache_path), False  # run cache ops

        nf, nm, ne, nc, n = cache.pop("result")
        if exists and rank in [-1, 0]:
            d = f"Scanning {cache_path}... {nf} images, {nm + ne} backgrounds, {nc} corrupt"
            tqdm(None, desc=d, total=n, initial=n)  # display cache results
        assert nf > 0 or not augment, f"No labels found in {cache_path}, can not start training."
        [cache.pop(k) for k in ("hash", "version")]
        labels, shapes, self.segments = zip(*cache.values())
        nl = len(np.concatenate(labels, 0))
        assert nl > 0 or not augment, f"All labels empty in {cache_path}, can not start training."
        self.labels = list(labels)
        self.shapes = np.array(shapes)
        self.img_files = list(cache.keys())  # update
        self.label_files = img2label_paths(cache.keys())  # update

        if min_items:
            include = np.array([len(x) > min_items for x in self.labels]).nonzero()[0].astype(int)
            self.img_files = [self.img_files[i] for i in include]
            self.label_files = [self.label_files[i] for i in include]
            self.labels = [self.labels[i] for i in include]
            self.segments = [self.segments[i] for i in include]
            self.shapes = self.shapes[include]

        n = len(self.shapes)
        bi = np.floor(np.arange(n) / batch_size).astype(np.int)  # batch index
        nb = bi[-1] + 1  # number of batches
        self.batch = bi  # batch index of image
        self.n = n
        self.indices = np.arange(n)

        #* filter labels to include only these classes (optional)
        include_class = []
        self.segments = list(self.segments)
        include_class_array = np.array(include_class).reshape(1, -1)
        for i, (label, segment) in enumerate(zip(self.labels, self.segments)):
            if include_class:
                j = (label[:, 0:1] == include_class_array).any(1)
                self.labels[i] = label[j]
                if segment:
                    self.segments[i] = [segment[idx] for idx, elem in enumerate(j) if elem]
            if single_cls:
                self.labels[i][:, 0] = 0

        # Rectangular Training
        if self.rect:
            # Sort by aspect ratio
            s = self.shapes
            ar = s[:, 1] / s[:, 0]  # aspect ratio
            irect = ar.argsort()
            self.img_files = [self.img_files[i] for i in irect]
            self.label_files = [self.label_files[i] for i in irect]
            self.labels = [self.labels[i] for i in irect]
            self.shapes = s[irect]
            self.segments = [self.segments[i] for i in irect]
            ar = ar[irect]

            shapes = [[1, 1]] * nb
            for i in range(nb):
                ari = ar[bi == i]
                mini, maxi = ari.min(), ari.max()
                if maxi < 1:
                    shapes[i] = [maxi, 1]
                elif mini > 1:
                    shapes[i] = [1, 1 / mini]

            self.batch_shapes = np.ceil(np.array(shapes) * img_size / stride + pad).astype(np.int32) * stride

        if cache_images == "ram" and not self.check_cache_ram():
            cache_images = False
        # Cache images into memory for faster training (WARNING: large datasets may need added opt. based on RAM)
        self.imgs = [None] * n
        if cache_images:
            gb = 0  # Gigabytes of cached images
            self.img_hw0, self.img_hw = [None] * n, [None] * n
            with ThreadPool(16) as pool:
                results = pool.imap(lambda i: (i, self.load_image(i)), self.indices)
                pbar = tqdm(results, total=n)
                for i, x in enumerate(pbar):
                    self.imgs[i], self.img_hw0[i], self.img_hw[i] = x
                    gb += self.imgs[i].nbytes
                    if rank in [-1, 0]:
                        pbar.desc = "Loading images into memory %g/%g (%.1fGB)" % (i + 1, n, gb / 1E9)
                pbar.close()

    def check_cache_ram(self, safety_margin=0.1, prefix=""):
        #* 检查内存是否足够放下所有图像
        b = 0
        n = min(self.n, 30)
        for _ in range(n):
            img = cv2.imread(random.choice(self.img_files))
            ratio = self.img_size / max(img.shape[0], img.shape[1])
            b += img.nbytes * ratio ** 2

        mem_required = b * self.n / n
        mem = psutil.virtual_memory()
        cache = mem_required * (1 + safety_margin) < mem.available
        return cache

    def cache_labels(self, cache_path='labels.cache5'):
        x = {}
        nm, nf, ne, nc = 0, 0, 0, 0
        desc = f"Scanning {cache_path.parent / cache_path.stem}..."
        with Pool(16) as pool:
            pbar = tqdm(
                pool.imap(verify_image_label, zip(self.img_files, self.label_files)),
                desc=desc,
                total=len(self.img_files),
            )
            for img_file, lb, shape, segment, nm_f, nf_f, ne_f, nc_f in pbar:
                nm += nm_f
                nf += nf_f
                ne += ne_f
                nc += nc_f
                if img_file:
                    x[img_file] = [lb, shape, segment]
                x[img_file] = [lb, shape, segment]
                pbar.desc = f"{desc} {nf} images, {nm + ne} backgrounds, {nc} corrupt"
        pbar.close()
        x["hash"] = get_hash(self.label_files + self.img_files)
        x["result"] = nf, nm, ne, nc, len(self.img_files)
        x["version"] = self.cache_version
        try:
            np.save(cache_path, x)
            cache_path.with_suffix(".cache.npy").rename(cache_path)  # remove .npy suffix
        except Exception as e:
            print(f'WARNING: Cache directory {cache_path.parent} is not writeable: {e}')
            exit(-1)
        return x

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        index = self.indices[index]
        hyp = self.hyp
        mosaic = self.mosaic
        if mosaic and random.random() < hyp["mosaic"]:
            img, labels = load_mosaic(self, index)
            shapes = None

            if random.random() < hyp["mixup"]:
                img2, labels2 = mixup(img, labels, *load_mosaic(self, random.choice(self.indices)))
        else:
            img, (h0, w0), (h, w) = load_image(self, index)
            shape = self.batch_shapes[self.batch[index]] if self.rect else self.img_size  # final letterboxed shape
            img, ratio, pad = letterbox(img, shape, auto=False, scaleup=self.augment)
            shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling

            labels = self.labels[index].copy()
            if labels.size:
                labels[:, 1:] = xywhn2xyxy(labels[:, 1:], ratio[0] * w, ratio[1] * h, padw=pad[0], padh=pad[1])
            if self.augment:
                img, labels = random_perspective(
                    img, labels, degrees=hyp["degrees"], translate=hyp["translate"],
                    scale=hyp["scale"], shear=hyp["shear"], perspective=hyp["perspective"]
                )
        nl = len(labels)
        if nl:
            labels[:, 1:5] = xyxy2xywhn(labels[:, 1:5], w=img.shape[1], h=img.shape[0], clip=True)

        if self.augment:
            img, labels = self.albumentations(img, labels)
            nl = len(labels)
            augment_hsv(img, h_gain=hyp["hsv_h"], s_gain=hyp["hsv_s"], v_gain=hyp["hsv_v"])

            if random.random() < hyp["flipud"]:
                img = np.flipud(img)
                if nl:
                    labels[:, 2] = 1 - labels[:, 2]

            if random.random() < hyp["fliplr"]:
                img = np.fliplr(img)
                if nl:
                    labels[:, 1] = 1 - labels[:, 1]

            img, labels = cutout(img, labels, p=0.15)
            nl = len(labels)

        labels_out = torch.zeros((nl, 6))
        if nl:
            labels_out[:, 1:] = torch.from_numpy(labels)
        img = img.transpose((2, 0, 1))[::-1]
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
        img, label, path, shapes, index = zip(*batch)
        for i, l in enumerate(label):
            l[:, 0] = i  # add target image index for build_targets()
        return torch.stack(img, 0), torch.cat(label, 0), path, shapes, index

    @staticmethod
    def collate_fn4(batch):
        img, label, path, shapes, index = zip(*batch)
        n = len(shapes) // 4
        img4, label4, path4, shapes4, index4 = [], [], path[:n], shapes[:n], index[:n]

        ho = torch.tensor([[0.0, 0, 0, 1, 0, 0]])
        wo = torch.tensor([[0.0, 0, 1, 0, 0, 0]])
        s = torch.tensor([[1, 1, 0.5, 0.5, 0.5, 0.5]])  #* scale

        for i in range(n):
            i *= 4
            if random.random() < 0.5:
                #* 百分之五十的概率只把四张图片中的第一张图片放大为原先的两倍
                img1 = F.interpolate(
                    img[i].unsqueeze(0).float(), scale_factor=2.0,
                    mode="bilinear", align_corners=False
                )[0].type(img[i].type())
                lb = label[i]
            else:
                #* 拼合四张图像 对其中的label[2:]进行偏移也就是+ho / +wo 并且由于全图变为原先的两倍 所以xywh需要*0.5
                img1 = torch.cat((torch.cat((img[i], img[i + 1]), 1), torch.cat((img[i + 2], img[i + 3]), 1)), 2)
                lb = torch.cat((label[i], label[i + 1] + ho, label[i + 2] + wo, label[i + 3] + ho + wo), 0) * s

            img4.append(img1)
            label4.append(lb)

        for i, lb in enumerate(label4):
            lb[:, 0] = i

        return torch.stack(img4, 0), torch.cat(label4, 0), path4, shapes4, index4


def verify_image_label(args):
    img_file, lb_file = args
    nm, nf, ne, nc, segments = 0, 0, 0, 0, []
    try:
        img = Image.open(img_file)
        img.verify()
        shape = exif_size(img)
        assert (shape[0] > 9) & (shape[1] > 9), 'image size <10 pixels'
        assert img.format.lower() in IMG_FORMATS, f'invalid image format {img.format}'
        if img.format.lower() in ("jpg", "jpeg"):
            with open(img_file, "rb") as f:
                f.seek(-2, 2)
                if f.read() != b"\xff\xd9":
                    ImageOps.exif_transpose(Image.open(img_file)).save(img_file, "JPEG", subsampling=0, quality=100)

        if os.path.isfile(lb_file):
            nf = 1
            with open(lb_file) as f:
                lb = [x.split() for x in f.read().splitlines() if (len(x) > 0)]
                if any(len(x) > 6 for x in lb):
                    classses = np.array([x[0] for x in lb], dtype=np.float32)
                    segments = [np.array(x[1:], dtype=np.float32).reshape(-1, 2) for x in lb]  # (cls, xy1...)
                    lb = np.concatenate((classses.reshape(-1, 1), segments2boxes(segments)), 1)
                lb = np.array(lb, dtype=np.float32)
            nl = len(lb)
            if nl:
                assert lb.shape[1] == 5, f"labels require 5 columns, {lb.shape[1]} columns detected"
                assert (lb >= 0).all(), f"negative label values {lb[lb < 0]}"
                assert (lb[:, 1:] <= 1).all(), f"non-normalized or out of bounds coordinates {lb[:, 1:][lb[:, 1:] > 1]}"
                _, i = np.unique(lb, axis=0, return_index=True)
                if len(i) < nl:
                    lb = lb[i]
                    if segments:
                        segments = [segments[x] for x in i]
            else:
                ne = 1
                lb = np.zeros((0, 5), dtype=np.float32)
            return img_file, lb, shape, segments, nm, nf, ne, nc
    except Exception as e:
        nc = 1
        return None, None, None, None, nm, nf, ne, nc

def load_mosaic(self, index):
    #* 将四张图片拼接在一张马赛克图像中
    label4 = []
    s = self.img_size

    xc ,yc = [int(random.uniform(s * 0.5, s * 1.5)) for _ in range(2)]  # mosaic center x, y
    #* 获取另外三张照片的索引
    indices = [index] + [random.randint(0, len(self.labels) - 1) for _ in range(3)]
    for i in indices:
        img, _, h, w = load_image(self, i)
        if i == 0:
            img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
            #* 第一张图片在右下角对其xc yc后其在img4中左上角的位置（可能有裁剪）
            x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
            #* 第一张照片所取到的部分在原图上的位置
            x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
        elif i == 1: #! 右上角
            x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
        elif i == 2: #! 左下角
            x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, max(xc, w), min(y2a - y1a, h)
        elif i == 3: #! 右下角
            x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
        img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
        #* 计算pad 也就是图像的左上角的相对偏移， 用于计算label
        padw = x1a - x1b
        padh = y1a - y1b

        x = self.labels[index]
        labels = x.copy()
        if x.size > 0:
            #* 将bbox的坐标转换到新的图像上 同时坐标体系由(cx, cy, w, h) -> (xmin, ymin, xmax, ymax)
            labels[:, 1] = w * (x[:, 1] - x[:, 3] / 2) + padw
            labels[:, 2] = h * (x[:, 2] - x[:, 4] / 2) + padh
            labels[:, 3] = w * (x[:, 1] + x[:, 3] / 2) + padw
            labels[:, 4] = h * (x[:, 2] + x[:, 4] / 2) + padh

        label4.append(labels)

    if len(label4):
        label4 = np.concatenate(label4, 0)
        np.clip(labels[:, 1:], 0, 2 * s, out=labels[:, 1:])

    img4, label4 = random_perspective(img4, label4, degrees=self.hyp["degrees"],
                                translate=self.hyp["translate"],
                                scale=self.hyp["scale"],
                                shear=self.hyp["shear"],
                                border=-s//2)

    return img4, label4

def load_mosaic9(self, index):
    labels9 = []
    s = self.img_size
    indices = [index] + [random.randint(0, len(self.labels) - 1) for _ in range(8)]
    for i, index in enumerate(indices):
        img, _, (h, w) = load_image(self, index)

        if i == 0:  # center
            img9 = np.full((s * 3, s * 3, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
            h0, w0 = h, w
            c = s, s, s + w, s + h  # xmin, ymin, xmax, ymax (base) coordinates
        elif i == 1:  # top
            c = s, s - h, s + w, s
        elif i == 2:  # top right
            c = s + wp, s - h, s + wp + w, s
        elif i == 3:  # right
            c = s + w0, s, s + w0 + w, s + h
        elif i == 4:  # bottom right
            c = s + w0, s + hp, s + w0 + w, s + hp + h
        elif i == 5:  # bottom
            c = s + w0 - w, s + h0, s + w0, s + h0 + h
        elif i == 6:  # bottom left
            c = s + w0 - wp - w, s + h0, s + w0 - wp, s + h0 + h
        elif i == 7:  # left
            c = s - w, s + h0 - h, s, s + h0
        elif i == 8:  # top left
            c = s - w, s + h0 - hp - h, s, s + h0 - hp

        padx, pady = c[:2]
        x1, y1, x2, y2 = [max(x, 0) for x in c]

        x = self.labels[index]
        labels = x.copy()
        if x.size > 0:
            labels[:, 1] = w * (x[:, 1] - x[:, 3] / 2) + padx
            labels[:, 2] = h * (x[:, 2] - x[:, 4] / 2) + pady
            labels[:, 3] = w * (x[:, 1] + x[:, 3] / 2) + padx
            labels[:, 4] = h * (x[:, 2] + x[:, 4] / 2) + pady
        labels9.append(labels)

        img9[y1:y2, x1:x2] = img[y1 - pady:, x1 - padx:]
        hp, wp = h, w    #* height, width previous

    yc, xc = [int(random.uniform(0, s)) for x in self.mosaic_border]  # mosaic center x, y
    #* 将原先的3s * 3s的图像裁剪成2s * 2s
    img9 = img9[yc:yc + 2 * s, xc:xc + 2 * s]

    if len(labels9):
        labels9 = np.concatenate(labels9, 0)
        labels9[:, [1, 3]] -= xc
        labels9[:, [2, 4]] -= yc

        np.clip(labels9[:, 1:], 0, 2 * s, out=labels9[:, 1:])

    img9, labels9 = random_perspective(img9, labels9,
                                    degrees=self.hyp["degrees"],
                                    translate=self.hyp["translate"],
                                    scale=self.hyp["scale"],
                                    shear=self.hyp["shear"],
                                    perspective=self.hyp["perspective"],
                                    border=self.mosaic_border)  # border to remove

    return img9, labels9

def load_image(self, index):

    img = self.imgs[index]
    if img is None:
        path = self.img_files[index]
        img = cv2.imread(path)  # BGR
        assert img is not None, "Image Not Found " + path
        h0, w0 = img.shape[:2]
        #! img_size 设置的是预处理后输出的图片尺寸
        r = self.img_size / max(h0, w0)  # resize image to img_size
        if r != 1:
            interp = cv2.INTER_AREA if r < 1 and not self.augment else cv2.INTER_LINEAR
            img = cv2.resize(img, (int(w0*r), int(h0*r)), interpolation=interp)
        return img, (h0, w0), img.shape[:2]  # img, hw_original, hw_resized
    else:
        return self.imgs[index], self.img_hw0[index], self.img_hw[index]


# Classification dataloaders -------------------------------------------------------------------------------------------
class ClassificationDataset(torchvision.datasets.ImageFolder):
    """
    YOLOv5 Classification Dataset.

    Arguments:
        root:  Dataset path
        transform:  torchvision transforms, used by default
        album_transform: Albumentations transforms, used if installed
    """

    def __init__(self, root, augment, imgsz, cache=False):
        """Initializes YOLOv5 Classification Dataset with optional caching, augmentations, and transforms for image
        classification.
        """
        super().__init__(root=root)
        self.torch_transforms = classify_transforms(imgsz)
        self.album_transforms = classify_albumentations(augment, imgsz) if augment else None
        self.cache_ram = cache is True or cache == "ram"
        self.cache_disk = cache == "disk"
        self.samples = [list(x) + [Path(x[0]).with_suffix(".npy"), None] for x in self.samples]  # file, index, npy, im

    def __getitem__(self, i):
        """Fetches and transforms an image sample by index, supporting RAM/disk caching and Augmentations."""
        f, j, fn, im = self.samples[i]  # filename, index, filename.with_suffix('.npy'), image
        if self.cache_ram and im is None:
            im = self.samples[i][3] = cv2.imread(f)
        elif self.cache_disk:
            if not fn.exists():  # load npy
                np.save(fn.as_posix(), cv2.imread(f))
            im = np.load(fn)
        else:  # read image
            im = cv2.imread(f)  # BGR
        if self.album_transforms:
            sample = self.album_transforms(image=cv2.cvtColor(im, cv2.COLOR_BGR2RGB))["image"]
        else:
            sample = self.torch_transforms(im)
        return sample, j


def create_classification_dataloader(
    path, imgsz=224, batch_size=16, augment=True, cache=False, rank=-1, workers=8, shuffle=True
):
    # Returns Dataloader object to be used with YOLOv5 Classifier
    """Creates a DataLoader for image classification, supporting caching, augmentation, and distributed training."""
    with torch_distributed_zero_first(rank):  # init dataset *.cache only once if DDP
        dataset = ClassificationDataset(root=path, imgsz=imgsz, augment=augment, cache=cache)
    batch_size = min(batch_size, len(dataset))
    nd = torch.cuda.device_count()
    nw = min([os.cpu_count() // max(nd, 1), batch_size if batch_size > 1 else 0, workers])
    sampler = None if rank == -1 else distributed.DistributedSampler(dataset, shuffle=shuffle)
    generator = torch.Generator()
    generator.manual_seed(6148914691236517205 + rank)
    return InfiniteDataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle and sampler is None,
        num_workers=nw,
        sampler=sampler,
        pin_memory=False,
        worker_init_fn=seed_worker,
        generator=generator,
    )  # or DataLoader(persistent_workers=True)
