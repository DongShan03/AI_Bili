import glob
import logging
import math
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import random
from pathlib import Path
from itertools import repeat

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ExifTags
from torch.utils.data import Dataset
from tqdm import tqdm
from multiprocessing.pool import ThreadPool
from copy import deepcopy

from YOLOv8.utils.utils import *
from YOLOv8.utils.augment import *
logger = logging.getLogger(__name__)

class BaseDataset(Dataset):
    def __init__(self, img_path, imgsz, cache=False, augment=True, hyp=None, rect=False, prefix="",
                batch_size=16, stride=32, pad=0.5, single_cls=False, classes=None, fraction=1.0, channels=3):
        super().__init__()
        self.img_path = img_path
        self.imgsz = imgsz
        self.augment = augment
        self.single_cls = single_cls
        self.fraction = fraction
        self.channels = channels
        self.prefix = prefix
        self.rect = rect
        self.batch_size = batch_size
        self.stride = stride
        self.pad = pad
        self.num_classes = classes
        self.cv2_flag = cv2.IMREAD_GRAYSCALE if channels == 1 else cv2.IMREAD_COLOR
        self.im_files = self.get_img_files(self.img_path)
        self.labels = self.get_labels()
        self.update_labels(include_class=classes)  # single_cls and include_class
        self.ni = len(self.labels)  # number of images
        if self.rect:
            assert self.batch_size is not None
            self.set_rectangle()
        else:
            self.shape = np.array([x["shape"] for x in self.labels])

        self.buffer = []
        self.max_buffer_length = min((self.ni, self.batch_size * 8, 1000)) if self.augment else 0

        self.ims, self.im_hw0, self.im_hw = [None] * self.ni, [None] * self.ni, [None] * self.ni
        self.npy_files = [Path(f).with_suffix('.npy') for f in self.im_files]
        self.cache = cache.lower() if isinstance(cache, str) else "ram" if cache is True else None
        if self.cache == "ram" and self.check_cache_ram():
            self.cache_images()
        elif self.cache == "disk" and self.check_cache_disk():
            self.cache_images()

        # Transforms
        self.transforms = self.build_transforms(hyp=hyp)

    def get_img_files(self, img_path):
        try:
            f = []
            for p in img_path if isinstance(img_path, list) else [img_path]:
                p = Path(p)
                if p.is_dir():
                    f += glob.glob(str(p / "**" / '*.*'), recursive=True)
                elif p.is_file():
                    t = t.read().strip().splitlines()
                    parent = str(p.parent) + os.sep
                    f += [x.replace("./", parent) if x.startswith("./") else x for x in t]
                else:
                    raise FileNotFoundError(f"{self.prefix}{p} does not exist")
            im_files = sorted(x.replace("/", os.sep) for x in f if x.rpartition(".")[-1].lower() in img_formats)
            assert im_files, f"{self.prefix}No images found in {img_path}"
        except Exception as e:
            raise FileNotFoundError(f"{self.prefix}Error loading data from {img_path}\n") from e

        if self.fraction < 1:
            im_files = im_files[:round(len(im_files) * self.fraction)]
        return im_files

    def load_image(self, i, rect_mode=True):
        im, f, fn = self.ims[i], self.im_files[i], self.npy_files[i]
        if im is None:
            if fn.exists():
                try:
                    im = np.load(fn)
                except Exception as e:
                    print(f"{self.prefix}Removing corrupt *.npy image file {fn} due to: {e}")
                    Path(fn).unlink(missing_ok=True)
                    im = imread(f, flags=self.cv2_flag)
            else:
                im = imread(f, flags=self.cv2_flag)
            if im is None:
                raise FileNotFoundError(f'Image Not Found {f}')

            h0, w0 = im.shape[:2]
            if rect_mode:   #* 保持原比例的前提下将图片长边缩放到指定大小
                r = self.imgsz / max(h0, w0)
                if r != 1:
                    w, h = (min(math.ceil(w0 * r), self.imgsz), min(math.ceil(h0 * r), self.imgsz))
                    im = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)
            elif not (h0 == w0 == self.imgsz):
                im = cv2.resize(im, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)
            if im.ndim == 2:
                im = im[..., None]

            if self.augment:
                self.ims[i], self.im_hw0[i], self.im_hw[i] = im, (h0, w0), im.shape[:2]
                self.buffer.append(i)
                if 1 < len(self.buffer) >= self.max_buffer_length:
                    j = self.buffer.pop(0)
                    if self.cache != 'ram':
                        self.ims[j], self.im_hw0[j], self.im_hw[j] = None, None, None

            return im, (h0, w0), im.shape[:2]
        return self.ims[i], self.im_hw0[i], self.im_hw[i]

    def cache_images(self):
        b, gb = 0, 1 << 30
        fcn, storage = (self.cache_images_to_disk, "Disk") if self.cache == "disk" else (self.load_image, "RAM")
        with ThreadPool(NUM_THREADS) as pool:
            result = pool.imap(fcn, range(self.ni))
            pbar = tqdm(enumerate(result), total=self.ni)
            for i, x in pbar:
                if self.cache == "disk":
                    b += self.npy_files[i].stat().st_size
                else:
                    self.ims[i], self.im_hw0[i], self.im_hw[i] = x
                    b += self.ims[i].nbytes
                pbar.desc = f"{self.prefix}Caching images ({b / gb:.1f}GB {storage})"
            pbar.close()

    def cache_images_to_disk(self, i):
        f = self.npy_files[i]
        if not f.exists():
            np.save(f.as_posix(), imread(self.im_files[i]), allow_pickle=False)

    def set_rectangle(self):
        bi = np.floor(np.arange(self.ni) / self.batch_size).astype(np.int32)
        nb = bi[-1] + 1
        s = np.array([x.pop("shape") for x in self.labels])
        ar = s[:, 0] / s[:, 1]
        irect = ar.argsort()
        self.shape = s[irect]
        self.im_files = [self.im_files[i] for i in irect]
        self.labels = [self.labels[i] for i in irect]
        ar = ar[irect]

        shapes = [[1, 1]] * nb
        for i in range(nb):
            ari = ar[bi == i]
            mini, maxi = ari.min(), ari.max()
            if maxi < 1:
                shapes[i] = [maxi, 1]
            elif mini > 1:
                shapes[i] = [1, 1 / mini]
        self.batch_shapes = np.ceil(np.array(shapes) * self.imgsz / self.stride + self.pad).astype(np.int32) * self.stride
        self.batch = bi

    def __getitem__(self, index):
        return self.transforms(self.get_image_and_label(index)), index

    def __len__(self):
        return len(self.labels)

    def update_labels(self, include_class):
        include_class_array = np.array(include_class).reshape(1, -1)
        for i in range(len(self.labels)):
            if include_class is not None:
                cls = self.labels[i]["cls"]
                bboxes = self.labels[i]["bboxes"]
                segments = self.labels[i].get("segments", None)
                keypoints = self.labels[i].get("keypoints", None)
                j = (cls == include_class_array).any(1)
                self.labels[i]["cls"] = cls[j]
                self.labels[i]["bboxes"] = bboxes[j]
                if segments:
                    self.labels[i]["segments"] = [segments[si] for si, idx in enumerate(j) if idx]
                if keypoints is not None:
                    self.labels[i]["keypoints"] = keypoints[j]
            if self.single_cls:
                self.labels[i]["cls"][:, 0] = 0

    def update_labels_info(self, label):
        return label

    def check_cache_ram(self, safety_margin=0.5):
        b, gb = 0, 1 << 30
        n = min(self.ni, 30)
        for _ in range(n):
            im = imread(random.choice(self.im_files))
            if im is None:
                continue
            ratio = self.imgsz / max(im.shape[0], im.shape[1])
            b += im.nbytes * ratio**2
        mem_required = b * self.ni / n * (1 + safety_margin)
        mem = __import__("psutil").virtual_memory()
        if mem_required > mem.available:
            self.cache = None
            print(f"{mem_required / gb:.1f}GB RAM required to cache images\n"+
                f"with {int(safety_margin * 100)}% safety margin but only\n"+
                f"{mem.available / gb:.1f}/{mem.total / gb:.1f}GB available, not caching images")
            return False
        return True

    def check_cache_disk(self, safety_margin=0.5):
        import shutil
        b, gb = 0, 1 << 30
        n = min(self.ni, 30)
        for _ in range(n):
            im_file = random.choice(self.im_files)
            im = imread(im_file)
            if im is None:
                continue
            b += im.nbytes
            if not os.access(Path(im_file).parent, os.W_OK):
                self.cache = None
                print(f"{self.prefix}Skipping caching images to disk, directory not writeable")
                return False
        disk_required = b * self.ni / n * (1 + safety_margin)
        total, used, free = shutil.disk_usage(Path(self.im_files[0]).parent)
        if disk_required > free:
            self.cache = None
            print(f"{self.prefix}{disk_required / gb:.1f}GB RAM required to cache images\n"+
                f"with {int(safety_margin * 100)}% safety margin but only\n"+
                f"{free / gb:.1f}/{total / gb:.1f}GB disk space available, not caching images")
            return False
        return True

    def get_image_and_label(self, index):
        label = deepcopy(self.labels[index])
        label.pop("shape", None)
        label["img"], label["ori_shape"], label["resized_shape"] = self.load_image(index)
        label["ratio_pad"] = (
            label["resized_shape"][0] / label["ori_shape"][0],
            label["resized_shape"][1] / label["ori_shape"][1],
        )
        if self.rect:
            label["rect_shape"] = self.batch_shapes[self.batch[index]]
        return self.update_labels_info(label)

    def build_transforms(self, hyp=None):
        raise NotImplementedError

    def get_labels(self):
        raise NotImplementedError


class YOLODataset(BaseDataset):
    def __init__(self, *args, data={}, task="detect", **kwargs):
        #* dataset = YOLODataset(img_path="path/to/images", data={"names": {0: "person"}}, task="detect")
        self.use_segments = task == "segment"
        self.use_keypoints = task == "pose"
        self.use_obb = task == "obb"
        self.data = data
        assert not (self.use_segments and self.use_keypoints), "Can not use both segments and keypoints."
        super().__init__(*args, channels=self.data.get("channels", 3), **kwargs)

    def cache_labels(self, path):
        x = {"labels": []}
        nm, nf, ne, nc, msgs = 0, 0, 0, 0, []
        desc = f"{self.prefix}Scanning {path.stem}..."
        total = len(self.im_files)
        nkpt, ndim = self.data.get("kpt_shape", (0, 0))
        if self.use_keypoints and (nkpt <= 0 or ndim not in {2, 3}):
            raise ValueError(
                "'kpt_shape' in data.yaml missing or incorrect. Should be a list with [number of "
                "keypoints, number of dims (2 for x,y or 3 for x,y,visible)], i.e. 'kpt_shape: [17, 3]'"
            )
        with ThreadPool(NUM_THREADS) as pool:
            result = pool.imap(
                func=verify_image_label,
                iterable=zip(
                    self.im_files,
                    self.label_files,
                    repeat(self.prefix),
                    repeat(self.use_keypoints),
                    repeat(len(self.data.get("names", range(self.num_classes)))),
                    repeat(nkpt),
                    repeat(ndim),
                    repeat(self.single_cls),
                )
            )
            pbar = tqdm(result, desc=desc, total=total)
            for im_file, lb, shape, segment, keypoint, nm_f, nf_f, ne_f, nc_f, msg in pbar:
                nm += nm_f
                nf += nf_f
                ne += ne_f
                nc += nc_f
                if im_file:
                    x["labels"].append(
                        {
                            "im_file": im_file,
                            "shape": shape,
                            "cls": lb[:, 0:1],
                            "bboxes": lb[:, 1:],
                            "segments": segment,
                            "keypoints": keypoint,
                            "normalized": True,
                            "bbox_format": "xywh",
                        }
                    )
                    if msg:
                        msgs.append(msg)
                pbar.desc = f"{desc} {nf} images, {nm + ne} backgrounds, {nc} corrupt"
            pbar.close()
            if msgs:
                print("\n".join(msgs))
            if nf == 0:
                print(f"{self.prefix}WARNING: No labels found in {path}")
            x["hash"] = get_hash(self.label_files + self.im_files)
            x["result"] = nf, nm, ne, nc, len(self.im_files)
            x["msgs"] = msgs
            save_dataset_cache_file(self.prefix, path, x, DATASET_CACHE_VERSION)
            return x

    def get_labels(self):
        self.label_files = img2label_paths(self.im_files)
        cache_path = Path(self.label_files[0]).parent.with_suffix(".cache")
        try:
            cache, exists = load_dataset_cache_file(cache_path), True
            assert cache["hash"] == get_hash(self.label_files + self.im_files)
            assert cache["version"] == DATASET_CACHE_VERSION
        except (FileNotFoundError, AssertionError, AttributeError):
            cache, exists = self.cache_labels(cache_path), False  # run cache ops

        nf, nm, ne, nc, n = cache.pop("result")
        if exists:
            d = f"Scanning {cache_path.stem}... {nf} images, {nm + ne} backgrounds, {nc} corrupt"
            tqdm(None, desc=self.prefix+d, total=n, initial=n)
            if cache["msgs"]:
                logger.info("\n".join(cache["msgs"]))

        [cache.pop(k) for k in ("hash", "version", "msgs")]
        labels = cache["labels"]
        if not labels:
            raise RuntimeError(
                f"No valid images found in {cache_path}. Images with incorrectly formatted labels are ignored."
            )
        self.im_files = [lb["im_file"] for lb in labels]
        lengths = ((len(lb["cls"]), len(lb["bboxes"]), len(lb["segments"])) for lb in labels)
        len_cls, len_boxes, len_segments = (sum(x) for x in zip(*lengths))
        if len_segments and len_boxes != len_segments:
            logger.warning(
                f"Box and segment counts should be equal, but got len(segments) = {len_segments}, "
                f"len(boxes) = {len_boxes}. To resolve this only boxes will be used and all segments will be removed. "
                "To avoid this please supply either a detect or segment dataset, not a detect-segment mixed dataset."
            )
            for lb in labels:
                lb["segments"] = []
        if len_cls == 0:
            logger.warning(f"Labels are missing or empty in {cache_path}, training may not work correctly.")
        return labels

    def build_transforms(self, hyp=None):
        if self.augment:
            hyp.mosaic = hyp.mosaic if self.augment and not self.rect else 0.0
            hyp.mixup = hyp.mixup if self.augment and not self.rect else 0.0
            hyp.cutmix = hyp.cutmix if self.augment and not self.rect else 0.0
            hyp.fliplr = hyp.fliplr if self.augment and not self.rect else 0.0
            hyp.flipud = hyp.flipup if self.augment and not self.rect else 0.0
            transfomers = v8_transforms(self, self.imgsz, hyp)
        else:
            transfomers = Compose([LetterBox(new_shape=(self.imgsz, self.imgsz), scaleup=False)])
        transfomers.append(
            Format(
                bbox_format="xywh",
                normalized=True,
                return_mask=self.use_segments,
                return_keypoints=self.use_keypoints,
                return_obb=self.use_obb,
                batch_idx=True,
                mask_ratio=hyp.mask_ratio,
                mask_overlap=hyp.overlap_mask,
                bgr=hyp.bgr if self.augment else 0.0,  # only affect training.
            )
        )
        return transfomers

    def close_mosaic(self, hyp):
        hyp.mosaic = 0.0
        hyp.copy_paste = 0.0
        hyp.mixup = 0.0
        hyp.cutmix = 0.0
        hyp.fliplr = 0.0
        hyp.flipud = 0.0
        self.transforms = self.build_transforms(hyp)

    def update_labels_info(self, label):
        bboxes = label.pop('bboxes')
        segments = label.pop("segments", [])
        keypoints = label.pop("keypoints", None)
        bbox_format = label.pop("bbox_format")
        normalized = label.pop("normalized")

        segment_resamples = 100 if self.use_obb else 1000
        if len(segments) > 0:
            max_len = max(len(s) for s in segments)
            segment_resamples = (max_len + 1) if segment_resamples < max_len else segment_resamples
            segments = np.stack(resample_segments(segments, n=segment_resamples), axis=0)
        else:
            segments = np.zeros((0, segment_resamples, 2), dtype=np.float32)
        label["instances"] = Instances(bboxes, segments, keypoints, bbox_format=bbox_format, normalized=normalized)
        return label

    @staticmethod
    def collate_fn(batch):
        old_batch, index = zip(*batch)
        new_batch = {}
        batch = [dict(sorted(b.items())) for b in old_batch]
        keys = batch[0].keys()
        values = list(zip(*[list(b.values()) for b in batch]))
        for i, k in enumerate(keys):
            value = values[i]
            if k in {"img", "text_feats"}:
                value = torch.stack(value, 0)
            elif k == "visuals":
                value = nn.utils.rnn.pad_sequence(value, batch_first=True)
            if k in {"masks", "keypoints", "bboxes", "cls", "segments", "obb"}:
                value = torch.cat(value, 0)
            new_batch[k] = value
        new_batch["batch_idx"] = list(new_batch["batch_idx"])
        for i in range(len(new_batch["batch_idx"])):
            new_batch["batch_idx"][i] += i
        new_batch["batch_idx"] = torch.cat(new_batch["batch_idx"], 0)
        return new_batch, index

    def coco_index(self, index):
        #* 该方法是专门为cocotools统计标签信息准备，不对图像和标签作任何处理
        o_shapes = self.shape[index][::-1] #* wh to hw

        x1 = self.labels[index]["cls"]
        x2 = self.labels[index]["bboxes"]
        labels = np.concatenate((x1.copy(), x2.copy()), axis=-1)
        return torch.from_numpy(labels), o_shapes
