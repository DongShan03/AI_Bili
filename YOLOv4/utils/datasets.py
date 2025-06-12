import glob
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import random
from itertools import repeat
from multiprocessing.pool import ThreadPool
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ExifTags
from torch.utils.data import Dataset
from tqdm import tqdm

from YOLOv4.utils.utils import torch_distributed_zero_first, letterbox, xyxy2xywh, xywh2xyxy
from YOLOv4.utils.image_transform import augment_hsv, random_perspective


img_formats = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng']  # acceptable image suffixes
vid_formats = ['mov', 'avi', 'mp4', 'mpg', 'mpeg', 'm4v', 'wmv', 'mkv']  # acceptable video suffixes

for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == 'Orientation':
        break

def get_hash(files):
    # Returns a single hash value of a list of files
    return sum(os.path.getsize(f) for f in files if os.path.isfile(f))

def exif_size(img):
    s = img.size
    try:
        rotation = dict(img._getexif().items())[orientation]
        if rotation == 6:  # rotation 270
            s = (s[1], s[0])
        elif rotation == 8:  # rotation 90
            s = (s[1], s[0])
    except:
        pass

    return s

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

class LoadImagesAndLabels(Dataset):  # for training/testing
    def __init__(self, path, img_size=640, batch_size=16, augment=False, hyp=None, rect=False, image_weights=False,
                cache_images=False, single_cls=False, stride=32, pad=0.0, rank=-1):
        pass
    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        pass

class LoadImages(object):
    def __init__(self, path, img_size=640, auto_size=32):
        p = str(Path(path))
        p = os.path.abspath(p)
        if '*' in p:
            files = sorted(glob.glob(p, recursive=True))
        elif os.path.isdir(p):
            files = sorted(glob.glob(os.path.join(p, '*.*')))
        elif os.path.isfile(p):
            files = [p]
        else:
            raise Exception('ERROR: %s does not exist' % p)

        images = [x for x in files if x.split('.')[-1].lower() in img_formats]
        videos = [x for x in files if x.split('.')[-1].lower() in vid_formats]
        ni, nv = len(images), len(videos)

        self.img_size = img_size
        self.auto_size = auto_size
        self.files = images + videos
        self.nf = ni + nv
        self.video_flag = [False] * ni + [True] * nv
        self.mode = 'images'
        if any(videos):
            self.new_video(videos[0])
        else:
            self.cap = None
        assert self.nf > 0, 'No images or videos found in %s. Supported formats are:\nimages: %s\nvideos: %s' % \
                            (p, img_formats, vid_formats)

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.nf:
            raise StopIteration
        path = self.files[self.count]

        if self.video_flag[self.count]:
            self.mode = 'video'
            ret_val, img0 = self.cap.read()
            if not ret_val:
                self.count += 1
                self.cap.release()
                if self.count == self.nf:  # last video
                    raise StopIteration
                else:
                    path = self.files[self.count]
                    self.new_video(path)
                    ret_val, img0 = self.cap.read()
            self.frame += 1
            print('video %g/%g (%g/%g) %s: ' % (self.count + 1, self.nf, self.frame, self.nframes, path), end='')

        else:
            self.count += 1
            img0 = cv2.imread(path)  # BGR
            assert img0 is not None, 'Image Not Found ' + path
            print('image %g/%g %s: ' % (self.count, self.nf, path), end='')

        img = letterbox(img0, new_shape=self.img_size, auto_size=self.auto_size)[0]

        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        return path, img, img0, self.cap

    def new_video(self, path):
        self.frame = 0
        self.cap = cv2.VideoCapture(path)
        self.nframes = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __len__(self):
        return self.nf

class LoadImagesAndLabels(Dataset):
    def __init__(self, path, img_size=640, batch_size=16, augment=False, hyp=None, rect=False, image_weights=False,
                cache_images=False, single_cls=False, stride=32, pad=0.0, rank=-1):
        self.img_size = img_size
        self.augment = augment
        self.hyp = hyp
        self.image_weights = image_weights
        self.rect = False if image_weights else rect
        self.mosaic = self.augment and not self.rect
        self.mosaic_border = [-img_size // 2, -img_size // 2]
        self.stride = stride

        def img2label_paths(img_paths):
            sa, sb = os.sep + 'images' + os.sep, os.sep + 'labels' + os.sep
            return [x.replace(sa, sb, 1).replace(x.split(".")[-1], "txt") for x in img_paths]

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
            self.img_files = sorted([x.replace('/', os.sep) for x in f if x.split('.')[-1].lower() in img_formats])
            assert self.img_files, 'No images found'
        except Exception as e:
            raise Exception('Error loading data from %s. %s' % (path, e))

        self.label_files = img2label_paths(self.img_files)
        cache_path = str(Path(self.label_files[0]).parent) + '.cache3'
        if os.path.isfile(cache_path):
            cache = torch.load(cache_path, weights_only=False)
            if cache["hash"] != get_hash(self.label_files + self.img_files):
                cache = self.cache_labels(cache_path)
        else:
            cache = self.cache_labels(cache_path)

        cache.pop("hash")
        labels, shapes = zip(*cache.values())
        self.labels = list(labels)
        self.shapes = np.array(shapes, dtype=np.float64)
        self.img_files = list(cache.keys())  # update
        self.label_files = img2label_paths(cache.keys())  # update

        n = len(shapes)
        bi = np.floor(np.arange(n) / batch_size).astype(np.int32)
        nb = bi[-1] + 1
        self.batch = bi
        self.n = n

        if self.rect:
            s = self.shapes     #* wh
            ar = s[:, 1] / s[:, 0]  #* aspect ratio
            irect = ar.argsort()
            self.img_files = [self.img_files[i] for i in irect]
            self.label_files = [self.label_files[i] for i in irect]
            self.labels = [self.labels[i] for i in irect]
            self.shapes = s[irect]  # wh
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

            nm, nf, ne, ns, nd = 0, 0, 0, 0, 0
            pbar = enumerate(self.label_files)
            if rank in [-1, 0]:
                pbar = tqdm(pbar, total=len(self.label_files))
            for i, file in pbar:
                label = self.labels[i]
                if label is not None and label.shape[0]:
                    assert label.shape[1] == 5, '> 5 label columns: %s' % file
                    assert (label >= 0).all(), 'negative labels: %s' % file
                    assert (label[:, 1:] <= 1).all(), 'non-normalized or out of bounds coordinate labels: %s' % file
                    if np.unique(label, axis=0).shape[0] < label.shape[0]:
                        nd += 1 #* 文件的标签有重复
                    if single_cls:
                        label[:, 0] = 0
                    self.labels[i] = label
                    nf += 1 #* 找到文件

                else:
                    #* 空文件无标签
                    ne += 1

                if rank in [-1, 0]:
                    pbar.desc = 'Scanning labels %s (%g found, %g missing, %g empty, %g duplicate, for %g images)' % (
                        cache_path.split(os.sep)[-1], nf, nm, ne, nd, n)

            if nf == 0:
                s = 'WARNING: No labels found in %s.' % (os.path.dirname(file) + os.sep)
                print(s)
                assert not augment, '%s. Can not train without labels.' % s

            self.imgs = [None] * n
            if cache_images:
                gb = 0
                self.img_hw0, self.img_hw = [None] * n, [None] * n
                #* zip(repeat(self), range(n)) -> (self, 0), (self, 1), ...  即为x
                results = ThreadPool(8).imap(lambda x: load_image(*x), zip(repeat(self), range(n)))
                pbar = tqdm(enumerate(results), total=n)
                for i, x in pbar:
                    self.imgs[i], self.img_hw0[i], self.img_hw[i] = x
                    gb += self.imgs[i].nbytes
                    if rank in [-1, 0]:
                        pbar.desc = 'Loading images into memory %g/%g (%.1fGB)' % (i + 1, n, gb / 1E9)

    def cache_labels(self, path='labels.cache3'):
        x = {}
        pbar = tqdm(zip(self.img_files, self.label_files), desc="Scanning images", total=len(self.img_files))
        for (img, label) in pbar:
            try:
                l = []
                im = Image.open(img)
                shape = exif_size(im)
                assert (shape[0] > 9)  & (shape[1] > 9), "image size < 10 pixels!"
                if os.path.isfile(label):
                    with open(label, 'r') as f:
                        l = np.array([x.split() for x in f.read().splitlines()], dtype=np.float32)

                if len(l) == 0:
                    l = np.zeros((0, 5), dtype=np.float32)
                x[img] = [l, shape]
            except Exception as e:
                print('WARNING: Ignoring corrupted image and/or label %s: %s' % (img, e))

        x["hash"] = get_hash(self.label_files + self.img_files)
        torch.save(x, path)  # save for next time
        return x

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        if self.image_weights:
            index = self.indices[index]

        hyp = self.hyp
        mosaic = self.mosaic and random.random() < hyp['mosaic']
        if mosaic:
            img, labels = load_mosaic(self, index)
            shapes = None

            if random.random() < hyp['mixup']:
                img2, label2 = load_mosaic(self, random.randint(0, len(self.labels) - 1))
                r = np.random.beta(8.0, 8.0)
                img = (img * r + img2 * (1 - r)).astype(np.uint8)
                labels = np.concatenate((labels, label2), 0)
        else:
            img, (h0, w0), (h, w) = load_image(self, index)
            #* self.batch[index] -> index所对应的batch 然后再取出对应的batch_shape
            shape = self.batch_shapes[self.batch[index]] if self.rect else self.img_size
            img, ratio, pad = letterbox(img, shape, auto=False, scaleup=self.augment)
            shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling

            labels = []
            x = self.labels[index]
            if x.size > 0:
                labels = x.copy()
                #* 将规范化xywh 转化为 坐标下的xyxy
                labels[:, 1] = ratio[0] * w * (x[:, 1] - x[:, 3] / 2) + pad[0]
                labels[:, 2] = ratio[1] * h * (x[:, 2] - x[:, 4] / 2) + pad[1]
                labels[:, 3] = ratio[0] * w * (x[:, 1] + x[:, 3] / 2) + pad[0]
                labels[:, 4] = ratio[1] * h * (x[:, 2] + x[:, 4] / 2) + pad[1]

        if self.augment:
            if not mosaic:
                img, labels = random_perspective(img, labels,
                                                degrees=hyp['degrees'],
                                                translate=hyp['translate'],
                                                scale=hyp['scale'],
                                                shear=hyp['shear'],
                                                perspective=hyp['perspective'])
            augment_hsv(img, h_gain=hyp['hsv_h'], s_gain=hyp['hsv_s'], v_gain=hyp['hsv_v'])

        nL = len(labels)
        if nL:
            labels[:, 1:5] = xyxy2xywh(labels[:, 1:5])
            labels[:, [2, 4]] /= img.shape[0]  # normalize height
            labels[:, [1, 3]] /= img.shape[1]  # normalize width

        if self.augment:
            if random.random() < hyp["flipud"]:
                img = np.flipud(img)
                if nL:
                    labels[:, 2] = 1 - labels[:, 2]

            if random.random() < hyp["fliplr"]:
                img = np.fliplr(img)
                if nL:
                    labels[:, 1] = 1 - labels[:, 1]

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
        img, label, path, shapes, index = zip(*batch)  # transposed
        for i, l in enumerate(label):   #* label第二维度的0号位置用于存放该label源于这一批的第几张图片
            l[:, 0] = i
        return torch.stack(img, 0), torch.cat(label, 0), path, shapes, index

class LoadImagesAndLabels9(Dataset):  # for training/testing
    def __init__(self, path, img_size=640, batch_size=16, augment=False, hyp=None, rect=False, image_weights=False,
                 cache_images=False, single_cls=False, stride=32, pad=0.0, rank=-1):
        self.img_size = img_size
        self.augment = augment
        self.hyp = hyp
        self.image_weights = image_weights
        self.rect = False if image_weights else rect
        self.mosaic = self.augment and not self.rect  # load 4 images at a time into a mosaic (only during training)
        self.mosaic_border = [-img_size // 2, -img_size // 2]
        self.stride = stride

        def img2label_paths(img_paths):
            # Define label paths as a function of image paths
            sa, sb = os.sep + 'images' + os.sep, os.sep + 'labels' + os.sep  # /images/, /labels/ substrings
            return [x.replace(sa, sb, 1).replace(x.split('.')[-1], 'txt') for x in img_paths]

        try:
            f = []  # image files
            for p in path if isinstance(path, list) else [path]:
                p = Path(p)  # os-agnostic
                if p.is_dir():  # dir
                    f += glob.glob(str(p / '**' / '*.*'), recursive=True)
                elif p.is_file():  # file
                    with open(p, 'r') as t:
                        t = t.read().splitlines()
                        parent = str(p.parent) + os.sep
                        f += [x.replace('./', parent) if x.startswith('./') else x for x in t]  # local to global path
                else:
                    raise Exception('%s does not exist' % p)
            self.img_files = sorted([x.replace('/', os.sep) for x in f if x.split('.')[-1].lower() in img_formats])
            assert self.img_files, 'No images found'
        except Exception as e:
            raise Exception('Error loading data from %s: %s' % (path, e))

        # Check cache
        self.label_files = img2label_paths(self.img_files)  # labels
        cache_path = str(Path(self.label_files[0]).parent) + '.cache3'  # cached labels
        if os.path.isfile(cache_path):
            cache = torch.load(cache_path, weights_only=False)  # load
            if cache['hash'] != get_hash(self.label_files + self.img_files):  # dataset changed
                cache = self.cache_labels(cache_path)  # re-cache
        else:
            cache = self.cache_labels(cache_path)  # cache

        # Read cache
        cache.pop('hash')  # remove hash
        labels, shapes = zip(*cache.values())
        self.labels = list(labels)
        self.shapes = np.array(shapes, dtype=np.float64)
        self.img_files = list(cache.keys())  # update
        self.label_files = img2label_paths(cache.keys())  # update

        n = len(shapes)  # number of images
        bi = np.floor(np.arange(n) / batch_size).astype(np.int)  # batch index
        nb = bi[-1] + 1  # number of batches
        self.batch = bi  # batch index of image
        self.n = n

        # Rectangular Training
        if self.rect:
            # Sort by aspect ratio
            s = self.shapes  # wh
            ar = s[:, 1] / s[:, 0]  # aspect ratio
            irect = ar.argsort()
            self.img_files = [self.img_files[i] for i in irect]
            self.label_files = [self.label_files[i] for i in irect]
            self.labels = [self.labels[i] for i in irect]
            self.shapes = s[irect]  # wh
            ar = ar[irect]

            # Set training image shapes
            shapes = [[1, 1]] * nb
            for i in range(nb):
                ari = ar[bi == i]
                mini, maxi = ari.min(), ari.max()
                if maxi < 1:
                    shapes[i] = [maxi, 1]
                elif mini > 1:
                    shapes[i] = [1, 1 / mini]

            self.batch_shapes = np.ceil(np.array(shapes) * img_size / stride + pad).astype(np.int) * stride

        nm, nf, ne, ns, nd = 0, 0, 0, 0, 0  # number missing, found, empty, datasubset, duplicate
        pbar = enumerate(self.label_files)
        if rank in [-1, 0]:
            pbar = tqdm(pbar)
        for i, file in pbar:
            l = self.labels[i]  # label
            if l is not None and l.shape[0]:
                assert l.shape[1] == 5, '> 5 label columns: %s' % file
                assert (l >= 0).all(), 'negative labels: %s' % file
                assert (l[:, 1:] <= 1).all(), 'non-normalized or out of bounds coordinate labels: %s' % file
                if np.unique(l, axis=0).shape[0] < l.shape[0]:  # duplicate rows
                    nd += 1  # print('WARNING: duplicate rows in %s' % self.label_files[i])  # duplicate rows
                if single_cls:
                    l[:, 0] = 0  # force dataset into single-class mode
                self.labels[i] = l
                nf += 1  # file found

            else:
                ne += 1  # file empty
                self.labels[i] = np.array([])

            if rank in [-1, 0]:
                pbar.desc = 'Scanning images %g/%g' % (i + 1, len(self.img_files))

        if nf == 0:
            s = 'WARNING: No labels found in %s.' % (os.path.dirname(file) + os.sep)
            print(s)
            assert not augment, '%s. Can not train without labels.' % s

        # Cache images into memory for faster training (WARNING: large datasets may exceed system RAM)
        self.imgs = [None] * n
        if cache_images:
            gb = 0  # Gigabytes of cached images
            self.img_hw0, self.img_hw = [None] * n, [None] * n
            results = ThreadPool(8).imap(lambda x: load_image(*x), zip(repeat(self), range(n)))  # 8 threads
            pbar = tqdm(enumerate(results), total=n)
            for i, x in pbar:
                self.imgs[i], self.img_hw0[i], self.img_hw[i] = x  # img, hw_original, hw_resized = load_image(self, i)
                gb += self.imgs[i].nbytes
                pbar.desc = 'Caching images (%.1fGB)' % (gb / 1E9)

    def cache_labels(self, path='labels.cache3'):
        # Cache dataset labels, check images and read shapes
        x = {}  # dict
        pbar = tqdm(zip(self.img_files, self.label_files), desc='Scanning images', total=len(self.img_files))
        for (img, label) in pbar:
            try:
                l = []
                im = Image.open(img)
                im.verify()  # PIL verify
                shape = exif_size(im)  # image size
                assert (shape[0] > 9) & (shape[1] > 9), 'image size <10 pixels'
                if os.path.isfile(label):
                    with open(label, 'r') as f:
                        l = np.array([x.split() for x in f.read().splitlines()], dtype=np.float32)  # labels
                if len(l) == 0:
                    l = np.zeros((0, 5), dtype=np.float32)
                x[img] = [l, shape]
            except Exception as e:
                print('WARNING: Ignoring corrupted image and/or label %s: %s' % (img, e))

        x['hash'] = get_hash(self.label_files + self.img_files)
        torch.save(x, path)  # save for next time
        return x

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        if self.image_weights:
            index = self.indices[index]

        hyp = self.hyp
        mosaic = self.mosaic and random.random() < hyp['mosaic']
        if mosaic:
            # Load mosaic
            #img, labels = load_mosaic(self, index)
            img, labels = load_mosaic9(self, index)
            shapes = None

            # MixUp https://arxiv.org/pdf/1710.09412.pdf
            if random.random() < hyp['mixup']:
                #img2, labels2 = load_mosaic(self, random.randint(0, len(self.labels) - 1))
                img2, labels2 = load_mosaic9(self, random.randint(0, len(self.labels) - 1))
                r = np.random.beta(8.0, 8.0)  # mixup ratio, alpha=beta=8.0
                img = (img * r + img2 * (1 - r)).astype(np.uint8)
                labels = np.concatenate((labels, labels2), 0)

        else:
            # Load image
            img, (h0, w0), (h, w) = load_image(self, index)

            # Letterbox
            shape = self.batch_shapes[self.batch[index]] if self.rect else self.img_size  # final letterboxed shape
            img, ratio, pad = letterbox(img, shape, auto=False, scaleup=self.augment)
            shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling

            # Load labels
            labels = []
            x = self.labels[index]
            if x.size > 0:
                # Normalized xywh to pixel xyxy format
                labels = x.copy()
                labels[:, 1] = ratio[0] * w * (x[:, 1] - x[:, 3] / 2) + pad[0]  # pad width
                labels[:, 2] = ratio[1] * h * (x[:, 2] - x[:, 4] / 2) + pad[1]  # pad height
                labels[:, 3] = ratio[0] * w * (x[:, 1] + x[:, 3] / 2) + pad[0]
                labels[:, 4] = ratio[1] * h * (x[:, 2] + x[:, 4] / 2) + pad[1]

        if self.augment:
            # Augment imagespace
            if not mosaic:
                img, labels = random_perspective(img, labels,
                                                degrees=hyp['degrees'],
                                                translate=hyp['translate'],
                                                scale=hyp['scale'],
                                                shear=hyp['shear'],
                                                perspective=hyp['perspective'])

            # Augment colorspace
            augment_hsv(img, h_gain=hyp['hsv_h'], s_gain=hyp['hsv_s'], v_gain=hyp['hsv_v'])


        nL = len(labels)  # number of labels
        if nL:
            labels[:, 1:5] = xyxy2xywh(labels[:, 1:5])  # convert xyxy to xywh
            labels[:, [2, 4]] /= img.shape[0]  # normalized height 0-1
            labels[:, [1, 3]] /= img.shape[1]  # normalized width 0-1

        if self.augment:
            # flip up-down
            if random.random() < hyp['flipud']:
                img = np.flipud(img)
                if nL:
                    labels[:, 2] = 1 - labels[:, 2]

            # flip left-right
            if random.random() < hyp['fliplr']:
                img = np.fliplr(img)
                if nL:
                    labels[:, 1] = 1 - labels[:, 1]

        labels_out = torch.zeros((nL, 6))
        if nL:
            labels_out[:, 1:] = torch.from_numpy(labels)

        # Convert
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
        img, label, path, shapes, index = zip(*batch)  # transposed
        for i, l in enumerate(label):
            l[:, 0] = i  # add target image index for build_targets()
        return torch.stack(img, 0), torch.cat(label, 0), path, shapes, index

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

def create_dataloader(path, imgsz, batch_size, stride, opt, hyp=None, augment=False, cache=False, pad=0.0, rect=False,
                    rank=-1, world_size=1, workers=8):
    with torch_distributed_zero_first(rank):
        dataset = LoadImagesAndLabels(path, imgsz, batch_size, augment=augment,
                                    hyp=hyp, rect=rect, cache_images=cache,
                                    single_cls=opt.single_cls, stride=int(stride),
                                    pad=pad, rank=rank)
    batch_size = min(batch_size, len(dataset))
    nw = min([os.cpu_count() // world_size, batch_size if batch_size > 1 else 0, workers])  # number of workers
    sampler = torch.utils.data.distributed.DistributedSampler(dataset) if rank != -1 else None
    dataloader = InfiniteDataLoader(dataset, batch_size=batch_size,
                                    num_workers=nw, sampler=sampler,
                                    pin_memory=True,
                                    collate_fn=LoadImagesAndLabels.collate_fn)

    return dataloader, dataset

def create_dataloader9(path, imgsz, batch_size, stride, opt, hyp=None, augment=False, cache=False, pad=0.0, rect=False,
                    rank=-1, world_size=1, workers=8):
    with torch_distributed_zero_first(rank):
        dataset = LoadImagesAndLabels9(path, imgsz, batch_size, augment=augment,
                                    hyp=hyp, rect=rect, cache_images=cache,
                                    single_cls=opt.single_cls, stride=int(stride),
                                    pad=pad, rank=rank)
    batch_size = min(batch_size, len(dataset))
    nw = min([os.cpu_count() // world_size, batch_size if batch_size > 1 else 0, workers])  # number of workers
    sampler = torch.utils.data.distributed.DistributedSampler(dataset) if rank != -1 else None
    dataloader = InfiniteDataLoader(dataset, batch_size=batch_size,
                                    num_workers=nw, sampler=sampler,
                                    pin_memory=True,
                                    collate_fn=LoadImagesAndLabels9.collate_fn)

    return dataloader, dataset
