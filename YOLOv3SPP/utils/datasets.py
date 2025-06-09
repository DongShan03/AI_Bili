import os, sys
sys.path.append(os.path.dirname(__file__))
from torch.utils.data import Dataset
import torch, random, math, shutil
from tqdm import tqdm
import numpy as np
import cv2
from .utils import xyxy2xywh
from PIL import Image, ExifTags

for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == "Orientation":
        break
image_format = ['.bmp', '.jpg', '.jpeg', '.png', '.tif', '.dng']

def exif_size(img):
    s = img.size
    try:
        rotation = dict(img._getexif().items())[orientation]
        if rotation == 6:
            s = (s[1], s[0])
        elif rotation == 8:
            s = (s[1], s[0])
    except:
        pass
    return s

class LoadImagesAndLabels(Dataset):
    def __init__(self, path,    #* 给到的值应当为yolo/train or yolo/val
                img_size=416,
                batch_size=16,
                augment=False,
                hyp=None,
                rect=False,     # 是否使用rectangular training
                cache_images=False,
                single_cls=False,
                pad=0.0, rank=-1):
        try:
            image_dir = os.path.join(path, "images")
            label_dir = os.path.join(path, "labels")
            file_names = sorted(os.listdir(image_dir))
            self.img_files = [os.path.join(image_dir, x) for x in file_names]
            self.label_files = [os.path.join(label_dir, x.replace(os.path.splitext(x)[-1], ".txt")) \
                                for x in file_names]
        except Exception as e:
            raise FileNotFoundError("Error loading data from {}. {}".format(path, e))
        n = len(self.img_files)
        assert n > 0, "No images found in %s." % (path)

        bi = np.floor(np.arange(n) / batch_size).astype(np.int)
        #* nb -> batch的数量
        nb = bi[-1] + 1

        self.n = n
        #* 记录哪些图片属于哪个batch
        self.batch = bi
        self.img_size = img_size
        self.augment = augment
        self.hyp = hyp
        self.rect = rect
        #! 注意: 开启rect后，mosaic就默认关闭
        self.mosaic = self.augment and not self.rect

        # 查看data文件下是否缓存有对应数据集的.shapes文件，里面存储了每张图像的width, heigh
        sp = os.path.join(path, "train.shape")
        try:
            with open(sp, "r") as f:  # read existing shapefile
                s = [x.split() for x in f.read().splitlines()]
                # 判断现有的shape文件中的行数(图像个数)是否与当前数据集中图像个数相等
                # 如果不相等则认为是不同的数据集，故重新生成shape文件
                assert len(s) == n, "shapefile out of aync"
        except Exception as e:
            if rank in [-1, 0]:
                image_files = tqdm(self.img_files, desc="Reading image shapes")
            else:
                image_files = self.img_files
            s = [exif_size(Image.open(f)) for f in image_files]
            np.savetxt(sp, s, fmt="%g")

        # 记录每张图像的原始尺寸
        self.shapes = np.array(s, dtype=np.float64)
        # 如果为ture，训练网络时，会使用等比缩放 使得最大边长为img_size，并且默认关闭mosaic
        if self.rect:
            s = self.shapes
            ar = s[:, 1] / s[:, 0]  # aspect ratio
            # argsort函数返回的是数组值从小到大的索引值
            # 按照高宽比例进行排序，这样后面划分的每个batch中的图像就拥有类似的高宽比
            irect = ar.argsort()
            self.img_files = [self.img_files[i] for i in irect]
            self.label_files = [self.label_files[i] for i in irect]
            self.shapes = s[irect]  # wh
            ar = ar[irect]

            #* 计算每个batch采用的统一尺度
            shapes = [[1, 1]] * nb
            for i in range(nb):
                ari = ar[bi == i]
                mini, maxi = ari.min(), ari.max()
                if maxi < 1:
                    shapes[i] = [maxi, 1]
                elif mini > 1:
                    shapes[i] = [1, 1 / mini]

            #* 计算每个batch输入网络的shape值(向上设置为32的整数倍)
            self.batch_shapes = np.ceil(np.array(shapes) * img_size / 32. + pad).astype(np.int) * 32

        self.imgs = [None] * n
        #! labels -> [classes, x, y, w, h]
        self.labels = [np.zeros((0, 5), dtype=np.float32)] * n
        nm, nf, ne, nd = 0, 0, 0, 0  # number missing, found, empty, duplicates

        if rect is True:
            np_labels_path = os.path.join(path, "labels.rect.npy")
        else:
            np_labels_path = os.path.join(path, "labels.norect.npy")

        labels_loaded = False
        if os.path.exists(np_labels_path):
            x = np.load(np_labels_path, allow_pickle=True)
            if len(x) == n:
                self.labels = x
                labels_loaded = True

        if rank in [-1, 0]:
            pbar = tqdm(self.label_files, desc="Loading labels")
        else:
            pbar = self.label_files

        for i, file in enumerate(pbar):
            if labels_loaded is True:
                l = self.labels[i]
            else:
                try:
                    with open(file, "r") as f:
                        l = np.array([x.split() for x in f.read().splitlines()], dtype=np.float32)
                except Exception as e:
                    print("An error occurred while loading %s: %s" % (file, e))
                    nm += 1  # missing
                    continue

            if l.shape[0]:
                assert l.shape[1] == 5, "> 5 labels per image is not supported"
                assert (l >= 0).all(), "negetive labels"
                assert (l[:, 1:] <= 1).all(), "non-normalized or out of bounds coordinate labels"

                if np.unique(l, axis=0).shape[0] < l.shape[0]:
                    nd += 1  # duplicate
                if single_cls:
                    l[:, 0] = 0  # force dataset into single-class mode
                self.labels[i] = l
                nf += 1  # file found


            else:
                ne += 1
            if rank in [-1, 0]:
                pbar.desc = "Caching labels (%g found, %g missing, %g empty, %g duplicate, for %g images)"\
                    % (nf, nm, ne, nd, n)
        assert nf > 0, "No labels found in %s. Can not train without labels." % path
        #* 如果标签信息没有被保存成numpy的格式，且训练样本数大于1000则将标签信息保存成numpy的格式
        if not labels_loaded and n > 1000:
            print("Saving labels to %s for faster future loading" % np_labels_path)
            np.save(np_labels_path, self.labels)

        if cache_images:
            gb = 0      # Gigabytes of cached images 用于记录缓存图像占用RAM大小
            if rank in [-1, 0]:
                pbar = tqdm(range(len(self.img_files)), desc="Caching images")
            else:
                pbar = range(len(self.img_files))

            self.img_hw0, self.img_hw = [None] * n, [None] * n
            for i in pbar:
                self.imgs[i], self.img_hw0[i], self.img_hw[i] = load_image(self, i)
                gb += self.imgs[i].nbytes
                if rank in [-1, 0]:
                    pbar.desc = "Caching images (%.1fGB)" % (gb / 1E9)

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        hyp = self.hyp
        if self.mosaic:
            img, labels = load_mosaic(self, index)
            shapes = None
        else:
            img, (h0, w0), (h, w) = load_image(self, index)

            shape = self.batch_shapes[self.batch[index]] if self.rect else self.img_size
            img, ratio, pad = letterbox(img, shape, auto=False, scale_up=self.augment)
            shapes = (h0, w0), ((h / h0, w / w0), pad)

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
                img, labels = random_affine(img, labels, degrees=hyp["degrees"],
                                            translate=hyp["translate"], scale=hyp["scale"],
                                            shear=hyp["shear"])
            augment_hsv(img, h_gain=hyp["hsv_h"], s_gain=hyp["hsv_s"], v_gain=hyp["hsv_v"])

        nL = len(labels)
        if nL:
            labels[:, 1:5] = xyxy2xywh(labels[:, 1:5])

            labels[:, [2, 4]] /= img.shape[0]  # normalized height (pixels)
            labels[:, [1, 3]] /= img.shape[1]  # normalized width (pixels)

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

        #* BGR2RGB  [H, W, C] -> [C, H, W]
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img, dtype=np.float32)

        return torch.from_numpy(img), labels_out, self.img_files[index], shapes, index

    def coco_index(self, index):
        #* 该方法是专门为cocotools统计标签信息准备，不对图像和标签作任何处理
        o_shapes = self.shapes[index][::-1] #* wh to hw

        x = self.labels[index]
        labels = x.copy()
        return torch.from_numpy(labels), o_shapes

    @staticmethod
    def collate_fn(batch):
        img, labels, path, shapes, index = zip(*batch)
        for i, l in enumerate(labels):
            l[:, 0] = i
        return torch.stack(img, 0), torch.cat(labels, 0), path, shapes, index


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

def letterbox(img: np.ndarray,
            new_shape=(416, 416),
            color=(114, 114, 114),
            auto=True,
            scale_fill=False,
            scale_up=True):
    #! 将图片缩放到指定大小
    shape = img.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scale_up:
        r = min(r, 1.0)

    ratio = r, r    #? 得到缩放比例
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    if auto:
        #* 这里的取余操作可以保证padding后的图片是32的整数倍
        dw, dh = np.mod(dw, 32), np.mod(dh, 32)
    elif scale_fill:     #* stretch 简单粗暴的将图片缩放到指定尺寸
        dw, dh = 0, 0
        new_unpad = new_shape
        ratio = new_shape[0] / shape[1], new_shape[1] / shape[0]

    #* pad移动到上下左右
    dw /= 2
    dh /= 2
    #* 原图尺寸不是想要得到的尺寸
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    return img, ratio, (dw, dh)


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

    img4, label4 = random_affine(img4, label4, degrees=self.hyp["degrees"],
                                translate=self.hyp["translate"],
                                scale=self.hyp["scale"],
                                shear=self.hyp["shear"],
                                border=-s//2)

    return img4, label4




def random_affine(img, targets=(), degrees=10, translate=.1, scale=.1, shear=10, border=0):
    """随机旋转，缩放，平移以及错切"""
    height = img.shape[0] + border * 2
    width = img.shape[1] + border * 2

    #* 旋转缩放
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    s = random.uniform(1 - scale, 1 + scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(img.shape[1] / 2, img.shape[0] / 2), scale=s)

    #* 平移
    T = np.eye(3)
    T[0, 2] = random.uniform(-translate, translate) * img.shape[0] + border  # x translation (pixels)
    T[1, 2] = random.uniform(-translate, translate) * img.shape[1] + border  # y translation (pixels)

    #* 错切
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 100)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 100)

    M = S @ T @ R
    if (border != 0) or (M != np.eye(3)).any():
        img = cv2.warpAffine(img, M[:2], dsize=(width, height), flags=cv2.INTER_LINEAR, borderValue=(114, 114, 114))

    # Transform label coordinates
    n = len(targets)
    if n:
        xy = np.ones((n * 4, 3))
        #* x1y1, x2y2, x1y2, x2y1
        xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)
        xy = (xy @ M.T)[:, :2].reshape(n, 8)

        #* 对transform后的bbox进行修正(假设变换后的bbox变成了菱形，此时要修正成矩形)
        x = xy[:, [0, 2, 4, 6]]  # [n, 4]
        y = xy[:, [1, 3, 5, 7]]  # [n, 4]
        xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T  # [n, 4]

        #* 对坐标进行裁剪，防止越界
        xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, width)
        xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, height)
        w = xy[:, 2] - xy[:, 0]
        h = xy[:, 3] - xy[:, 1]

        #* 计算调整后的每个box的面积
        area = w * h
        # 计算调整前的每个box的面积
        area0 = (targets[:, 3] - targets[:, 1]) * (targets[:, 4] - targets[:, 2])
        # 计算每个box的比例
        ar = np.maximum(w / (h + 1e-16), h / (w + 1e-16))  # aspect ratio
        # 选取长宽大于4个像素，且调整前后面积比例大于0.2，且比例小于10的box
        i = (w > 4) & (h > 4) & (area / (area0 * s + 1e-16) > 0.2) & (ar < 10)

        targets = targets[i]
        targets[:, 1:5] = xy[i]

    return img, targets
def create_folder(path="./new_folder"):
    # Create floder
    if os.path.exists(path):
        shutil.rmtree(path)  # dalete output folder
    os.makedirs(path)  # make new output folder
