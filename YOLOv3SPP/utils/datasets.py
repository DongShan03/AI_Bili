from torch.utils.data import Dataset
import torch
import os, tqdm
import numpy as np
import cv2
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
            image_dir = [x for x in image_dir if x.split(".")[-1] in image_format]
            file_names = os.listdir(image_dir).sort()
            self.img_files = [os.path.join(image_dir, x) for x in file_names]
            self.label_files = [os.path.join(label_dir, x.replace(os.path.splitext(x)[-1], ".txt")) for x in file_names]
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
            shapes = [[1 * 1]] * nb
            for i in range(nb):
                ari = ar[bi == i]
                mini, maxi = ari.min(), ari.max()
                if maxi < 1:
                    shapes[i] = [maxi, 1]
                elif mini > 1:
                    shapes[i] = [1, 1 / mini]
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
                pbar.desc = "Caching labels (&g found, %g missing, %g empty, %g duplicate, for %g images)"\
                    % (nf, nm, ne, nd, g)
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
    pass
