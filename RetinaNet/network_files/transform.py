import torch, os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from torch import nn, Tensor
import math
from RetinaNet.network_files.image_list import ImageList
from torch.jit.annotations import List, Tuple, Dict, Optional

class GeneralizedRCNNTransform(nn.Module):
    def __init__(self, min_size, max_size, image_mean, image_std):
        super().__init__()
        if not isinstance(min_size, (list, tuple)):
            min_size = (min_size,)
        self.min_size = min_size
        self.max_size = max_size
        self.image_mean = image_mean
        self.image_std = image_std

    def normalize(self, image):
        dtype, device = image.dtype, image.device
        mean = torch.as_tensor(self.image_mean, dtype=dtype, device=device)
        std = torch.as_tensor(self.image_std, dtype=dtype, device=device)
        return (image - mean[:, None, None]) / std[:, None, None]

    def torch_choice(self, l):
        # type: (List[int]) -> int
        index = int(torch.empty(1).uniform_(0, float(len(l))).item())
        return l[index]

    def resize(self, image, target):
        # type: (Tensor, Optional[Dict[str, Tensor]]) -> (Tensor, Optional[Dict[str, Tensor]])
        h, w = image.shape[-2:]
        im_shape = torch.tensor(image.shape[-2:])
        min_size = float(torch.min(im_shape))
        max_size = float(torch.max(im_shape))
        if self.training:
            size = float(self.torch_choice(self.min_size))
        else:
            size = float(self.min_size[-1])
        scale_factor = size / min_size

        if max_size * scale_factor > self.max_size:
            scale_factor = self.max_size / max_size
        #! bilinear只支持4D Tensor
        image = torch.nn.functional.interpolate(
            image[None], scale_factor=scale_factor,
            mode="bilinear", align_corners=False
        )[0]

        if target is None:
            return image, target
        #! 这里有所修改需要判断
        bbox = target["boxes"]
        target["boxes"] = bbox * scale_factor
        return image, target

    def max_by_axis(self, the_list):
        # type: (List[List[int]]) -> List[int]
        maxes = the_list[0]
        for sublist in the_list[1:]:
            for index, item in enumerate(sublist):
                maxes[index] = max(maxes[index], item)
        return maxes

    def batch_images(self, images, size_divisible=32):
        # type: (List[Tensor], int) -> Tensor
        max_size = self.max_by_axis([list(img.shape) for img in images])
        stride = float(size_divisible)
        max_size[1] = int(math.ceil(float(max_size[1]) / stride) * stride)
        max_size[2] = int(math.ceil(float(max_size[2]) / stride) * stride)
        #! [batch_size, channel, height, width]
        batch_shape = [len(images)] + max_size
        batched_imgs = images[0].new_full(batch_shape, 0)
        for img, pad_img in zip(images, batched_imgs):
            pad_img[:img.shape[0], :img.shape[1], :img.shape[2]].copy_(img)
        return batched_imgs

    def postprocess(self, result, image_shapes, original_image_sizes):
        # type: (List[Dict[str, Tensor]], List[Tuple[int, int]], List[Tuple[int, int]]) -> List[Dict[str, Tensor]]
        if self.training:
            return result

        # 遍历每张图片的预测信息，将boxes信息还原回原尺度
        for i, (pred, im_s, o_im_s) in enumerate(zip(result, image_shapes, original_image_sizes)):
            boxes = pred["boxes"]
            scale_factor = o_im_s[0] / im_s[0]
            boxes = boxes * scale_factor  # 将bboxes缩放回原图像尺度上
            result[i]["boxes"] = boxes
        return result

    def __repr__(self):
        """自定义输出实例化对象的信息，可通过print打印实例信息"""
        format_string = self.__class__.__name__ + '('
        _indent = '\n    '
        format_string += "{0}Normalize(mean={1}, std={2})".format(_indent, self.image_mean, self.image_std)
        format_string += "{0}Resize(min_size={1}, max_size={2}, mode='bilinear')".format(_indent, self.min_size,
                                                                                        self.max_size)
        format_string += '\n)'
        return format_string

    def forward(self, images, targets=None):
        images = [img for img in images]
        for i in range(len(images)):
            image = images[i]
            target_index = targets[i] if targets is not None else None

            if image.dim() != 3:
                raise ValueError("image should be in 3d")

            image = self.normalize(image)
            image, target_index = self.resize(image, target_index)
            images[i] = image
            if targets is not None and target_index is not None:
                targets[i] = target_index
        image_sizes = [img.shape[-2:] for img in images]
        images = self.batch_images(images)
        images_sizes_list = torch.jit.annotate(List[Tuple[int, int]], [])
        for image_size in image_sizes:
            assert len(image_size) == 2
            images_sizes_list.append((image_size[0], image_size[1]))
        image_list = ImageList(images, images_sizes_list)
        return image_list, targets
