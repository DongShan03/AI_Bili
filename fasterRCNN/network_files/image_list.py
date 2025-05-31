from torch import Tensor
import torch
from torch.jit.annotations import List, Tuple

class ImageList:
    def __init__(self, tensors, image_sizes):
        # type: (Tensor, List[Tuple[int, int]]) -> None
        self.tensors = tensors
        self.image_sizes = image_sizes

    def to(self, device):
        # type: (torch.device) -> ImageList
        cast_tensor = self.tensors.to(device)
        return ImageList(cast_tensor, self.image_sizes)
