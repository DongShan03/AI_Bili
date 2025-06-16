import sys, os, math, warnings, random
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from YOLOv5.opt import opt
from YOLOv5.model.model import YOLOv5, Detect
from YOLOv5.utils.datasets import LoadImagesAndLabels
from YOLOv5.utils.utils import *
from YOLOv5.utils.coco_utils import get_coco_api_from_dataset
from YOLOv5.utils.train_eval_utils import train_one_epoch, evaluate

import numpy as np
import torch.optim.lr_scheduler as lr_scheduler
from torch.cuda import amp
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from tensorboardX import SummaryWriter

warnings.filterwarnings('ignore')

model = YOLOv5(opt.cfg)
model.eval()
x = torch.randn(1, 3, 900, 900)
x = x.resize_(1, 3, 1280, 1280)
y = model.forward(x)
print(y[0].shape)
