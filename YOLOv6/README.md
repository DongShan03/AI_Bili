YOLOv6采用了RepVGG的重参数化

解耦边界框预测头（与YOLOX相似）

引入SIoU计算box损失

采用Anchor-free

采用TAL（**Task alignment learning—任务对齐学习**）

### SIoU

相较于CIoU，SIoU将真实边界框和目标边界框的中心点连线与水平（垂直）线的角度考虑进来

### TAL

先在各个特征层计算真实边界框与预测边界框的IoU与分类得分相乘得到score，进行分类检测任务对齐

对于每一个真实边界框选择出k个最大的score对应bbox（top-k）

选择bbox所使用的anchor中心落在真实边界框内的为正样本

如果一个anchor box对应了多个真实边界框，选择真实框与预测框IoU最大的对应anchor负责该真实框

### VariFocal Loss

针对正负样本有不平衡的问题和正样本中不等权的问题，来发现更多有价值的正样本
