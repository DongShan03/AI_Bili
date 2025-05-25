from https://space.bilibili.com/18161609

### ReLU

nn.ReLU(inplace=True)可以理解为以增大计算量为代价减少模型占用内存？

### 卷积

out = (in - KernelSize + 2 * padding) / Stride + 1

### 感受野

F(i) = (F(i+1) - 1) * Stride + KernelSize

F(i)表示第i层的感受野，从特征层倒退

### BN层

BN层是对同一批数据同一通道计算均值与方差

BN层训练需要控制training参数，使用时batch_size尽量大

尽量放在Conv和激活层（如ReLU）之间，同时Conv层不需要bias

### nn.Identity

原封不动的输出

### tensorboard

运行命令：tensorboard --logdir  dir_log_path
