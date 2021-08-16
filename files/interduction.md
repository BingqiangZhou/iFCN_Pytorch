
# iFCN_Pytorch相关介绍

修改[torchvision](https://pytorch.org/vision/stable/models.html#classification)给出的分类模型，实现FCN，并在FCN的基础上，实现iFCN。

- [iFCN_Pytorch相关介绍](#ifcn_pytorch相关介绍)
  - [1. FCN网络实现](#1-fcn网络实现)
    - [1.1 修改图像分类网络](#11-修改图像分类网络)
    - [1.2 结合不同步长的浅层与深层的信息](#12-结合不同步长的浅层与深层的信息)
  - [2. iFCN网络实现](#2-ifcn网络实现)
    - [2.1 修改FCN网络第一层](#21-修改fcn网络第一层)
    - [2.2 微调FCN模型参数](#22-微调fcn模型参数)
  - [3. 随机采样](#3-随机采样)
  - [4. 数据对的生成与加载](#4-数据对的生成与加载)
    - [4.1 生成数据对](#41-生成数据对)
    - [4.2 加载数据集](#42-加载数据集)
  - [5. 后处理GrabCut Optimization](#5-后处理grabcut-optimization)
  - [6. 训练](#6-训练)
  - [7. 评估NOC(Number of Click)指标](#7-评估nocnumber-of-click指标)
  - [8. 交互式分割应用Demo](#8-交互式分割应用demo)
  - [参考论文](#参考论文)

## 1. FCN网络实现

### 1.1 修改图像分类网络

根据[Fully Convolutional Networks for Semantic Segmentation](https://arxiv.org/abs/1603.04042.pdf)中`4.1. From classifier to dense FCN`的描述，丢弃最后的全局池化层，并且改全链接层为全卷积层，然后上采样（利用反卷积或者上采样层），最后再追加一个1x1的卷积层作为预测类别分数的分类层。

### 1.2 结合不同步长的浅层与深层的信息

根据[Fully Convolutional Networks for Semantic Segmentation](https://arxiv.org/abs/1603.04042.pdf)中`4.2. Combining what and where`的描述

- **FCN-32s**: 将步长等于32的特征上采样32倍，并通过一个1x1的卷积层得到预测分数
- **FCN-16s**: 将步长等于32的特征上采样2倍，并通过一个1x1的卷积层得到预测分数`S16_1`，将步长等于16的特征通过一个1x1的卷积层得到预测分数`S16_2`，将`S16_1`与`S16_2`相加得到`S16_3`，再上采样16倍，并通过一个1x1的卷积层得到预测分数
- **FCN-8s**: 将步长等于16的特征上采样2倍，并通过一个1x1的卷积层得到预测分数`S8_1`，将`S16_3`上采样2倍，并通过一个1x1的卷积层得到预测分数`S8_1`，将`S8_1`与`S8_2`相加，再上采样16倍，并通过一个1x1的卷积层得到预测分数

需要注意的是：

- 由于不是所有的分类模型中都包括论文中对应的池化层，上面的描述中将原文中的相应步长的池化层的结果改成了相应步长的特征
- 为了保证任意大小的输入，以及上采样后的大小与预测的大小相同，这里利用了一个额外的双边上采样来保证这一点
- 将全链接层转换为卷积层的过程与原FCN方式不同，全部使用卷积核为1的卷积，未对特征图尺寸进行缩小，而原FCN中使用卷积核为7的卷积，缩小了尺寸，见[源FCN代码(caffe)](https://github.com/shelhamer/fcn.berkeleyvision.org/tree/master/voc-fcn8s)，[FCN pytorch实现](https://github.com/wkentaro/pytorch-fcn/tree/master/torchfcn/models)

## 2. iFCN网络实现

### 2.1 修改FCN网络第一层

根据[Deep Interactive Object Selection](https://arxiv.org/pdf/1411.4038.pdf)中`3. The proposed algorithm`的描述，将前背景交互的距离图作为额外的通道与原图像级联输入到FCN模型中，因此需要修改FCN网络的第一层的输入层通道为5层。

### 2.2 微调FCN模型参数

根据[Deep Interactive Object Selection](https://arxiv.org/pdf/1411.4038.pdf)中`3.3. Fine tuning FCN models`的描述，作者尝试了将额外两个输入通道对应模型参数的设置为0以及设置为已有参数的平均值，但是在实际实验中，这两种参数初始化方式得到的结果没有什么差别。

需要注意的是: 由于这里的FCN基于[torchvision](https://pytorch.org/vision/stable/models.html#classification)中的图像分类网络而来，因此这里不加载FCN模型参数。

## 3. 随机采样

根据[Deep Interactive Object Selection](https://arxiv.org/pdf/1411.4038.pdf)中`3.2. Simulating user interactions`写的前背景采样策略代码见[utils/random_sampling.py](./utils/random_sampling.py)，其中`d_margin`与`d_step`的默认值论文中没有提到，这里参考了[isl-org/Intseg](https://github.com/isl-org/Intseg)库中代码[genIntSegPairs.m](https://github.com/isl-org/Intseg/blob/master/genIntSegPairs.m)的设置，分别将其设置为5，10。随机采样示例图[images/test_random_sampling.png](./images/test_random_sampling.png)

## 4. 数据对的生成与加载

### 4.1 生成数据对

使用随机采样策略，分别对VOC2012训练集、验证集（[VOC2012官方下载地址](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/)，[镜像下载地址](https://pjreddie.com/projects/pascal-voc-dataset-mirror/)）中的每幅图片中的每个对象生成N对前背景交互，前背景交互以图像的形式存储（交互打点的位置值为1，其他位置值为0）。见代码[generate_data_pairs.py](./generate_data_pairs.py)，去掉了面积小于400的对象以及对象过于细长没有采到点的对象。

- VOC2012训练集（1464张图像，3507个对象）：每个对象生成15对前背景交互，用于训练。（实际采样3057\*15\*2=91710张交互图像）
- VOC2012验证集（1449张图像，3427个对象）：每个对象生成1对前背景交互，用于训练时，进行验证。（实际采样3008*2=6016张交互图像）

### 4.2 加载数据集

数据集以以下三种形式进行加载，在实际实验中，使用第**2**种，数据加载更加全面，另外需要注意的是第3种**无需提前生成数据对**，而第1种、第2种需要通过[generate_data_pairs.py](./generate_data_pairs.py)提前生成数据对。已经生成好的数据对可在[release-voc2012 dataset with interactives](https://github.com/BingqiangZhou/iFCN_Pytorch/releases/tag/voc2012)中下载。

1. 加载VOC2012数据集中的图像，随机在15对前背景交互中加载一对，见代码[datasets/voc_with_interactives.py](./datasets/voc_with_interactives.py)（将`VOCSegmentationWithInteractive`类`main_data`参数设置为`'image'`）。（训练集一轮1464个数据对、验证集一轮1449个数据对）
2. 加载生成的前景交互，再取VOC2012数据集中对应的图像以及对应的背景交互，见代码[datasets/voc_with_interactives.py](./datasets/voc_with_interactives.py)（将`VOCSegmentationWithInteractive`类`main_data`参数设置为`'interactive'`）。（训练集一轮3507*15=52605、验证集3427个数据对）
3. 加载VOC2012数据集中的图像，随机采样生成一对前背景交互，[datasets/voc_random_sample.py](./datasets/voc_random_sample.py)（训练集一轮1464个数据对、验证集一轮1449个数据对）

## 5. 后处理GrabCut Optimization

在原论文[Deep Interactive Object Selection](https://arxiv.org/pdf/1411.4038.pdf)中为`Graph Cut Optimization`，但由于OpenCV中有[GrabCut的方法](https://docs.opencv.org/master/d8/d83/tutorial_py_grabcut.html)，直接调用比较简单，并且GrabCut相较于GraphCut有不少改进，这里直接调用[cv.grabCut](https://docs.opencv.org/master/d3/d47/group__imgproc__segmentation.html#ga909c1dda50efcbeaa3ce126be862b37f)。

将前景采样点为圆心，5为半径的圆内的点，作为确定的前景点，同样，前景采样点为圆心，5为半径的圆内的点，作为确定的背景点，将其他预测为前背景的点作为可能的前背景点，利用grabCut迭代5次，得到优化后的结果，代码见[utils/grabcut.py](./utils/grabcut.py)以及[utils/trainval.py](./utils/trainval.py)。

主要注意的时，在实验中发现，由于背景策略2是在其他目标对象区域内打点，当目标对象相似时，前背景颜色高斯混合模型相似，使得结果无法得到优化，甚至起副作用（IoU指标下降）。

GraphCut与GrabCut相关文章：

- [16. 如何通过缝隙抠出前景 - GraphCut 和 GrabCut - Wang Hawk的文章 - 知乎](https://zhuanlan.zhihu.com/p/64615890)
- [图像分割技术介绍 - SIGAI的文章 - 知乎](https://zhuanlan.zhihu.com/p/49512872)

## 6. 训练

训练时使用[二值交叉熵损失BCE](https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html?highlight=bce#torch.nn.BCEWithLogitsLoss)、[Adam优化器](https://pytorch.org/docs/stable/generated/torch.optim.Adam.html?highlight=adam#torch.optim.Adam)，学习率设置为默认的`1e-3`，权重衰减参数设置为`1e-5`。

运行训练代码命令可参考[files/commands.txt](./files/commands.txt)

## 7. 评估NOC(Number of Click)指标

每一次都在最大预测错误区域中心取点（取与区域边界距离最远的点），直到达到指定的IoU（例如，85%）或者达到最大的取点数（例如，20），代码见[noc_eval.py](./noc_eval.py)，其中：

- 第一次取点为目标对象中心点
- 当取到最大的取点数还没有达到指定的IoU时，则将最大的取点数作为当前目标对象的取点数，用于后续求整个数据集上的平均NOC指标。

## 8. 交互式分割应用Demo

基于matplotlib的交互式分割应用Demo，见[demo.py](./demo.py)（本地运行，建议运行`demo.py`）与[demo.ipynb](./demo.ipynb)（服务器上无法显示窗口，建议运行`demo.ipynb`），相关操作如下：

- 鼠标操作：
  - 左键：交互前景
  - 右键：交互背景
- 键盘操作
  - "ctrl + alt + s"键：保存分割结果
  - "n"键：下一张图像
  - ”p“键：上一张图像

## 参考论文

- [Fully Convolutional Networks for Semantic Segmentation](https://arxiv.org/abs/1603.04042.pdf)
- [Deep Interactive Object Selection](https://arxiv.org/pdf/1411.4038.pdf)
