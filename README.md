
# iFCN_Pytorch

修改[torchvision](https://pytorch.org/vision/stable/models.html#classification)给出的分类模型，实现FCN，并在FCN的基础上，实现iFCN的训练验证等过程。

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

使用随机采样策略，分别对VOC2012训练集、验证集（[VOC2012官方下载地址](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/)，[镜像下载地址](https://pjreddie.com/projects/pascal-voc-dataset-mirror/)）中的每幅图片中的每个对象生成N对前背景交互，前背景交互以图像的形式存储（交互打点的位置值为1，其他位置值为0）。见代码[generate_data_pairs.py](./generate_data_pairs.py)。

- VOC2012训练集（1464张图像，3507个对象）：每个对象生成15对前背景交互，用于训练。（共3507\*15\*2=105210张交互图像）
- VOC2012验证集（1449张图像，3427个对象）：每个对象生成1对前背景交互，用于训练时，进行验证。（共3427*2=6854张交互图像）

### 4.2 加载数据集

数据集以以下三种形式进行加载，在实际实验中，使用第**2**种，数据加载更加全面，另外需要注意的是第3种**无需提前生成数据对**，而第1种、第2种需要通过[generate_data_pairs.py](./generate_data_pairs.py)提前生成数据对。

1. 加载VOC2012数据集中的图像，随机在15对前背景交互中加载一对，见代码[datasets/voc_with_interactives.py](./datasets/voc_with_interactives.py)（将`VOCSegmentationWithInteractive`类`main_data`参数设置为`'image'`）。（训练集一轮1464个数据对、验证集一轮1449个数据对）
2. 加载生成的前景交互，再取VOC2012数据集中对应的图像以及对应的背景交互，见代码[datasets/voc_with_interactives.py](./datasets/voc_with_interactives.py)（将`VOCSegmentationWithInteractive`类`main_data`参数设置为`'interactive'`）。（训练集一轮3507*15=52605、验证集3427个数据对）
3. 加载VOC2012数据集中的图像，随机采样生成一对前背景交互，[datasets/voc_random_sample.py](./datasets/voc_random_sample.py)（训练集一轮1464个数据对、验证集一轮1449个数据对）

## 5. 后处理Graph Cut Optimization

## 6. 训练

## 7. 验证IoU、NOC(Number of Click)指标

## 8. 交互式分割应用Demo

参考论文：

- [Fully Convolutional Networks for Semantic Segmentation](https://arxiv.org/abs/1603.04042.pdf)
- [Deep Interactive Object Selection](https://arxiv.org/pdf/1411.4038.pdf)
