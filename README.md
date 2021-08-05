
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

- 由于不是所有的分类模型中都包括论文中对应的池化层，上面的描述中将原文中的相应步长的池化层的结果改成了相应步长的特征。
- 为了保证任意大小的输入，以及上采样后的大小与预测的大小相同，这里利用了一个额外的双边上采样来保证这一点。

## 2. iFCN网络实现

### 2.1 修改FCN网络第一层

根据[Deep Interactive Object Selection](https://arxiv.org/pdf/1411.4038.pdf)中`3. The proposed algorithm`的描述，将前背景交互的距离图作为额外的通道与原图像级联输入到FCN模型中，因此需要修改FCN网络的第一层的输入层通道为5层。

### 2.2 微调FCN模型参数

根据[Deep Interactive Object Selection](https://arxiv.org/pdf/1411.4038.pdf)中`3.3. Fine tuning FCN models`的描述，作者尝试了将额外两个输入通道对应模型参数的设置为0以及设置为已有参数的平均值，但是在实际实验中，这两种参数初始化方式得到的结果没有什么差别。

需要注意的是: 由于这里的FCN基于[torchvision](https://pytorch.org/vision/stable/models.html#classification)中的图像分类网络而来，因此这里不加载FCN模型参数。

## 3. 随机采样

## 4. 数据加载、训练

## 5. 后处理Graph Cut Optimization

## 6. 验证IoU、NOC(Number of Click)指标

## 7. 交互式分割应用Demo

参考论文：

- [Fully Convolutional Networks for Semantic Segmentation](https://arxiv.org/abs/1603.04042.pdf)
- [Deep Interactive Object Selection](https://arxiv.org/pdf/1411.4038.pdf)
