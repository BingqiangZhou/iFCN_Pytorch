# iFCN_Pytorch

修改[torchvision](https://pytorch.org/vision/stable/models.html#classification)给出的分类模型，实现FCN，并在FCN的基础上，实现iFCN。

## [相关介绍/实现细节](./files/interduction.md)

- [iFCN_Pytorch相关介绍](./files/interduction.md#ifcn_pytorch相关介绍)
  - [1. FCN网络实现](./files/interduction.md#1-fcn网络实现)
    - [1.1 修改图像分类网络](./files/interduction.md#11-修改图像分类网络)
    - [1.2 结合不同步长的浅层与深层的信息](./files/interduction.md#12-结合不同步长的浅层与深层的信息)
  - [2. iFCN网络实现](./files/interduction.md#2-ifcn网络实现)
    - [2.1 修改FCN网络第一层](./files/interduction.md#21-修改fcn网络第一层)
    - [2.2 微调FCN模型参数](./files/interduction.md#22-微调fcn模型参数)
  - [3. 随机采样](./files/interduction.md#3-随机采样)
  - [4. 数据对的生成与加载](./files/interduction.md#4-数据对的生成与加载)
    - [4.1 生成数据对](./files/interduction.md#41-生成数据对)
    - [4.2 加载数据集](./files/interduction.md#42-加载数据集)
  - [5. 后处理GraphCut Optimization](./files/interduction.md#5-后处理graphcut-optimization)
  - [6. 训练](./files/interduction.md#6-训练)
  - [7. 评估NOC(Number of Click)指标](./files/interduction.md#7-评估nocnumber-of-click指标)
  - [8. 交互式分割应用Demo](./files/interduction.md#8-交互式分割应用demo)
  - [参考论文](./files/interduction.md#参考论文)

## 参考论文

- AlexNet: [One weird trick for parallelizing convolutional neural networks](https://arxiv.org/abs/1404.5997)
- VGG: [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)
- GoogLeNet: [Going Deeper with Convolutions](https://arxiv.org/abs/1409.4842)
- ResNet: [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- FCN: [Fully Convolutional Networks for Semantic Segmentation](https://arxiv.org/abs/1411.4038)
- iFCN: [Deep Interactive Object Selection](https://arxiv.org/abs/1603.04042)

## iFCN模型以及验证集评估指标

**注：以下模型未经过调参与优化，是未完全收敛的模型。**

### AlexNet

|      网络名称      | IoU</br>(voc2012 val) | NOC</br>(85% IoU) | epochs | batch size</br>for training | 模型大小 |                                            模型下载地址                                            |
| :----------------: | :-----------------------: | :-------------------: | :----: | :-----------------------------: | :------: | :-------------------------------------------------------------------------------------------------: |
| AlexNet_32s_deconv |           48.1%           |         19.0         |   11   |               128               |  142MB  | [下载](https://github.com/BingqiangZhou/iFCN_Pytorch/releases/download/alexnet/alexnet_32s_deconv.pkl) |
| AlexNet_16s_deconv |           50.5%           |         18.8         |   18   |               128               |   78MB   | [下载](https://github.com/BingqiangZhou/iFCN_Pytorch/releases/download/alexnet/alexnet_16s_deconv.pkl) |
| AlexNet_8s_deconv |           54.6%           |         17.9         |   29   |               128               |   78MB   | [下载](https://github.com/BingqiangZhou/iFCN_Pytorch/releases/download/alexnet/alexnet_8s_deconv.pkl) |

### VGG系列

|     网络名称     | IoU</br>(voc2012 val) | NOC</br>(85% IoU) | epochs | batch size</br>for training | 模型大小 |                                         模型下载地址                                         |
| :--------------: | :-----------------------: | :-------------------: | :----: | :-----------------------------: | :------: | :-------------------------------------------------------------------------------------------: |
| VGG11_32s_deconv |           51.9%           |         18.6         |   8   |               48               |  171MB  | [下载](https://github.com/BingqiangZhou/iFCN_Pytorch/releases/download/vgg/vgg11_32s_deconv.pkl) |
| VGG11_16s_deconv |           52.0%           |         19.1         |   15   |               48               |  107MB  | [下载](https://github.com/BingqiangZhou/iFCN_Pytorch/releases/download/vgg/vgg11_16s_deconv.pkl) |
| VGG11_8s_deconv |           54.6%           |         18.7         |   9   |               48               |  107MB  | [下载](https://github.com/BingqiangZhou/iFCN_Pytorch/releases/download/vgg/vgg11_8s_deconv.pkl) |
|                 |                           |                       |       |                                 |         |                                                                                               |
| VGG13_32s_deconv |           52.2%           |         18.5         |   11   |               32               |  172MB  | [下载](https://github.com/BingqiangZhou/iFCN_Pytorch/releases/download/vgg/vgg13_32s_deconv.pkl) |
| VGG13_16s_deconv |           52.9%           |         18.9         |   11   |               32               |  108MB  | [下载](https://github.com/BingqiangZhou/iFCN_Pytorch/releases/download/vgg/vgg13_16s_deconv.pkl) |
| VGG13_8s_deconv |           59.4%           |         17.9         |   9   |               32               |  108MB  | [下载](https://github.com/BingqiangZhou/iFCN_Pytorch/releases/download/vgg/vgg13_8s_deconv.pkl) |
|                 |                           |                       |       |                                 |         |                                                                                               |
| VGG16_32s_deconv |           52.4%           |         18.5         |   8   |               32               |  192MB  | [下载](https://github.com/BingqiangZhou/iFCN_Pytorch/releases/download/vgg/vgg16_32s_deconv.pkl) |
| VGG16_16s_deconv |           56.0%           |         18.7         |   7   |               32               |  128MB  | [下载](https://github.com/BingqiangZhou/iFCN_Pytorch/releases/download/vgg/vgg16_16s_deconv.pkl) |
| VGG16_8s_deconv |           58.2%           |         18.2         |   9   |               32               |  128MB  | [下载](https://github.com/BingqiangZhou/iFCN_Pytorch/releases/download/vgg/vgg16_8s_deconv.pkl) |
|                 |                           |                       |       |                                 |         |                                                                                               |
| VGG19_32s_deconv |           53.2%           |         18.5         |   13   |               32               |  212MB  | [下载](https://github.com/BingqiangZhou/iFCN_Pytorch/releases/download/vgg/vgg19_32s_deconv.pkl) |
| VGG19_16s_deconv |           56.2%           |         18.8         |   17   |               32               |  149MB  | [下载](https://github.com/BingqiangZhou/iFCN_Pytorch/releases/download/vgg/vgg19_16s_deconv.pkl) |
| VGG19_8s_deconv |           61.7%           |         16.7         |   10   |               32               |  149MB  | [下载](https://github.com/BingqiangZhou/iFCN_Pytorch/releases/download/vgg/vgg19_8s_deconv.pkl) |

### GoogLeNet

|       网络名称       | IoU</br>(voc2012 val) | NOC</br>(85% IoU) | epochs | batch size</br>for training | 模型大小 |                                              模型下载地址                                              |
| :------------------: | :-----------------------: | :-------------------: | :----: | :-----------------------------: | :------: | :-----------------------------------------------------------------------------------------------------: |
| GoogLeNet_32s_deconv |           63.1%           |         16.1         |   7   |               48               |   41MB   | [下载](https://github.com/BingqiangZhou/iFCN_Pytorch/releases/download/googlenet/googlenet_32s_deconv.pkl) |
| GoogLeNet_16s_deconv |           66.7%           |         14.9         |   15   |               48               |   26MB   | [下载](https://github.com/BingqiangZhou/iFCN_Pytorch/releases/download/googlenet/googlenet_16s_deconv.pkl) |
| GoogLeNet_8s_deconv |           65.3%           |         15.7         |   10   |               48               |   26MB   | [下载](https://github.com/BingqiangZhou/iFCN_Pytorch/releases/download/googlenet/googlenet_8s_deconv.pkl) |

### ResNet系列

|       网络名称       | IoU</br>(voc2012 val) | NOC</br>(85% IoU) | epochs | batch size</br>for training | 模型大小 |                                             模型下载地址                                             |
| :------------------: | :-----------------------: | :-------------------: | :----: | :-----------------------------: | :------: | :--------------------------------------------------------------------------------------------------: |
| ResNet18_32s_deconv |           61.0%           |         16.7         |   8   |               96               |   53MB   | [下载](https://github.com/BingqiangZhou/iFCN_Pytorch/releases/download/resnet/resnet18_32s_deconv.pkl) |
| ResNet18_16s_deconv |           63.8%           |         16.5         |   6   |               96               |   45MB   | [下载](https://github.com/BingqiangZhou/iFCN_Pytorch/releases/download/resnet/resnet18_16s_deconv.pkl) |
|  ResNet18_8s_deconv  |           70.9%           |         13.3         |   11   |               96               |   45MB   |  [下载](https://github.com/BingqiangZhou/iFCN_Pytorch/releases/download/resnet/resnet18_8s_deconv.pkl)  |
|                     |                           |                       |       |                                 |         |                                                                                                     |
| ResNet34_32s_deconv |           61.5%           |         16.9         |   12   |               64               |   91MB   | [下载](https://github.com/BingqiangZhou/iFCN_Pytorch/releases/download/resnet/resnet34_32s_deconv.pkl) |
| ResNet34_16s_deconv |           64.2%           |         16.5         |   9   |               64               |   83MB   | [下载](https://github.com/BingqiangZhou/iFCN_Pytorch/releases/download/resnet/resnet34_16s_deconv.pkl) |
|  ResNet34_8s_deconv  |           72.2%           |         12.6         |   22   |               64               |   83MB   |  [下载](https://github.com/BingqiangZhou/iFCN_Pytorch/releases/download/resnet/resnet34_8s_deconv.pkl)  |
|                     |                           |                       |       |                                 |         |                                                                                                     |
| ResNet50_32s_deconv |           62.6%           |         16.2         |   16   |               32               |  130MB  | [下载](https://github.com/BingqiangZhou/iFCN_Pytorch/releases/download/resnet/resnet50_32s_deconv.pkl) |
| ResNet50_16s_deconv |           64.8%           |         15.2         |   8   |               32               |   98MB   | [下载](https://github.com/BingqiangZhou/iFCN_Pytorch/releases/download/resnet/resnet50_16s_deconv.pkl) |
|  ResNet50_8s_deconv  |           71.8%           |         12.4         |   31   |               32               |   98MB   |  [下载](https://github.com/BingqiangZhou/iFCN_Pytorch/releases/download/resnet/resnet50_8s_deconv.pkl)  |
|                     |                           |                       |       |                                 |         |                                                                                                     |
| ResNet101_32s_deconv |           63.0%           |         16.0         |   20   |               24               |  203MB  | [下载](https://github.com/BingqiangZhou/iFCN_Pytorch/releases/download/resnet/resnet101_32s_deconv.pkl) |
| ResNet101_16s_deconv |           65.4%           |         15.2         |   13   |               24               |  171MB  | [下载](https://github.com/BingqiangZhou/iFCN_Pytorch/releases/download/resnet/resnet101_16s_deconv.pkl) |
| ResNet101_8s_deconv |           73.1%           |         12.2         |   17   |               24               |  171MB  | [下载](https://github.com/BingqiangZhou/iFCN_Pytorch/releases/download/resnet/resnet101_8s_deconv.pkl) |
|                     |                           |                       |       |                                 |         |                                                                                                     |
| ResNet152_32s_deconv |           62.3%           |         16.3         |   8   |               16               |  263MB  | [下载](https://github.com/BingqiangZhou/iFCN_Pytorch/releases/download/resnet/resnet152_32s_deconv.pkl) |
| ResNet152_16s_deconv |           64.7%           |         15.5         |   12   |               16               |  231MB  | [下载](https://github.com/BingqiangZhou/iFCN_Pytorch/releases/download/resnet/resnet152_16s_deconv.pkl) |
| ResNet152_8s_deconv |           73.3%           |         11.9         |   13   |               16               |  231MB  | [下载](https://github.com/BingqiangZhou/iFCN_Pytorch/releases/download/resnet/resnet152_8s_deconv.pkl) |
