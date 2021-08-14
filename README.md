
# iFCN_Pytorch

修改[torchvision](https://pytorch.org/vision/stable/models.html#classification)给出的分类模型，实现FCN，并在FCN的基础上，实现iFCN。[相关介绍](./interduction.md)

## 参考论文

- [Fully Convolutional Networks for Semantic Segmentation](https://arxiv.org/abs/1603.04042.pdf)
- [Deep Interactive Object Selection](https://arxiv.org/pdf/1411.4038.pdf)

## iFCN模型以及验证集评估指标

### AlexNet

| 网络名称 | IoU | NOC (85% IoU)| epochs | 模型大小 | 模型下载地址 |
| :---: | :---: | :---: | :---:| :---:| :---:|
| AlexNet_32s_deconv | 48.1% | - | 11 | 142MB | [下载](https://github.com/BingqiangZhou/iFCN_Pytorch/releases/tag/alexnet) |
| AlexNet_16s_deconv | 50.5% | - | 18 | 78MB | [下载](https://github.com/BingqiangZhou/iFCN_Pytorch/releases/tag/alexnet) |
| AlexNet_8s_deconv | 54.6% | - | 29 | 78MB | [下载](https://github.com/BingqiangZhou/iFCN_Pytorch/releases/tag/alexnet) |

### ResNet系列

| 网络名称 | IoU | NOC(85% IoU)| epochs |模型大小 | 模型下载地址 |
| :---: | :---: | :---: | :---:| :---: | :---: |
| ResNet18_32s_deconv | 61.0% | - | 8 | 53MB | [下载](https://github.com/BingqiangZhou/iFCN_Pytorch/releases/tag/resnet) |
| ResNet18_16s_deconv | 63.8% | - | 6 | 45MB | [下载](https://github.com/BingqiangZhou/iFCN_Pytorch/releases/tag/resnet) |
| ResNet18_8s_deconv | 70.9% | - | 11 | 45MB | [下载](https://github.com/BingqiangZhou/iFCN_Pytorch/releases/tag/resnet) |
|||||
| ResNet34_32s_deconv | 61.5% | - | 12 | 91MB | [下载](https://github.com/BingqiangZhou/iFCN_Pytorch/releases/tag/resnet) |
| ResNet34_16s_deconv | 64.2% | - | 9 | 83MB | [下载](https://github.com/BingqiangZhou/iFCN_Pytorch/releases/tag/resnet) |
| ResNet34_8s_deconv | 72.2% | - | 22 | 83MB | [下载](https://github.com/BingqiangZhou/iFCN_Pytorch/releases/tag/resnet) |
|||||
| ResNet50_32s_deconv | 62.6% | - | 16 | 130MB | [下载](https://github.com/BingqiangZhou/iFCN_Pytorch/releases/tag/resnet) |
| ResNet50_16s_deconv | 64.8% | - | 8 | 98MB | [下载](https://github.com/BingqiangZhou/iFCN_Pytorch/releases/tag/resnet) |
| ResNet50_8s_deconv | 71.8% | - | 31 | 98MB | [下载](https://github.com/BingqiangZhou/iFCN_Pytorch/releases/tag/resnet) |
|||||
| ResNet101_32s_deconv | - | - | - | - | [下载](https://github.com/BingqiangZhou/iFCN_Pytorch/releases/tag/resnet) |
| ResNet101_16s_deconv | - | - | - | - | [下载](https://github.com/BingqiangZhou/iFCN_Pytorch/releases/tag/resnet) |
| ResNet101_8s_deconv | 73.1% | - | 17 | 171MB | [下载](https://github.com/BingqiangZhou/iFCN_Pytorch/releases/tag/resnet) |
|||||
| ResNet152_32s_deconv | - | - | - | - | [下载](https://github.com/BingqiangZhou/iFCN_Pytorch/releases/tag/resnet) |
| ResNet152_16s_deconv | - | - | - | - | [下载](https://github.com/BingqiangZhou/iFCN_Pytorch/releases/tag/resnet) |
| ResNet152_8s_deconv | - | - | - | - | [下载](https://github.com/BingqiangZhou/iFCN_Pytorch/releases/tag/resnet) |
