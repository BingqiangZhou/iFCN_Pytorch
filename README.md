
# iFCN_Pytorch

修改[torchvision](https://pytorch.org/vision/stable/models.html#classification)给出的分类模型，实现FCN，并在FCN的基础上，实现iFCN。[相关介绍](./interduction.md)

## 参考论文

- AlexNet: [One weird trick for parallelizing convolutional neural networks](https://arxiv.org/abs/1404.5997)
- VGG: [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)
- GoogLeNet: [Going Deeper with Convolutions](https://arxiv.org/abs/1409.4842)
- ResNet: [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- FCN: [Fully Convolutional Networks for Semantic Segmentation](https://arxiv.org/abs/1411.4038)
- iFCN: [Deep Interactive Object Selection](https://arxiv.org/abs/1603.04042)

## iFCN模型以及验证集评估指标

### AlexNet

| 网络名称 | IoU | NOC(85% IoU)| epochs | 模型大小 | 模型下载地址 |
| :---: | :---: | :---: | :---:| :---:| :---:|
| AlexNet_32s_deconv | 48.1% | 19.0 | 11 | 142MB | [下载](https://github.com/BingqiangZhou/iFCN_Pytorch/releases/tag/alexnet) |
| AlexNet_16s_deconv | 50.5% | 18.8 | 18 | 78MB | [下载](https://github.com/BingqiangZhou/iFCN_Pytorch/releases/tag/alexnet) |
| AlexNet_8s_deconv | 54.6% | 17.9 | 29 | 78MB | [下载](https://github.com/BingqiangZhou/iFCN_Pytorch/releases/tag/alexnet) |

### VGG

### GoogLeNet

### ResNet系列

| 网络名称 | IoU | NOC(85% IoU)| epochs |模型大小 | 模型下载地址 |
| :---: | :---: | :---: | :---:| :---: | :---: |
| ResNet18_32s_deconv | 61.0% | 16.7 | 8 | 53MB | [下载](https://github.com/BingqiangZhou/iFCN_Pytorch/releases/tag/resnet) |
| ResNet18_16s_deconv | 63.8% | 16.5 | 6 | 45MB | [下载](https://github.com/BingqiangZhou/iFCN_Pytorch/releases/tag/resnet) |
| ResNet18_8s_deconv | 70.9% | 13.3 | 11 | 45MB | [下载](https://github.com/BingqiangZhou/iFCN_Pytorch/releases/tag/resnet) |
|||||
| ResNet34_32s_deconv | 61.5% | 16.9 | 12 | 91MB | [下载](https://github.com/BingqiangZhou/iFCN_Pytorch/releases/tag/resnet) |
| ResNet34_16s_deconv | 64.2% | 16.5 | 9 | 83MB | [下载](https://github.com/BingqiangZhou/iFCN_Pytorch/releases/tag/resnet) |
| ResNet34_8s_deconv | 72.2% | 12.6 | 22 | 83MB | [下载](https://github.com/BingqiangZhou/iFCN_Pytorch/releases/tag/resnet) |
|||||
| ResNet50_32s_deconv | 62.6% | - | 16 | 130MB | [下载](https://github.com/BingqiangZhou/iFCN_Pytorch/releases/tag/resnet) |
| ResNet50_16s_deconv | 64.8% | - | 8 | 98MB | [下载](https://github.com/BingqiangZhou/iFCN_Pytorch/releases/tag/resnet) |
| ResNet50_8s_deconv | 71.8% | - | 31 | 98MB | [下载](https://github.com/BingqiangZhou/iFCN_Pytorch/releases/tag/resnet) |
|||||
| ResNet101_32s_deconv | 63.0% | - | 20 | 203MB | [下载](https://github.com/BingqiangZhou/iFCN_Pytorch/releases/tag/resnet) |
| ResNet101_16s_deconv | 65.4% | - | 13 | 171MB | [下载](https://github.com/BingqiangZhou/iFCN_Pytorch/releases/tag/resnet) |
| ResNet101_8s_deconv | 73.1% | - | 17 | 171MB | [下载](https://github.com/BingqiangZhou/iFCN_Pytorch/releases/tag/resnet) |
|||||
| ResNet152_32s_deconv | - | - | - | - | [下载](https://github.com/BingqiangZhou/iFCN_Pytorch/releases/tag/resnet) |
| ResNet152_16s_deconv | - | - | - | - | [下载](https://github.com/BingqiangZhou/iFCN_Pytorch/releases/tag/resnet) |
| ResNet152_8s_deconv | 73.3% | - | 13 | 231MB | [下载](https://github.com/BingqiangZhou/iFCN_Pytorch/releases/tag/resnet) |
