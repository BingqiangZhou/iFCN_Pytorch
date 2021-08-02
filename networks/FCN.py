import torch.nn as nn
import torch.nn.functional as F
from importlib import import_module

class FCN(nn.Module):
    '''
        reference paper:
            Fully Convolutional Networks for Semantic Segmentation, https://arxiv.org/pdf/1411.4038.pdf
    '''
    def __init__(self, backbone_name, num_classes=21) -> None:
        super(FCN, self).__init__()
        assert backbone_name in ['AlexNet', 'VGG', 
                            'ResNet18', 'ResNet34', 'ResNet50', 'ResNet101', 'ResNet152', 'SqueezeNet']
        
        if 'ResNet' in backbone_name:
            self.backbone_net = import_module('classifiers.ResNet', 'networks').NetRemoveFCLayer(backbone_name.lower())
        else:
            self.backbone_net = import_module(f'classifiers.{backbone_name}', 'networks').NetRemoveFCLayer()
        
        self.classifier = FCNClassifier(self.backbone_net.out_channels, num_classes, stride_out=8)

    def forward(self, x):
        x_s32, x_s16, x_s8 = self.backbone_net(x)
        print(x_s32.shape, x_s16.shape, x_s8.shape)
        out = self.classifier(x_s32, x_s16, x_s8)
        return out

class FCNClassifier(nn.Module):
    def __init__(self, in_channels, num_classes=21, stride_out=32):
        super(FCNClassifier, self).__init__()
        assert stride_out in [8, 16, 32] # fcn_8s, fcn_16s, fcn_32s
        
        if stride_out == 32:
            self.classifier = FCNClassifier32s(in_channels, num_classes)
        elif stride_out == 16:
            self.classifier = FCNClassifier16s(in_channels, num_classes)
        else:
            self.classifier = FCNClassifier8s(in_channels, num_classes)

    def forward(self, x_s32, x_s16, x_s8):
        out = self.classifier(x_s32, x_s16, x_s8)
        return out

class FCNClassifier32s(nn.Module):
    def __init__(self, in_channels, num_classes=21):
        super(FCNClassifier32s, self).__init__()

        self.upsample_s32 = nn.ConvTranspose2d(in_channels[0], num_classes, 32, 32, bias=False)

        self.classifier = nn.Conv2d(num_classes, num_classes, 1)
    
    def forward(self, x_s32, x_s16, x_s8):
        out = self.upsample_s32(x_s32)
        out = self.classifier(out)
        return out

class FCNClassifier16s(nn.Module):
    def __init__(self, in_channels, num_classes=21):
        super(FCNClassifier16s, self).__init__()

        self.upsample_s32 = nn.ConvTranspose2d(in_channels[0], num_classes, 2, 2, bias=False)
        self.conv_s16 = nn.Conv2d(in_channels[1], num_classes, 1, 1, bias=False)
        
        self.upsample_s16 = nn.ConvTranspose2d(num_classes, num_classes, 16, 16, bias=False)
        self.classifier = nn.Conv2d(num_classes, num_classes, 1)

    def forward(self, x_s32, x_s16, x_s8):
        out_s16_1 = self.upsample_s32(x_s32)
        out_s16_2 = self.conv_s16(x_s16)
        _, _, h1, w1 =  out_s16_1.shape
        _, _, h2, w2 =  out_s16_2.shape
        if h1 != h2 or w1 != w2:
            out_s16_1 = F.interpolate(out_s16_1, (h2, w2))
        
        out = self.upsample_s16(out_s16_1 + out_s16_2)
        out = self.classifier(out)
        return out

class FCNClassifier8s(nn.Module):
    def __init__(self, in_channels, num_classes=21):
        super(FCNClassifier8s, self).__init__()

        self.upsample_s32 = nn.ConvTranspose2d(in_channels[0], num_classes, 2, 2, bias=False)
        
        self.conv_s16 = nn.Conv2d(in_channels[1], num_classes, 1, 1, bias=False)
        self.upsample_s16 = nn.ConvTranspose2d(num_classes, num_classes, 2, 2, bias=False)
        
        self.upsample_s8 = nn.ConvTranspose2d(num_classes, num_classes, 2, 2, bias=False)
        self.conv_s8 = nn.Conv2d(in_channels[2], num_classes, 1, 1, bias=False)
        
        self.upsample_s8 = nn.ConvTranspose2d(num_classes, num_classes, 8, 8, bias=False)
        self.classifier = nn.Conv2d(num_classes, num_classes, 1)

    def forward(self, x_s32, x_s16, x_s8):
        out_s16_1 = self.upsample_s32(x_s32)
        out_s16_2 = self.conv_s16(x_s16)
        _, _, h1, w1 =  out_s16_1.shape
        _, _, h2, w2 =  out_s16_2.shape
        if h1 != h2 or w1 != w2:
            out_s16_1 = F.interpolate(out_s16_1, (h2, w2))

        out_s8_1 = self.upsample_s16(out_s16_1 + out_s16_2)
        out_s8_2 = self.conv_s8(x_s8)
        _, _, h1, w1 =  out_s8_1.shape
        _, _, h2, w2 =  out_s8_2.shape
        if h1 != h2 or w1 != w2:
            out_s8_1 = F.interpolate(out_s8_1, (h2, w2))

        out = self.upsample_s8(out_s8_1 + out_s8_2)
        out = self.classifier(out)
        return out
