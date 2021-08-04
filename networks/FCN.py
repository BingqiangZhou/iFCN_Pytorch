import torch.nn as nn
import torch.nn.functional as F
from importlib import import_module

from utils import upsample_size_to_target

class FCN(nn.Module):
    '''
        reference paper:
            Fully Convolutional Networks for Semantic Segmentation, https://arxiv.org/pdf/1411.4038.pdf
    '''
    def __init__(self, backbone_name, num_classes=21, stride_out=8, upsample_type='deconv') -> None:
        super(FCN, self).__init__()
        assert backbone_name in ['AlexNet', 'VGG', 
                            'ResNet18', 'ResNet34', 'ResNet50', 'ResNet101', 'ResNet152', 'SqueezeNet']
        assert stride_out in [8, 16, 32] # fcn_8s, fcn_16s, fcn_32s
        assert upsample_type in ['deconv', 'interpolate']
        
        if 'ResNet' in backbone_name:
            self.backbone_net = import_module('classifiers.ResNet', 'networks').NetRemoveFCLayer(backbone_name.lower())
        else:
            self.backbone_net = import_module(f'classifiers.{backbone_name}', 'networks').NetRemoveFCLayer()
        
        self.classifier = FCNClassifier(self.backbone_net.out_channels, num_classes, stride_out=8, upsample_type=upsample_type)

    def forward(self, x):
        x_s32, x_s16, x_s8 = self.backbone_net(x)
        # print(x_s32.shape, x_s16.shape, x_s8.shape)
        out = self.classifier(x_s32, x_s16, x_s8)
        return out

class FCNClassifier(nn.Module):
    def __init__(self, in_channels, num_classes=21, stride_out=32, upsample_type='deconv'):
        super(FCNClassifier, self).__init__()
        assert stride_out in [8, 16, 32] # fcn_8s, fcn_16s, fcn_32s
        assert upsample_type in ['deconv', 'interpolate']
        
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
    def __init__(self, in_channels, num_classes=21, upsample_type='deconv'):
        super(FCNClassifier32s, self).__init__()
        assert upsample_type in ['deconv', 'interpolate']
        self.upsample_s32 = upsample_layer(32, upsample_type=upsample_type, in_channels=in_channels[0], out_channels=num_classes)
        self.classifier = nn.Conv2d(num_classes, num_classes, 1)
    
    def forward(self, x_s32, x_s16, x_s8):
        out = self.upsample_s32(x_s32)
        out = self.classifier(out)
        return out

class FCNClassifier16s(nn.Module):
    def __init__(self, in_channels, num_classes=21, upsample_type='deconv'):
        super(FCNClassifier16s, self).__init__()
        assert upsample_type in ['deconv', 'interpolate']
        self.upsample_s32 = upsample_layer(2, upsample_type=upsample_type, in_channels=in_channels[0], out_channels=num_classes)
        self.conv_s16 = nn.Conv2d(in_channels[1], num_classes, 1, 1, bias=False)
        
        self.upsample_s16 = upsample_layer(16, upsample_type=upsample_type, in_channels=num_classes, out_channels=num_classes)
        self.classifier = nn.Conv2d(num_classes, num_classes, 1)

    def forward(self, x_s32, x_s16, x_s8):
        out_s16_1 = self.upsample_s32(x_s32)
        out_s16_2 = self.conv_s16(x_s16)
        
        out_s16_1 = upsample_size_to_target(out_s16_1, out_s16_2)
        
        out = self.upsample_s16(out_s16_1 + out_s16_2)
        out = self.classifier(out)
        return out

class FCNClassifier8s(nn.Module):
    def __init__(self, in_channels, num_classes=21, upsample_type='deconv'):
        super(FCNClassifier8s, self).__init__()
        assert upsample_type in ['deconv', 'interpolate']

        self.upsample_s32 = upsample_layer(2, upsample_type=upsample_type, in_channels=in_channels[0], out_channels=num_classes)
        
        self.conv_s16 = nn.Conv2d(in_channels[1], num_classes, 1, 1, bias=False)
        self.upsample_s16 = upsample_layer(2, upsample_type=upsample_type, in_channels=num_classes, out_channels=num_classes)
        
        self.conv_s8 = nn.Conv2d(in_channels[2], num_classes, 1, 1, bias=False)
        self.upsample_s8 = upsample_layer(8, upsample_type=upsample_type, in_channels=num_classes, out_channels=num_classes)
        self.classifier = nn.Conv2d(num_classes, num_classes, 1)

    def forward(self, x_s32, x_s16, x_s8):
        out_s16_1 = self.upsample_s32(x_s32)
        out_s16_2 = self.conv_s16(x_s16)
        
        out_s16_1 = upsample_size_to_target(out_s16_1, out_s16_2)

        out_s8_1 = self.upsample_s16(out_s16_1 + out_s16_2)
        out_s8_2 = self.conv_s8(x_s8)
        
        out_s8_1 = upsample_size_to_target(out_s8_1, out_s8_2)

        out = self.upsample_s8(out_s8_1 + out_s8_2)
        out = self.classifier(out)
        return out

def upsample_layer(scale_factor, upsample_type='deconv', in_channels=0, out_channels=0):
    if upsample_type == 'deconv':
        upsample = nn.ConvTranspose2d(in_channels, out_channels, scale_factor, scale_factor, bias=False)
    else:
        upsample = nn.Sequential(
            nn.Upsample(scale_factor=scale_factor),
            nn.Conv2d(in_channels, out_channels, 1)
        )
    return upsample
