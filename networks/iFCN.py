import torch
import torch.nn as nn

from .FCN import FCN
from .utils import upsample_size_to_target, BasicConv2d

class iFCN(nn.Module):
    '''
        reference paper:
            Deep Interactive Object Selection, https://arxiv.org/abs/1603.04042.pdf
    '''
    def __init__(self, backbone_name, in_channels=5, num_classes=1, stride_out=8, upsample_type='deconv') -> None:
        super(iFCN, self).__init__()
        assert backbone_name in ['AlexNet', 'GoogLeNet', 'VGG11', 'VGG11_BN',
                                'VGG13', 'VGG13_BN', 'VGG16', 'VGG16_BN', 'VGG19', 'VGG19_BN',
                                'ResNet18', 'ResNet34', 'ResNet50', 'ResNet101', 'ResNet152']
        assert stride_out in [8, 16, 32] # fcn_8s, fcn_16s, fcn_32s
        assert upsample_type in ['deconv', 'interpolate']
        
        self.fcn = FCN(backbone_name, num_classes, stride_out=stride_out, upsample_type=upsample_type)

        # modify first conv's in_channels
        if backbone_name == 'GoogLeNet':
            first_conv = self.fcn.backbone_net.net.conv1.conv
            use_bias = True if first_conv.bias is not False else False
            self.fcn.backbone_net.net.conv1.conv = nn.Conv2d(in_channels, first_conv.out_channels, first_conv.kernel_size, 
                                                            first_conv.stride, first_conv.padding, bias=use_bias)
            # self.fcn.backbone_net.net.conv1 = BasicConv2d(in_channels, first_conv.out_channels, 
            #                                             kernel_size=first_conv.kernel_size, stride=first_conv.stride, 
            #                                             padding=first_conv.padding, bias=use_bias)
        elif 'VGG' in backbone_name:
            convs = list(self.fcn.backbone_net.net.conv1)
            first_conv = convs[0]
            use_bias = True if first_conv.bias is not False else False
            in_channels = 5
            convs[0] = nn.Conv2d(in_channels, first_conv.out_channels, first_conv.kernel_size, 
                                                        first_conv.stride, first_conv.padding, bias=use_bias)
            self.fcn.backbone_net.net.conv1 = nn.Sequential(*convs)
        else:
            first_conv = self.fcn.backbone_net.net.conv1
            use_bias = True if first_conv.bias is not False else False
            self.fcn.backbone_net.net.conv1 = nn.Conv2d(in_channels, first_conv.out_channels, first_conv.kernel_size, 
                                                        first_conv.stride, first_conv.padding, bias=use_bias)

    def forward(self, x):
        out = self.fcn(x)
        out = upsample_size_to_target(out, x)
        return out


# # net = iFCN('ResNet18', stride_out=8, upsample_type='deconv')
# # net = iFCN('AlexNet', stride_out=8, upsample_type='deconv')
# net = iFCN('AlexNet', stride_out=8, upsample_type='interpolate')
# net = iFCN('GoogLeNet', stride_out=8, upsample_type='deconv')
# net = iFCN('VGG11', stride_out=8, upsample_type='deconv')
# # net = iFCN('VGG11_BN', stride_out=8, upsample_type='deconv')
# print(net)

# x = torch.rand(1, 5, 384, 384)
# out = net(x)
# print(out.shape)
