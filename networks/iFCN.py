import torch
import torch.nn as nn
import torch.nn.functional as F

from FCN import FCN

class iFCN(nn.Module):
    '''
        reference paper:
            Deep Interactive Object Selection, https://arxiv.org/abs/1603.04042.pdf
    '''
    def __init__(self, backbone_name, in_channels=5, num_classes=1, stride_out=8, upsample_type='deconv', ) -> None:
        super(iFCN, self).__init__()
        assert backbone_name in ['AlexNet', 'VGG', 
                            'ResNet18', 'ResNet34', 'ResNet50', 'ResNet101', 'ResNet152', 'SqueezeNet']
        assert stride_out in [8, 16, 32] # fcn_8s, fcn_16s, fcn_32s
        assert upsample_type in ['deconv', 'interpolate']
        
        self.fcn = FCN(backbone_name, num_classes, stride_out=stride_out, upsample_type=upsample_type)

        # modify first conv's in_channels
        first_conv = self.fcn.backbone_net.net.conv1
        use_bias = True if first_conv.bias is not False else False
        self.fcn.backbone_net.net.conv1 = nn.Conv2d(in_channels, first_conv.out_channels, first_conv.kernel_size, 
                                                    first_conv.stride, first_conv.padding, bias=use_bias)

    def forward(self, x):
        out = self.fcn(x)
        _, _, h1, w1 =  out.shape
        _, _, h2, w2 =  x.shape
        if h1 != h2 or w1 != w2:
            out = F.interpolate(out, (h2, w2))
        return out

x = torch.rand(1, 5, 384, 384)
# net = iFCN('ResNet18')
net = iFCN('AlexNet', stride_out=16, upsample_type='interpolate')

out = net(x)
print(out.shape)