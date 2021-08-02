import torch
import torch.nn as nn
from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck, model_urls

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

def resnet_forward_impl(self, x):
    
    x_s2 = self.conv1(x) # (h/2, w/2)
    x_s2 = self.bn1(x_s2)
    x_s2 = self.relu(x_s2)
    x_s4 = self.maxpool(x_s2) # (h/4, w/4)

    x_s4 = self.layer1(x_s4) # (h/4, w/4)
    x_s8 = self.layer2(x_s4) # (h/8, w/8)
    x_s16 = self.layer3(x_s8) # (h/16, w/16)
    x_s32 = self.layer4(x_s16) # (h/32, w/32)
    
    # x = self.avgpool(x)
    # x = torch.flatten(x, 1)
    # x = self.fc(x)

    return x_s32, x_s16, x_s8

class NetRemoveFCLayer(nn.Module):
    '''
        reference urls:
            https://pytorch.org/hub/pytorch_vision_resnet/
            https://pytorch.org/vision/stable/_modules/torchvision/models/resnet.html
    '''

    def __init__(self, arch: str, num_classes=21, pretrained: bool = False, progress: bool = True, **kwargs):
        super(NetRemoveFCLayer, self).__init__()
        assert arch in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']

        self.model_infos = {
            'resnet18': {
                'block': BasicBlock,
                'layers': [2, 2, 2, 2] # 1(conv1) + (2+2+2+2)x2(conv2-conv5) + 1(fc) = 18
            },
            'resnet34': {
                'block': BasicBlock,
                'layers': [3, 4, 6, 3] # 1(conv1) + (3+4+6+3)x2(conv2-conv5) + 1(fc) = 34
            },
            'resnet50': {
                'block': Bottleneck,
                'layers': [3, 4, 6, 3] # 1(conv1) + (3+4+6+3)x3(conv2-conv5) + 1(fc) = 50
            },
            'resnet101': {
                'block': Bottleneck,
                'layers': [3, 4, 23, 3] # 1(conv1) + (3+4+23+3)x3(conv2-conv5) + 1(fc) = 101
            },
            'resnet152': {
                'block': Bottleneck,
                'layers': [3, 8, 36, 3] # 1(conv1) + (3+8+36+3)x3(conv2-conv5) + 1(fc) = 152
            },
        }
        
        block = self.model_infos[arch]['block']
        layers = self.model_infos[arch]['layers']
        model_url = model_urls[arch]

        self.net = ResNet(block, layers, **kwargs)
        
        # out channel of resnet's conv layers
        out_channel = 512 if self.model_infos[arch]['block'] is BasicBlock else 2048 
        self.out_channels = [out_channel, out_channel//2, out_channel//4]

        if pretrained:
            state_dict = load_state_dict_from_url(model_url, progress=progress)
            self.net.load_state_dict(state_dict)
        
        # overload forward function
        self.net.forward = lambda x: resnet_forward_impl(self.net, x) # remove avgpool and fc layer
        # self.conv_fc = nn.Conv2d(self.out_channels, num_classes, 1)

    def forward(self, x):
        x_s32, x_s16, x_s8 = self.net(x)
        # x_s32 = self.conv_fc(x_s32)
        return x_s32, x_s16, x_s8