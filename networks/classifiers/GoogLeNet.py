import torch
import torch.nn as nn
from torchvision.models.googlenet import GoogLeNet, model_urls

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

def googlenet_forward_impl(self, x):
    
    # N x 3 x 224 x 224
    x = self.conv1(x)
    # N x 64 x 112 x 112
    x = self.maxpool1(x)
    # N x 64 x 56 x 56
    x = self.conv2(x)
    # N x 64 x 56 x 56
    x = self.conv3(x)
    # N x 192 x 56 x 56
    x = self.maxpool2(x) 

    # N x 192 x 28 x 28
    x_s8 = self.inception3a(x) # x_s8 [n, 256, h, w]
    # N x 256 x 28 x 28
    x = self.inception3b(x_s8)
    # N x 480 x 28 x 28
    x = self.maxpool3(x)
    # N x 480 x 14 x 14
    x = self.inception4a(x) 
    # N x 512 x 14 x 14
    # aux1: Optional[Tensor] = None
    # if self.aux1 is not None:
    #     if self.training:
    #         aux1 = self.aux1(x)

    x_s16 = self.inception4b(x) # x_s16 [n, 512, h, w]
    # N x 512 x 14 x 14
    x = self.inception4c(x)
    # N x 512 x 14 x 14
    x = self.inception4d(x) 
    # N x 528 x 14 x 14
    # aux2: Optional[Tensor] = None
    # if self.aux2 is not None:
    #     if self.training:
    #         aux2 = self.aux2(x)

    x = self.inception4e(x)
    # N x 832 x 14 x 14
    x = self.maxpool4(x)
    # N x 832 x 7 x 7
    x = self.inception5a(x)
    # N x 832 x 7 x 7
    x_s32 = self.inception5b(x) # x_s32 [n, 1024, h, w]
    # N x 1024 x 7 x 7

    # x = self.avgpool(x)
    # # N x 1024 x 1 x 1
    # x = torch.flatten(x, 1)
    # # N x 1024
    # x = self.dropout(x)
    # x = self.fc(x)
    # # N x 1000 (num_classes)

    return x_s32, x_s16, x_s8

def googlenet_forward(self, x):
    # x = self._transform_input(x)
    x_s32, x_s16, x_s8 = self._forward(x)
    return x_s32, x_s16, x_s8

class NetWithConvFC(nn.Module):
    '''
        reference urls:
            https://pytorch.org/hub/pytorch_vision_googlenet/
            https://pytorch.org/vision/stable/_modules/torchvision/models/googlenet.html
    '''

    def __init__(self, pretrained: bool = False, progress: bool = True, **kwargs):
        super(NetWithConvFC, self).__init__()

        kwargs['init_weights'] = False #not pretrained
        self.net = GoogLeNet(**kwargs)
        self.net.aux_logits = False
        self.net.aux1 = None  # type: ignore[assignment]
        self.net.aux2 = None  # type: ignore[assignment]
        out_channel = 1024 # out channel of resnet's conv layers
        self.out_channels = [out_channel, out_channel//2, out_channel//4]

        if pretrained:
            state_dict = load_state_dict_from_url(model_urls['googlenet'], progress=progress)
            self.net.load_state_dict(state_dict)
        
        # overload forward function
        self.net._forward = lambda x: googlenet_forward_impl(self.net, x) # remove avgpool and fc layer
        self.net.forward = lambda x: googlenet_forward(self.net, x) # overload forward
        # self.conv_fc = nn.Conv2d(self.out_channels, num_classes, 1)

    def forward(self, x):
        x_s32, x_s16, x_s8 = self.net(x)
        # x_s32 = self.conv_fc(x_s32)
        return x_s32, x_s16, x_s8