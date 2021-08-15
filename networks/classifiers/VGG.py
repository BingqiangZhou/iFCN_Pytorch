import torch
import torch.nn as nn
from torchvision.models.vgg import model_urls
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

class NetWithConvFC(nn.Module):
    def __init__(self, arch, pretrained=False, progress=True):
        super(NetWithConvFC, self).__init__()

        assert arch in ['vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19_bn', 'vgg19']
        
        init_weights = not pretrained

        self.net = VGGWithConvFC(arch, init_weights)
        self.out_channels = [4096, 512, 256]
        if pretrained:
            state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
            state_dict = self._update_state_dict(state_dict)
            self.net.load_state_dict(state_dict)

    def _update_state_dict(self, old_state_dict):
        new_state_dict = self.net.state_dict()
        old_state_dict_keys = list(old_state_dict.keys())
        layer_index = 0
        for k, v in new_state_dict.items():
            if 'fc' not in k:
                if 'num_batches_tracked' in k:
                    continue
                new_state_dict[k] = old_state_dict[old_state_dict_keys[layer_index]]
                layer_index += 1
            else:
                break
        return new_state_dict

    def forward(self, x):
        x_s32, x_s16, x_s8 = self.net(x)
        return x_s32, x_s16, x_s8


class VGGWithConvFC(nn.Module):
    '''
        reference urls:
            https://pytorch.org/vision/stable/_modules/torchvision/models/vgg.html
    '''
    def __init__(self, arch, init_weights=True):
        super(VGGWithConvFC, self).__init__()

        assert arch in ['vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19_bn', 'vgg19']

        # there is a maxpool layer between the conv in_channl list and list
        cfgs = {
            'vgg11': [[64], [128], [256, 256], [512, 512], [512, 512]],
            'vgg13': [[64, 64], [128, 128], [256, 256], [512, 512], [512, 512]],
            'vgg16': [[64, 64], [128, 128], [256, 256, 256], [512, 512, 512], [512, 512, 512]],
            'vgg19': [[64, 64], [128, 128], [256, 256, 256, 256], [512, 512, 512, 512], [512, 512, 512, 512]],
        }
        vgg_cfg = cfgs[arch[:5]]

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        with_batch_norm = True if 'bn' in arch else False

        in_channels = 3
        out_channel_list = vgg_cfg[0]
        self.conv1 = self._make_layers(in_channels, out_channel_list, batch_norm=with_batch_norm)
        in_channels = out_channel_list[-1]
        out_channel_list = vgg_cfg[1]
        self.conv2 = self._make_layers(in_channels, out_channel_list, batch_norm=with_batch_norm)
        in_channels = out_channel_list[-1]
        out_channel_list = vgg_cfg[2]
        self.conv3 = self._make_layers(in_channels, out_channel_list, batch_norm=with_batch_norm)
        in_channels = out_channel_list[-1]
        out_channel_list = vgg_cfg[3]
        self.conv4 = self._make_layers(in_channels, out_channel_list, batch_norm=with_batch_norm)
        in_channels = out_channel_list[-1]
        out_channel_list = vgg_cfg[4]
        self.conv5 = self._make_layers(in_channels, out_channel_list, batch_norm=with_batch_norm)

        self.conv_fc = nn.Sequential(
            nn.Conv2d(512, 4096, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(4096, 4096, kernel_size=1),
            nn.ReLU(inplace=True),
        )

        if init_weights:
            self._initialize_weights()
    
    def _make_layers(self, in_channels, out_channel_list, batch_norm=False):
        layers = []
        for v in out_channel_list:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x) # [n, c, h/2, w/2]
        x = self.conv2(x)
        x = self.maxpool(x) # [n, c, h/4, w/4]
        x = self.conv3(x)
        x_s8 = self.maxpool(x) # [n, c, h/8, w/8]
        x = self.conv4(x_s8)
        x_s16 = self.maxpool(x) # [n, c, h/16, w/16]
        x = self.conv5(x_s16)
        x = self.maxpool(x) # [n, c, h/32, w/32]
        x_s32 = self.conv_fc(x)
        return x_s32, x_s16, x_s8

# # arch = 'vgg11'
# arch = 'vgg11_bn'
# net = NetWithConvFC(arch, pretrained=True)
# # net = NetWithConvFC(arch, pretrained=False)
# # print(net)

# x = torch.rand(1, 3, 384, 384)
# x_s32, x_s16, x_s8 = net(x)
# print(x_s32.shape, x_s16.shape, x_s8.shape)
# print(net.state_dict().keys())
# state_dict = load_state_dict_from_url(model_urls[arch],progress=True)
# print(state_dict.keys())