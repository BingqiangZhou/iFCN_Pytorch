import torch
import torch.nn as nn

class NetRemoveFCLayer(nn.Module):
    '''
        reference urls:
            https://pytorch.org/vision/stable/_modules/torchvision/models/alexnet.html
    '''

    def __init__(self):
        super(NetRemoveFCLayer, self).__init__()
        
        self.net = AlexNetWithoutFC()
        self.out_channels = [4096, 192, 64]

    def forward(self, x):
        x_s32, x_s16, x_s8 = self.net(x)
        return x_s32, x_s16, x_s8


class AlexNetWithoutFC(nn.Module):
    '''
        reference urls:
            https://pytorch.org/vision/stable/_modules/torchvision/models/alexnet.html
    '''

    def __init__(self):
        super(AlexNetWithoutFC, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)
        self.conv2 = nn.Conv2d(64, 192, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(192, 384, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        
        # convert fc to conv
        self.conv6= nn.Conv2d(256, 4096, kernel_size=1)
        self.conv7 = nn.Conv2d(4096, 4096, kernel_size=1)
        # self.conv8 = nn.Conv2d(4096, num_classes, kernel_size=1)
        
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)

    def forward(self, x):
        x_s4 = self.relu(self.conv1(x))         # [n, 64, h/4, w/4]
        x_s8_ = self.maxpool(x_s4)              # [n, 64, h/8, w/8]  # out
        x_s8 = self.relu(self.conv2(x_s8_))     # [n, 192, h/8, w/8]
        x_s16_ = self.maxpool(x_s8)             # [n, 192, h/16, w/16] # out
        x_s16 = self.relu(self.conv3(x_s16_))   # [n, 384, h/16, w/16]
        x_s16 = self.relu(self.conv4(x_s16))    # [n, 256, h/16, w/16]
        x_s16 = self.relu(self.conv5(x_s16))    # [n, 256, h/16, w/16]
        x_s32 = self.maxpool(x_s16)             # [n, 256, h/32, w/32]
        x_s32 = self.relu(self.conv6(x_s32))    # [n, 4096, h/32, w/32]
        x_s32 = self.relu(self.conv7(x_s32))    # [n, 4096, h/32, w/32] # out
        # x_s32 = self.relu(self.conv8(x_s32))    # [n, nums_classes, h/32, w/32] # out
        return x_s32, x_s16_, x_s8_



