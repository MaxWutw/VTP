import math
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from Transformer_Encoder import TransformerEncoder
import copy
from transformer import Transformer

class ConvFrontend(nn.Module):
    """
    Convolutional frontend
    nn.Conv2d : torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
    nn.Conv3d : torch.nn.Conv3d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
    """
    def __init__(self):
        super(ConvFrontend, self).__init__()
        self.frontend3D = nn.Sequential(
            ## conv1
            nn.Conv3d(3, 64, kernel_size=(5,5,5), stride=(1,2,2), padding=(2,2,2)),
            nn.BatchNorm3d(64),
            nn.ReLU()
        )
        self.frontend2D = nn.Sequential(
            ## conv2
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2,2), padding=(1,1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1,1), padding=(1,1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1,1), padding=(1,1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # gray region
            ## conv3
            # nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(2,2), padding=(1,1)),
            # nn.BatchNorm2d(256),
            # nn.ReLU(),
            # nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1,1), padding=(1,1)),
            # nn.BatchNorm2d(256),
            # nn.ReLU(),
            # nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1,1), padding=(1,1)),
            # nn.BatchNorm2d(256),
            # nn.ReLU(),
            # ## conv4
            # nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(2,2), padding=(1,1)),
            # nn.BatchNorm2d(512),
            # nn.ReLU(),
            # nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1,1), padding=(1,1)),
            # nn.BatchNorm2d(512),
            # nn.ReLU(),
            # nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1,1), padding=(1,1)),
            # nn.BatchNorm2d(512),
            # nn.ReLU(),
            # ## conv5
            # nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(2,2), padding=(1,1)),
            ## fc
            # nn.Linear(512, 512)
        )
    def forward(self, x):
        x = self.frontend3D(x)

        x = x.transpose(1, 2)
        x = x.contiguous()
        x = x.view(-1, 64, x.size(3), x.size(4))

        x = self.frontend2D(x)

        return x

class VTPBlock(nn.Module):
    def __init__(self):
        super(VTPBlock, self).__init__()
        self.transformer_encoder = TransformerEncoder(512, 512, 1, 4, 0.1)

    def forward(self, x):
        x = self.transformer_encoder(x)
        return x

def get_clones(module, N):
    # print('hello')
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class VTP(nn.Module):
    def __init__(self, with_vtp=False):
        super(VTP, self).__init__()
        # Frontend CNN
        self.with_vtp = with_vtp
        self.frontend = ConvFrontend()
        if self.with_vtp:
            self.vtpblock = get_clones(VTPBlock(), 4)
        self.transformer = Transformer(2000, 512, 2, 4, 0.1)
        self.fc1 = nn.Linear(5529600, 512)
        
    def forward(self, x, txt):
        x = self.frontend(x)
        x = x.view(1, -1)
        x = self.fc1(x)
        # if self.with_vtp:
        #     for i in range(4):
        #         x = self.vtpblock[i](x)
        print(x.shape) # torch.Size([1, 200, 512])
        print(txt[0])
        x = self.transformer(x, txt[0])
        print(x.shape)
        # print('hello')

        return x
