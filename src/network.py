import os
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import init_weights



class ResBlock(nn.Module):
    def __init__(self, n_features=256, kernel_size=3, stride=1, batch_norm=True, activation=nn.ReLU()):
        super().__init__()
        m = []
        for i in range(2):
            m.append(nn.Conv2d(n_features,n_features,kernel_size,stride,padding=1))
            if batch_norm:
                m.append(nn.BatchNorm2d(n_features))
            if i == 0 and activation is not None:
                m.append(activation)
        self.body = nn.Sequential(*m)

    def forward(self,x):
        res = self.body(x) + x
        return res

class Generator(nn.Module):

    def __init__(self, n_resblocks,init_w=True):
        super().__init__()

        self.head = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=64,kernel_size=7,stride=1,padding=3),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU()
        )
        self.down_convolution = nn.Sequential(
            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,stride=2,padding=1),
            nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,stride=2,padding=1),
            nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU()
        )

        body = []
        for _ in range(n_resblocks):
            body.append(ResBlock())
        self.body = nn.Sequential(*body)

        self.upsampler = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256,out_channels=128,kernel_size=3,stride=2, padding=1, output_padding=1),
            nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=128,out_channels=64,kernel_size=3,stride=2, padding=1, output_padding=1),
            nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64,out_channels=3,kernel_size=7,stride=1,padding=3)
        )
    
        if init_w:
            init_weights(self)
            

    def forward(self,input):
        x = self.head(input)
        x = self.down_convolution(x)
        x = self.body(x)
        output = self.upsampler(x)
        output = (torch.tanh(output) + 1) / 2
        return(output)

class Discriminator(nn.Module):

    def __init__(self,use_sigmoid=True):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=32,kernel_size=3,stride=1,padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,stride=2,padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3,stride=2,padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.LeakyReLU()
        )

        self.out = nn.Conv2d(in_channels=256,out_channels=1,kernel_size=3,stride=1,padding=1)
        self.use_sigmoid = use_sigmoid
    
    def forward(self,input):
        x = self.body(input)
        output = self.out(x)
        if self.use_sigmoid:
            output = torch.sigmoid(output)
        return(output)