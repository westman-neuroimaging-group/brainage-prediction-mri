#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Residual Attention Network module - feature extractor in AVRA
"""
import torch.nn as nn

class ResidualNet3D(nn.Module):
    def __init__(self, z=1,width_f=1):
        super(ResidualNet3D, self).__init__()
        
        # number of output filters from each block
        num_filters = [8,16,32,64,128]
        num_filters = [int(width_f*f) for f in num_filters] # factor to generate a wider network
        k=0 # block number counter
        
        conv1 = nn.Sequential(
            nn.Conv3d(z, num_filters[k], kernel_size=7, stride=2, padding=3, bias = False),
            nn.BatchNorm3d(num_filters[k]),
            nn.ReLU(inplace=True)
        )
        maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        resblock1 = ResidualModule3D(num_filters[k], num_filters[k+1])
        k+=1

        resblock2 = ResidualModule3D(num_filters[k], num_filters[k+1], stride=2)
        k+=1

        resblock3 = ResidualModule3D(num_filters[k], num_filters[k+1], stride=2)
        k+=1
        
        resblock4 = ResidualModule3D(num_filters[k], num_filters[k+1], stride=2)
        
        k+=1
        resblock5= ResidualModule3D(num_filters[k], num_filters[k])
        resblock6 = ResidualModule3D(num_filters[k], num_filters[k])
        avgpoolblock = nn.Sequential(
            nn.BatchNorm3d(num_filters[k]),
            nn.ReLU(inplace=True),
            nn.AvgPool3d(kernel_size=3, stride=1)
        )

        self.features=nn.Sequential(
                conv1,maxpool,
                resblock1,
                resblock2,
                resblock3,
                resblock4,resblock5,resblock6,
                avgpoolblock
                      )
        
    def forward(self, x):
        out = self.features(x)
        
        # Flatten before passing to fully connected network 
        out = out.view(out.size(0), -1)        
        return out


class ResidualModule3D(nn.Module):
    '''
    A residual module, used in the Residual Attention Network.
    '''
    def __init__(self, inplanes, planes, stride=1):
        super(ResidualModule3D, self).__init__()
        
        planes_4 = int(planes/4) # bottlenecking
        self.inplanes = inplanes
        self.planes = planes
        self.stride = stride
        
        self.bn1 = nn.BatchNorm3d(inplanes)
        self.relu1 = nn.LeakyReLU()
        self.conv1 = nn.Conv3d(inplanes,planes_4, kernel_size=1, stride=1, bias = False)
        
        self.bn2 = nn.BatchNorm3d(planes_4)
        self.relu2 = nn.LeakyReLU()
        self.conv2 = nn.Conv3d(planes_4, planes_4, kernel_size=3, stride=stride, padding = 1, bias = False)
        
        self.bn3 = nn.BatchNorm3d(planes_4)
        self.relu3 = nn.LeakyReLU()
        self.conv3 = nn.Conv3d(planes_4, planes, kernel_size=1, stride=1, bias = False)
        
        self.conv4 = nn.Conv3d(inplanes, planes, kernel_size=1, stride=stride, bias = False)
        # downsampling?
        self.downsample = (self.inplanes != self.planes) or (self.stride !=1 )
    def forward(self, x):

        residual = x
        out = self.bn1(x)
        out1 = self.relu1(out)
        out = self.conv1(out1)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu3(out)
        out = self.conv3(out)
        if self.downsample:
            residual = self.conv4(out1)
        out += residual
        return out
    
