import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import sys
from torch.utils.data import DataLoader, Dataset

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()

        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = None
        if stride != 1:
            self.downsample = nn.Sequential(
                conv1x1(in_planes=in_channels, out_planes=out_channels, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        
    def forward(self, x):
        residual = x
        # print("1 ", o.shape)
        out = self.conv1(x)
        # print("2 ", out.shape)
        out = self.bn1(out)
        # print("3 ", out.shape)
        out = self.relu(out)
        # print("4 ", out.shape)
        out = self.conv2(out)
        # print("5 ", out.shape)
        out = self.bn2(out)
        # print("6 ", out.shape)
        if self.downsample is not None:
            residual = self.downsample(x)
            # print("DOWNSAMPLE ", out.shape)
        out += residual
        out = self.relu(out)
        return out

"""
image의 높이, 넓이가 둘 다 32의 배수여야 함. 그 외의 경우는 해봐야 알 것 같음
제일 바람직한 상황은 이미지가 224 x 224인 경우. (ImageNet)
정사각형 아니고, 둘다 32의 배수기만 해도 돌아감
"""
class ResNet18(nn.Module):
    def __init__(self, in_channels, heigth=224, width=224, labelNum=1000):
        super(ResNet18, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxp = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.block1 = nn.Sequential(
            ResidualBlock(in_channels=64, out_channels=64),
            ResidualBlock(in_channels=64, out_channels=64)
        )
        self.block2 = nn.Sequential(
            ResidualBlock(in_channels=64, out_channels=128, stride=2),
            ResidualBlock(in_channels=128, out_channels=128)
        )
        self.block3 = nn.Sequential(
            ResidualBlock(in_channels=128, out_channels=256, stride=2),
            ResidualBlock(in_channels=256, out_channels=256)
        )
        self.block4 = nn.Sequential(
            ResidualBlock(in_channels=256, out_channels=512, stride=2),
            ResidualBlock(in_channels=512, out_channels=512)
        )
        self.avgp = nn.AvgPool2d(kernel_size=(width//32, heigth//32))
        self.fc = nn.Linear(in_features=512, out_features=labelNum)

    
    def forward(self, x):
        # print('0 ', x.shape)
        o = self.conv(x)
        # print('1 ', o.shape)
        o = self.maxp(o)
        # print('2 ', o.shape)
        o = self.block1(o)
        # print('3 ', o.shape)
        o = self.block2(o)
        # print('4 ', o.shape)
        o = self.block3(o)
        # print('5 ', o.shape)
        o = self.block4(o)
        # print('6 ', o.shape)
        o = self.avgp(o)
        # print('7 ', o.shape)
        o = o.view(o.size(0), -1)
        # print('8 ', o.shape)
        o = self.fc(o)
        return o