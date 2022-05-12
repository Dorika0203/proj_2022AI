import torch.nn as nn
# from torchdiffeq import odeint_adjoint as odeint
from torchdiffeq import odeint
import torch

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
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out




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
        o = self.conv(x)
        o = self.maxp(o)
        o = self.block1(o)
        o = self.block2(o)
        o = self.block3(o)
        o = self.block4(o)
        o = self.avgp(o)
        o = o.view(o.size(0), -1)
        o = self.fc(o)
        return o


















class ConcatConv2d(nn.Module):

    def __init__(self, dim_in, dim_out, ksize=3, stride=1, padding=0, dilation=1, groups=1, bias=True, transpose=False):
        super(ConcatConv2d, self).__init__()
        module = nn.ConvTranspose2d if transpose else nn.Conv2d
        self._layer = module(
            dim_in + 1, dim_out, kernel_size=ksize, stride=stride, padding=padding, dilation=dilation, groups=groups,
            bias=bias
        )
    def forward(self, t, x):
        tt = torch.ones_like(x[:, :1, :, :]) * t
        ttx = torch.cat([tt, x], 1)
        return self._layer(ttx)


class OdeFunction1(nn.Module):

    def __init__(self, channel):
        super(OdeFunction1, self).__init__()
        self.conv = ConcatConv2d(channel,channel,3,1,1)
        self.bn = nn.BatchNorm2d(channel)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, t, x):
        out = self.conv(t, x)
        out = self.bn(x)
        out = self.relu(x)
        return out

class OdeFunction2(nn.Module):

    def __init__(self, channel):
        super(OdeFunction2, self).__init__()
        self.conv1 = ConcatConv2d(channel,channel,3,1,1)
        self.bn1 = nn.BatchNorm2d(channel)
        self.relu = nn.ReLU(inplace=False)

        self.conv2 = ConcatConv2d(channel, channel, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(channel)
        self.nfe = 0

    def forward(self, t, x):
        out = self.conv1(t,x)
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv2(t, out)
        out = self.bn2(out)
        return out

class OdeBlock(nn.Module):

    def __init__(self, odefunc, n_step=0):
        super(OdeBlock, self).__init__()
        self.odefunc = odefunc
        self.integration_time = torch.tensor([0, 1]).float()
        self.n_step = 0
        self.option = dict()
        if n_step != 0:
            self.n_step = (self.integration_time[1]-self.integration_time[0])/n_step
            self.option.update({'step_size': self.n_step})

    def forward(self, x):
        self.integration_time = self.integration_time.type_as(x)
        if self.n_step != 0:
            out = odeint(
                self.odefunc, x, self.integration_time, rtol=1e-3, atol=1e-3,
                method='euler', options=self.option
                )
        else:
            out = odeint(self.odefunc, x, self.integration_time, rtol=1e-3, atol=1e-3)
        return out[1]








"""
Implementation of Model 1.
Copying the exact design of ResNet-18
To use exace amount of convolutional layer, used 'Euler' method for ODE function.
specified step number, to decide amount of ODE function in ODE block.
"""
class Model_1(nn.Module):
    def __init__(self, in_channels, heigth=224, width=224, labelNum=1000):
        super(Model_1,self).__init__()

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxp = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.block1 = nn.Sequential(
            OdeBlock(OdeFunction1(channel=64), n_step=4)
        )
        self.block2 = nn.Sequential(
            ResidualBlock(in_channels=64, out_channels=128, stride=2),
            OdeBlock(OdeFunction1(channel=128), n_step=2)
        )
        self.block3 = nn.Sequential(
            ResidualBlock(in_channels=128, out_channels=256, stride=2),
            OdeBlock(OdeFunction1(channel=256), n_step=2)
        )
        self.block4 = nn.Sequential(
            ResidualBlock(in_channels=256, out_channels=512, stride=2),
            OdeBlock(OdeFunction1(channel=512), n_step=2)
        )
        self.avgp = nn.AvgPool2d(kernel_size=(width//32, heigth//32))
        self.fc = nn.Linear(in_features=512, out_features=labelNum)
    
    def forward(self, x):

        o = self.conv(x)
        o = self.bn(o)
        o = self.relu(o)
        o = self.maxp(o)

        o = self.block1(o)
        o = self.block2(o)
        o = self.block3(o)
        o = self.block4(o)

        o = self.avgp(o)
        o = o.view(o.size(0), -1)
        o = self.fc(o)
        return o


"""
Implementation of Model 2.
Modified order of down sampling ResBlocks and same-size ResBlocks.
All down sampling ResBlocks are at front, and all same-size ResBlocks are at back.
ResBlocks at back are changed to ODE block, without step number assigned.
The Last ODE block uses default 'Dopri5' to obtain ordinary differential equation.
"""
class Model_2(nn.Module):

    def __init__(self, in_channels, heigth=224, width=224, labelNum=1000):
        super(Model_2,self).__init__()

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxp = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.downsample = nn.Sequential(
            ResidualBlock(in_channels=64, out_channels=128, stride=2),
            ResidualBlock(in_channels=128, out_channels=256, stride=2),
            ResidualBlock(in_channels=256, out_channels=512, stride=2),
        )
        self.odeBlock = OdeBlock(OdeFunction2(channel=512))
        self.avgp = nn.AvgPool2d(kernel_size=(width//32, heigth//32))
        self.fc = nn.Linear(in_features=512, out_features=labelNum)
    
    def forward(self, x):

        o = self.conv(x)
        o = self.bn(o)
        o = self.relu(o)
        o = self.maxp(o)

        o = self.downsample(o)
        o = self.odeBlock(o)

        o = self.avgp(o)
        o = o.view(o.size(0), -1)
        o = self.fc(o)
        return o