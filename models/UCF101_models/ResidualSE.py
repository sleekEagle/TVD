import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conv3D, self).__init__()
        self.conv = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1), bias=False)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class ResBlock3D(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, stride=1):
        super(ResBlock3D, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, mid_channels, kernel_size=(1, 1, 1), stride=1, bias=False)
        self.bn1 = nn.BatchNorm3d(mid_channels)
        self.conv2 = nn.Conv3d(mid_channels, mid_channels, kernel_size=(3, 3, 3), stride=stride, padding=(1, 1, 1), bias=False)
        self.bn2 = nn.BatchNorm3d(mid_channels)
        self.conv3 = nn.Conv3d(mid_channels, out_channels, kernel_size=(1, 1, 1), stride=1, bias=False)
        self.bn3 = nn.BatchNorm3d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=(1, 1, 1), stride=stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck3D(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, stride=1):
        super(Bottleneck3D, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, mid_channels, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm3d(mid_channels)
        self.conv2 = nn.Conv3d(mid_channels, mid_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(mid_channels)
        self.conv3 = nn.Conv3d(mid_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out += identity
        out = self.relu(out)
        return out

class SEBlock3D(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock3D, self).__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)

    def forward(self, x):
        B, C, T, H, W = x.size()
        y = x.mean(dim=(2, 3, 4))
        y = torch.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y))
        y = y.view(B, C, 1, 1, 1)
        return x * y

class BottleneckWithSE3D(Bottleneck3D):
    def __init__(self, in_channels, mid_channels, out_channels, stride=1):
        super().__init__(in_channels, mid_channels, out_channels, stride)
        self.se = SEBlock3D(out_channels)

    def forward(self, x):
        out = super().forward(x)
        out = self.se(out)
        return out

class Res1(nn.Module):
    def __init__(self):
        super(Res1, self).__init__()
        self.layer = self._make_layer(in_channels=32, mid_channels=72, out_channels=162, num_blocks=5, stride=2)

    def _make_layer(self, in_channels, mid_channels, out_channels, num_blocks, stride):
        layers = []
        for i in range(num_blocks):
            if i == 0:
                layers.append(ResBlock3D(in_channels, mid_channels, out_channels, stride))
            else:
                layers.append(ResBlock3D(out_channels, mid_channels, out_channels, stride=1))
        return nn.Sequential(*layers)
    def forward(self, x):
        return self.layer(x)
class Res2(nn.Module):
    def __init__(self):
        super(Res2, self).__init__()
        self.layer = self._make_layer(in_channels=32, mid_channels=80, out_channels=200, num_blocks=5, stride=2)

    def _make_layer(self, in_channels, mid_channels, out_channels, num_blocks, stride):
        layers = []
        for i in range(num_blocks):
            if i == 0:
                layers.append(ResBlock3D(in_channels, mid_channels, out_channels, stride))
            else:
                layers.append(ResBlock3D(out_channels, mid_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)
class Res3(nn.Module):
    def __init__(self):
        super(Res3, self).__init__()
        self.layer = self._make_layer(in_channels=32, mid_channels=64, out_channels=128, num_blocks=5, stride=2)

    def _make_layer(self, in_channels, mid_channels, out_channels, num_blocks, stride):
        layers = []
        for i in range(num_blocks):
            if i == 0:
                layers.append(ResBlock3D(in_channels, mid_channels, out_channels, stride))
            else:
                layers.append(ResBlock3D(out_channels, mid_channels, out_channels, stride=1))
        return nn.Sequential(*layers)
    def forward(self, x):
        return self.layer(x)
class Res4(nn.Module):
    def __init__(self):
        super(Res4, self).__init__()
        self.layer = self._make_layer(in_channels=32, mid_channels=72, out_channels=162, num_blocks=5, stride=2)

    def _make_layer(self, in_channels, mid_channels, out_channels, num_blocks, stride):
        layers = []
        for i in range(num_blocks):
            if i == 0:
                layers.append(BottleneckWithSE3D(in_channels, mid_channels, out_channels, stride))
            else:
                layers.append(BottleneckWithSE3D(out_channels, mid_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)

class ChannelWise3D(nn.Module):
    def __init__(self, in_channels=630, out_channels=630):
        super(ChannelWise3D, self).__init__()
        self.conv3d = nn.Conv3d(  in_channels=in_channels,out_channels=out_channels,kernel_size=(1, 1, 1), stride=(1, 1, 1),    padding=(0, 0, 0),    bias=False)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv3d(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ResidualSE(nn.Module):
    def __init__(self, num_classes):
        super(ResidualSE, self).__init__()
        self.conv1 = Conv3D(in_channels=3, out_channels=32)
        self.res1 = Res4()
        self.conv2 = ChannelWise3D(in_channels=162, out_channels=256)
        self.globalaveragepooling = nn.AdaptiveAvgPool3d(1)
        self.fc1 = nn.Linear(in_features=256, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=num_classes)

        self.dropout2 = nn.Dropout(p=0.5)

    def forward(self, x):
        x = x.permute(0,2,1,3,4)
        x = self.conv1(x)
        x = self.res1(x)
        x = self.conv2(x)
        x = self.globalaveragepooling(x)
        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x
