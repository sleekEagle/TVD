import torch
import torch.nn as nn
from torch.nn.modules.utils import _triple
import math

class SpatioTemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(SpatioTemporalConv, self).__init__()
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)

        spatial_kernel_size = [1, kernel_size[1], kernel_size[2]]
        spatial_stride = [1, stride[1], stride[2]]
        spatial_padding = [0, padding[1], padding[2]]

        temporal_kernel_size = [kernel_size[0], 1, 1]
        temporal_stride = [stride[0], 1, 1]
        temporal_padding = [padding[0], 0, 0]

        intermed_channels = int(math.floor((kernel_size[0] * kernel_size[1] * kernel_size[2] * in_channels * out_channels) /
                            (kernel_size[1] * kernel_size[2] * in_channels + kernel_size[0] * out_channels)))

        self.spatial_conv = nn.Conv3d(in_channels, intermed_channels, spatial_kernel_size,
                                       stride=spatial_stride, padding=spatial_padding, bias=bias)
        self.bn = nn.BatchNorm3d(intermed_channels)
        self.relu = nn.ReLU()

        self.temporal_conv = nn.Conv3d(intermed_channels, out_channels, temporal_kernel_size,
                                       stride=temporal_stride, padding=temporal_padding, bias=bias)

    def forward(self, x):
        x = self.relu(self.bn(self.spatial_conv(x)))
        x = self.temporal_conv(x)
        return x

class SpatioTemporalResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, downsample=False):
        super(SpatioTemporalResBlock, self).__init__()
        
        self.downsample = downsample
        padding = kernel_size // 2

        if self.downsample:
            self.downsampleconv = SpatioTemporalConv(in_channels, out_channels, 1, stride=2)
            self.downsamplebn = nn.BatchNorm3d(out_channels)
            self.conv1 = SpatioTemporalConv(in_channels, out_channels, kernel_size, padding=padding, stride=2)
        else:
            self.conv1 = SpatioTemporalConv(in_channels, out_channels, kernel_size, padding=padding)

        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu1 = nn.ReLU()
        self.conv2 = SpatioTemporalConv(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.outrelu = nn.ReLU()

    def forward(self, x):
        res = self.relu1(self.bn1(self.conv1(x)))    
        res = self.bn2(self.conv2(res))

        if self.downsample:
            x = self.downsamplebn(self.downsampleconv(x))

        return self.outrelu(x + res)

class SpatioTemporalResLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, layer_size,
                 block_type=SpatioTemporalResBlock,
                 downsample=False):
        
        super(SpatioTemporalResLayer, self).__init__()

        self.block1 = block_type(in_channels, out_channels, kernel_size, downsample)

        # Prepare module list to hold all (layer_size - 1) blocks
        self.blocks = nn.ModuleList([])
        for i in range(layer_size - 1):
            # All these blocks are identical and have downsample=False by default
            self.blocks += [block_type(out_channels, out_channels, kernel_size)]

    def forward(self, x):
        x = self.block1(x)
        for block in self.blocks:
            x = block(x)

        return x

class R2Plus1DNet(nn.Module):
    def __init__(self, layer_sizes):
        super(R2Plus1DNet, self).__init__()

        # First conv
        self.conv1 = SpatioTemporalConv(3, 32, [3, 7, 7], stride=[1, 2, 2], padding=[1, 3, 3])
        
        # Output of conv2 is same size as of conv1; no downsampling needed.
        self.conv2 = SpatioTemporalResLayer(32, 32, 3,
                                             layer_sizes[0])
        
        # Each of the final three layers doubles num_channels while performing downsampling 
        # inside the first block.
        self.conv3 = SpatioTemporalResLayer(32 ,64 ,3 ,layer_sizes[1], downsample=True)
        
        self.conv4 = SpatioTemporalResLayer(64 ,128 ,3 ,layer_sizes[2], downsample=True)
        
        self.conv5 = SpatioTemporalResLayer(128 ,256 ,3 ,layer_sizes[3], downsample=True)

        # Global average pooling of the output
        self.pool = nn.AdaptiveAvgPool3d(1)

    def forward(self,x):
        x = self.conv1(x)
        
        x = self.conv2(x)
        
        x = self.conv3(x)
        
        x = self.conv4(x)
        
        x = self.conv5(x)

        x = self.pool(x)

        return x

        
# Classifier that produces class probabilities
class R2Plus1DClassifier(nn.Module):
    def __init__(self,num_classes ,layer_sizes):
        
        super(R2Plus1DClassifier,self).__init__()

            
        # Initialize ResNet feature extractor
        self.res2plus1d= R2Plus1DNet(layer_sizes)
        
        # Linear layer for classification
        self.linear= nn.Linear(256,num_classes)
        
    
    def forward(self, x):
        # Permute input tensor from (batch_size, num_frames, C, H, W) to (batch_size, C, num_frames, H, W)
        x = x.permute(0, 2, 1, 3, 4)

        # Pass through ResNet feature extractor
        x = self.res2plus1d(x)
    
        # Flatten the tensor (remove extra dimensions)
        x = x.view(x.size(0), -1)  # Flatten to (batch_size, 512)
    
        # Pass through linear layer
        x = self.linear(x)
    
        return x


def initialize_weights(module):
    if isinstance(module, nn.Conv3d):
        nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)