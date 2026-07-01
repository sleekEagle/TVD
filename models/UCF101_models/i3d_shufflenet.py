from typing import List, Tuple, Dict, Optional, Union
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import shufflenet_v2_x1_0
from torchvision.models.video import r3d_18, R3D_18_Weights 

import math

class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block."""
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Linear(channel, channel // reduction, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(channel // reduction, channel, bias=True)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        b, c, t, h, w = x.size()
        # Squeeze: Global Average Pooling
        y = F.adaptive_avg_pool3d(x, 1).view(b, c)
        # Excitation: FC -> ReLU -> FC -> Sigmoid
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y).view(b, c, 1, 1, 1)
        # Scale: Channel-wise multiplication
        return x * y.expand_as(x)

class TemporalAttention(nn.Module):
    """Temporal Attention Module."""
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y.expand_as(x)

class SpatialAttention(nn.Module):
    """Spatial Attention Module."""
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv3d(channels, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        attn = self.sigmoid(self.conv(x))  # [B, 1, T, H, W]
        return x * attn

class HybridAttention(nn.Module):
    """Hybrid Attention combining Temporal and Spatial Attention."""
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.temporal_attn = TemporalAttention(channels, reduction)
        self.spatial_attn = SpatialAttention(channels)
    
    def forward(self, x):
        x = self.temporal_attn(x)
        x = self.spatial_attn(x)
        return x

class TransformerTemporalEncoder(nn.Module):
    """Transformer-based Temporal Encoder."""
    def __init__(self, embed_dim: int, num_heads: int, num_layers: int, dropout: float = 0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x):
        # x: [B, T, C]
        x = self.transformer_encoder(x)  # [B, T, C]
        x = self.norm(x)
        return x

class ChannelShuffle(nn.Module):
    def __init__(self, groups: int):
        super().__init__()
        self.groups = groups

    def forward(self, x):
        batch, channels, time, height, width = x.size()
        channels_per_group = channels // self.groups
        x = x.view(batch, self.groups, channels_per_group, time, height, width)
        x = torch.transpose(x, 1, 2).contiguous()
        x = x.view(batch, channels, time, height, width)
        return x

class TemporalShuffleBlock(nn.Module):
    def __init__(self, channels: int, temporal_stride: int = 1):
        super().__init__()
        self.temporal_conv = nn.Conv3d(
            channels, channels, 
            kernel_size=(3, 1, 1),
            stride=(temporal_stride, 1, 1),
            padding=(1, 0, 0),
            groups=channels
        )
        self.bn = nn.GroupNorm(num_groups=32, num_channels=channels)
        self.shuffle = ChannelShuffle(groups=4)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.relu(self.bn(self.temporal_conv(x)))
        x = self.shuffle(x)
        return x

class AttentionBasedFusion(nn.Module):
    """Attention-Based Feature Fusion Module."""
    def __init__(self, i3d_channels: int, shufflenet_channels: int):
        super().__init__()
        self.query = nn.Linear(i3d_channels, i3d_channels)
        self.key = nn.Linear(shufflenet_channels, shufflenet_channels)
        self.value = nn.Linear(shufflenet_channels, i3d_channels)
        self.softmax = nn.Softmax(dim=-1)
        self.fc = nn.Linear(i3d_channels + shufflenet_channels, i3d_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, i3d_feat, shufflenet_feat):
        # i3d_feat: [B, C_i3d, T, H, W]
        # shufflenet_feat: [B, C_shuff, T, H, W]
        
        # Global Average Pooling
        i3d_pooled = F.adaptive_avg_pool3d(i3d_feat, 1).view(i3d_feat.size(0), -1)  # [B, C_i3d]
        shuff_pooled = F.adaptive_avg_pool3d(shufflenet_feat, 1).view(shufflenet_feat.size(0), -1)  # [B, C_shuff]
        
        # Compute attention scores
        Q = self.query(i3d_pooled)  # [B, C_i3d]
        K = self.key(shuff_pooled)   # [B, C_shuff]
        scores = torch.matmul(Q, K.transpose(0, 1)) / math.sqrt(Q.size(-1))  # [B, B]
        attn_weights = self.softmax(scores)  # [B, B]
        
        # Apply attention
        V = self.value(shuff_pooled)  # [B, C_i3d]
        attn_output = torch.matmul(attn_weights, V)  # [B, C_i3d]
        
        # Fuse features
        fused = torch.cat((i3d_pooled, attn_output), dim=1)  # [B, C_i3d + C_i3d]
        fused = self.relu(self.fc(fused))  # [B, C_i3d]
        
        # Reshape to [B, C_i3d, 1, 1, 1] and expand
        fused = fused.view(fused.size(0), fused.size(1), 1, 1, 1)
        fused = fused.expand_as(i3d_feat)  # [B, C_i3d, T, H, W]
        
        # Add residual connection
        fused = fused + i3d_feat  # [B, C_i3d, T, H, W]
        
        return fused

class EnhancedI3DShuffleNet(nn.Module):
    """
    Enhanced hybrid architecture combining I3D and ShuffleNet concepts
    with advanced features for video classification.
    """
    def __init__(
        self,
        num_classes: int,
        pretrained: bool = True,
        dropout_prob: float = 0.5,
        temporal_module: str = 'transformer',  # 'i3d', 'shuffle', or 'transformer'
        use_attention: bool = True,
        aux_loss: bool = False,
        transformer_layers: int = 2,
        transformer_heads: int = 4
    ):
        super().__init__()
        
        # I3D backbone initialization
        if pretrained:
            weights = R3D_18_Weights.DEFAULT
            self.i3d_backbone = r3d_18(weights=weights)
        else:
            self.i3d_backbone = r3d_18(weights=None)
            
        # ShuffleNet backbone initialization
        self.shuffle_backbone = shufflenet_v2_x1_0(pretrained=pretrained)
        
        # Get feature dimensions
        self.i3d_features = self.i3d_backbone.fc.in_features  # Typically 512 for r3d_18
        self.shuffle_features = self.shuffle_backbone.fc.in_features  # Typically 1024 for shufflenet_v2_x1_0
        
        # Remove original fully connected layers
        self.i3d_backbone.fc = nn.Identity()
        self.shuffle_backbone.fc = nn.Identity()
        
        # Temporal modeling
        self.temporal_module = temporal_module
        if temporal_module == 'shuffle':
            self.temporal_blocks = nn.ModuleList([
                TemporalShuffleBlock(self.i3d_features // 2),
                TemporalShuffleBlock(self.i3d_features // 2)
            ])
        elif temporal_module == 'transformer':
            self.temporal_encoder = TransformerTemporalEncoder(
                embed_dim=self.i3d_features,
                num_heads=transformer_heads,
                num_layers=transformer_layers,
                dropout=0.1
            )
        else:
            raise ValueError(f"Unsupported temporal_module: {temporal_module}")
        
        # Attention mechanism
        self.use_attention = use_attention
        if use_attention:
            self.attention = HybridAttention(self.i3d_features)
        
        # Feature fusion
        self.fusion = AttentionBasedFusion(self.i3d_features, self.shuffle_features)
        
        # Enhanced classifier head with residual connections and GroupNorm
        self.classifier = nn.Sequential(
            nn.Dropout3d(dropout_prob),
            nn.Conv3d(self.i3d_features, self.i3d_features // 2, kernel_size=1),
            nn.GroupNorm(num_groups=16, num_channels=self.i3d_features // 2),
            nn.ReLU(inplace=True),
            nn.Dropout3d(dropout_prob),
            nn.Conv3d(self.i3d_features // 2, num_classes, kernel_size=1)
        )
        
        # Auxiliary classifier for deep supervision
        self.aux_loss = aux_loss
        if aux_loss:
            self.aux_classifier = nn.Sequential(
                nn.AdaptiveAvgPool3d(1),
                nn.Flatten(),
                nn.Linear(self.i3d_features, self.i3d_features // 2),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_prob),
                nn.Linear(self.i3d_features // 2, num_classes)
            )
        
    def _process_shuffle_features(self, x: torch.Tensor) -> torch.Tensor:
        b, c, t, h, w = x.shape
        x = x.transpose(1, 2).contiguous()  # [B, T, C, H, W]
        x = x.view(-1, c, h, w)  # [B*T, C, H, W]
        x = self.shuffle_backbone(x)  # [B*T, C_shuff]
        x = x.view(b, t, -1, 1, 1)  # [B, T, C_shuff, 1, 1]
        x = x.transpose(1, 2).contiguous()  # [B, C_shuff, T, 1, 1]
        return x.expand(-1, -1, t, h, w)  # [B, C_shuff, T, H, W]
    
    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # I3D feature extraction
        x = x.permute(0,2,1,3,4)
        i3d_features = self.i3d_backbone(x)  # [B, C_i3d, T, H, W]
        
        # ShuffleNet feature extraction
        shuffle_features = self._process_shuffle_features(x)  # [B, C_shuff, T, H, W]
        
        # Temporal modeling
        if self.temporal_module == 'shuffle':
            split_size = i3d_features.size(1) // 2
            x1, x2 = torch.split(i3d_features, split_size, dim=1)
            x1 = self.temporal_blocks[0](x1)
            x2 = self.temporal_blocks[1](x2)
            i3d_features = torch.cat([x1, x2], dim=1)
        elif self.temporal_module == 'transformer':
            b, c, t, h, w = i3d_features.size()
            i3d_pooled = F.adaptive_avg_pool3d(i3d_features, (1, h, w)).view(b, c, t)  # [B, C, T]
            i3d_pooled = i3d_pooled.permute(0, 2, 1)  # [B, T, C]
            i3d_pooled = self.temporal_encoder(i3d_pooled)  # [B, T, C]
            i3d_pooled = i3d_pooled.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)  # [B, C, T, 1, 1]
            i3d_features = i3d_features + i3d_pooled  # Residual connection
        
        # Apply attention if enabled
        if self.use_attention:
            i3d_features = self.attention(i3d_features)
        
        # Feature fusion using Attention-Based Fusion
        fused_features = self.fusion(i3d_features, shuffle_features)  # [B, C_i3d, T, H, W]
        
        # Classification
        output = self.classifier(fused_features)  # [B, num_classes, T, H, W]
        output = F.adaptive_avg_pool3d(output, 1).view(output.size(0), -1)  # [B, num_classes]
        
        if self.aux_loss and self.training:
            aux_output = self.aux_classifier(i3d_features)  # [B, num_classes]
            return output, aux_output
                
        return output

# def create_model(config) -> nn.Module:
#     """Factory method to create appropriate video classification model."""
#     num_classes = len(config.classes)

#     if config.model_type == 'i3dshufflenet':
#         model = EnhancedI3DShuffleNet(
#             num_classes=num_classes,
#             pretrained=config.pretrained,
#             dropout_prob=0.5,
#             temporal_module=config.temporal_module,  # 'i3d', 'shuffle', or 'transformer'
#             use_attention=config.use_attention,
#             aux_loss=config.aux_loss,
#             transformer_layers=config.transformer_layers,
#             transformer_heads=config.transformer_heads
#         )
#     else:
#         raise ValueError(f"Unsupported model type: {config.model_type}")

#     model = model.to(config.device)
#     return model