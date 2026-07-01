import torch
import torch.nn as nn
import torchvision.models.video as video_models

class R3DModel(nn.Module):
    """R3D model for video classification."""
    def __init__(self, num_classes: int, pretrained: bool = True, dropout_prob: float = 0.5):
        super(R3DModel, self).__init__()
        self.model = video_models.r3d_18(weights='KINETICS400_V1' if pretrained else None)
        
        self.model.fc = nn.Sequential(
            nn.Dropout(dropout_prob),
            nn.Linear(self.model.fc.in_features, num_classes)
        )

    def forward(self, x):
        x = x.permute(0,2,1,3,4)
        return self.model(x)
    
class AttentionBlock(nn.Module):
    """Spatial-Temporal Attention Mechanism"""
    def __init__(self, in_channels):
        super().__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(in_channels, in_channels // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels // 4, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        self.spatial_attention = nn.Sequential(
            nn.Conv3d(in_channels, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        channel_attn = self.channel_attention(x)
        spatial_attn = self.spatial_attention(x)
        return x * channel_attn * spatial_attn

class EnhancedR3DModel(nn.Module):
    """Update R3D model with dynamic feature handling"""
    def __init__(
        self, 
        num_classes: int, 
        pretrained: bool = True, 
        dropout_prob: float = 0.5
    ):
        super().__init__()
        
        # Base model initialization
        self.base_model = video_models.r3d_18(weights='KINETICS400_V1' if pretrained else None)
        
        # Remove the original fully connected layer
        self.features = nn.Sequential(*list(self.base_model.children())[:-1])
        
        # Create a feature dimension calculator
        self._calculate_feature_dimensions()
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_prob),
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_prob / 2),
            nn.Linear(512, num_classes)
        )

    def _calculate_feature_dimensions(self):
        """Dynamically calculate feature dimensions"""
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 16, 224, 224)
            features = self.features(dummy_input)
            
            # Flatten the features
            self.feature_dim = features.view(features.size(0), -1).size(1)

    def forward(
        self, 
        x: torch.Tensor
    ) -> torch.Tensor:
        # Extract features
        features = self.features(x)
        
        # Flatten features
        features = features.view(features.size(0), -1)
        
        # Classification
        output = self.classifier(features)
        
        return output    