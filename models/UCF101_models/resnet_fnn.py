from torchvision import models
import torch.nn as nn
import torch

class ResNetFNN(nn.Module):
    def __init__(self, num_classes):
        super(ResNetFNN, self).__init__()
        self.efficientnet = models.resnet50(pretrained=True)
        self.efficientnet = nn.Sequential(*list(self.efficientnet.children())[:-1])
        
        # Freeze the pre-trained parameters for all pre-trained layers accept for the last layer
        for param in self.efficientnet.parameters():
            param.requires_grad = False
            
        # Modified classifier (the final layer) structure
        self.classifier = nn.Sequential(
            nn.Linear(2048 * 10, 1024),  
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024,512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        b, t, c, h, w = x.shape
        
        # Process each frame through EfficientNet
        frame_features = []
        for i in range(t):
            frame = x[:, i, :, :, :]
            features = self.efficientnet(frame)
            frame_features.append(features.squeeze(-1).squeeze(-1))
        
        # Concatenate features from all frames
        x = torch.cat(frame_features, dim=1) 
        x = self.classifier(x)
        return x