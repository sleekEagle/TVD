import torch
import torch.nn as nn
from torchvision import models

class ResNetLSTM(nn.Module):
    def __init__(self, num_classes):
        super(ResNetLSTM, self).__init__()
        
        self.resnet = models.resnet50(pretrained=True)
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-2])  
        
        for param in self.resnet.parameters():
            param.requires_grad = False
        
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.rnn = nn.LSTM(input_size=2048, hidden_size=512, num_layers=2, batch_first=True, bidirectional=True)
        
        self.fc = nn.Linear(512 * 2, num_classes) 
        
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        b, t, c, h, w = x.shape
        
        frame_features = []
        for i in range(t):
            frame = x[:, i, :, :, :]  
            features = self.resnet(frame)  
            features = self.pool(features)
            features = features.flatten(1)  
            frame_features.append(features)
        
        x = torch.stack(frame_features, dim=1) 

        x, _ = self.rnn(x) 
        
        x = x[:, -1, :]  
        
        x = self.dropout(x)
        
        x = self.fc(x) 
        return x
