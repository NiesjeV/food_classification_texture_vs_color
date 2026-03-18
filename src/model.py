import torch
import torch.nn as nn
import torch.nn.functional as F

#Dit is de model architectuur

class SimpleCNN(nn.Module):
    """Eenvoudig CNN voor food classificatie"""
    
    def __init__(self, num_classes=80):
        super().__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        
        #poolen
        self.pool = nn.MaxPool2d(2, 2)
        
        # Na 2x poolen: 224x224 -> 56x56
        # 32 filters * 56 * 56 = 100352
        self.fc1 = nn.Linear(32 * 56 * 56, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
        #dropout
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x): #haal data door het model
        # dus eerst Convoluties + pooling
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        
        # daarna Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected: beslissen bijwelk lable het plaatje hoort
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x