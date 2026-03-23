import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================
# Squeeze-and-Excitation Block
# ============================================
class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block.
    Learns which channels are important and reweights them.
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),            # [B, C, 1, 1]
            nn.Flatten(),                       # [B, C]
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        weight = self.fc(x)                     # [B, C]
        return x * weight.view(x.size(0), x.size(1), 1, 1)

# ============================================
# FoodCNN with SE blocks
# ============================================
class FoodCNN(nn.Module):
    """
    CNN for food classification with Squeeze-and-Excitation blocks.
    """
    def __init__(self, num_classes=80, dropout_rate=0.3, use_se=True):
        super().__init__()
        
        self.use_se = use_se
        
        # Block 1: 3 -> 32
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        if use_se:
            self.se1 = SEBlock(32)
        
        # Block 2: 32 -> 64
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        if use_se:
            self.se2 = SEBlock(64)
        
        # Block 3: 64 -> 128
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        if use_se:
            self.se3 = SEBlock(128)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout2d(dropout_rate)
        
        # After 3x pooling with 256x256 input: 256 -> 128 -> 64 -> 32
        self.feature_size = 128 * 32 * 32
        self.fc1 = nn.Linear(self.feature_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        
        self.fc_dropout = nn.Dropout(dropout_rate)
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)         # 256 -> 128
        if self.use_se:
            x = self.se1(x)
        
        # Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)         # 128 -> 64
        if self.use_se:
            x = self.se2(x)
        
        # Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool(x)         # 64 -> 32
        if self.use_se:
            x = self.se3(x)
        
        # FC
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc_dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc_dropout(x)
        x = self.fc3(x)
        return x

def accuracy(predictions, targets):
    preds = predictions.argmax(dim=1)
    correct = (preds == targets).float()
    return correct.mean().item()
