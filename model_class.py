import torch
import torch.nn as nn
import torch.nn.functional as F

class SepsisCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(SepsisCNN, self).__init__()
        
        # Match the architecture used during training
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        
        # The number 1548800 must match your trained model
        self.fc1 = nn.Linear(1548800, 64)
        self.fc2 = nn.Linear(64, num_classes)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
