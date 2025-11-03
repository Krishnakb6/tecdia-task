
import torch
import torch.nn as nn
import torch.nn.functional as F

class MotionValidatorCNN(nn.Module):
    def __init__(self, input_height=128, input_width=128):
        super(MotionValidatorCNN, self).__init__()
        
        
      
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
       

      
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
       
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        
        
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        flattened_size = 64 * 16 * 16

        
        self.fc1 = nn.Linear(flattened_size, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 1) 
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        
        
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        
        x = x.view(x.size(0), -1) 
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return self.sigmoid(x)