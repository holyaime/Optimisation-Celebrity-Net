import torch
import torch.nn as nn
import torch.nn.functional as F

class EfficientNetB0(nn.Module):
    def __init__(self, num_classes=6):
        super(EfficientNetB0, self).__init__()
        
        # Couche d'entrée
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.act1 = nn.SiLU()
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Bloc 2
        self.conv2_1 = nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2_1 = nn.BatchNorm2d(16)
        self.act2_1 = nn.SiLU()
        self.conv2_2 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2_2 = nn.BatchNorm2d(16)
        self.act2_2 = nn.SiLU()
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Bloc 3
        self.conv3_1 = nn.Conv2d(16, 24, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn3_1 = nn.BatchNorm2d(24)
        self.act3_1 = nn.SiLU()
        self.conv3_2 = nn.Conv2d(24, 24, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3_2 = nn.BatchNorm2d(24)
        self.act3_2 = nn.SiLU()
        self.conv3_3 = nn.Conv2d(24, 24, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3_3 = nn.BatchNorm2d(24)
        self.act3_3 = nn.SiLU()
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Bloc 4
        self.conv4_1 = nn.Conv2d(24, 40, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn4_1 = nn.BatchNorm2d(40)
        self.act4_1 = nn.SiLU()
        self.conv4_2 = nn.Conv2d(40, 40, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4_2 = nn.BatchNorm2d(40)
        self.act4_2 = nn.SiLU()
        self.conv4_3 = nn.Conv2d(40, 40, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4_3 = nn.BatchNorm2d(40)
        self.act4_3 = nn.SiLU()
        self.pool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Bloc 5
        self.conv5_1 = nn.Conv2d(40, 80, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn5_1 = nn.BatchNorm2d(80)
        self.act5_1 = nn.SiLU()
        self.conv5_2 = nn.Conv2d(80, 80, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn5_2 = nn.BatchNorm2d(80)
        self.act5_2 = nn.SiLU()
        self.conv5_3 = nn.Conv2d(80, 80, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn5_3 = nn.BatchNorm2d(80)
        self.act5_3 = nn.SiLU()
        self.pool5 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Couches finales
        self.dropout1 = nn.Dropout(0.2)
        self.fc = nn.Linear(1920, num_classes)
        self.dropout2 = nn.Dropout(0.3)
        self.weight_decay = 1e-5
        
    def forward(self, x):
        # Couche d'entrée
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.pool1(x)
        
        # Bloc 2
        x = self.conv2_1(x)
        x = self.bn2_1(x)
        x = self.act2_1(x)
        x = self.conv2_2(x)
        x = self.bn2_2(x)
        x = self.act2_2(x)
        x = self.pool2(x)
        
        # Bloc 3
        x = self.conv3_1(x)
        x = self.bn3_1(x)
        x = self.act3_1(x)
        x = self.conv3_2(x)
        x = self.bn3_2(x)
        x = self.act3_2(x)
        x = self.conv3_3(x)
        x = self.bn3_3(x)
        x = self.act3_3(x)
        x = self.pool3(x)
        
        # Bloc 4
        x = self.conv4_1(x)
        x = self.bn4_1(x)
        x = self.act4_1(x)
        x = self.conv4_2(x)
        x = self.bn4_2(x)
        x = self.act4_2(x)
        x = self.conv4_3(x)
        x = self.bn4_3(x)
        x = self.act4_3(x)
        x = self.pool4(x)
        
        # Bloc 5
        x = self.conv5_1(x)
        x = self.bn5_1(x)
        x = self.act5_1(x)
        x = self.conv5_2(x)
        x = self.bn5_2(x)
        x = self.act5_2(x)
        x = self.conv5_3(x)
        x = self.bn5_3(x)
        x = self.act5_3(x)
        x = self.pool5(x)
        
        # Couches finales
        x = x.view(x.size(0), -1)
        x = self.dropout1(x)
        x = self.fc(x)
        x = self.dropout2(x)
        
        return x

