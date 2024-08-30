import torch.nn as nn
import torch.nn.functional as F
import torch
class SimplifiedMBConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, expand_ratio=1, dropout_rate=0.5):
        super(SimplifiedMBConvBlock, self).__init__()
        mid_channels = in_channels * expand_ratio
        self.use_residual = (in_channels == out_channels and stride == 1)
        
        self.expand_conv = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.expand_bn = nn.BatchNorm2d(mid_channels)
        
        self.depthwise_conv = nn.Conv2d(mid_channels, mid_channels, kernel_size=kernel_size, stride=stride,
                                        padding=kernel_size // 2, groups=mid_channels, bias=False)
        self.depthwise_bn = nn.BatchNorm2d(mid_channels)
        
        self.project_conv = nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False)
        self.project_bn = nn.BatchNorm2d(out_channels)

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        identity = x
        out = F.relu6(self.expand_bn(self.expand_conv(x)))
        out = F.relu6(self.depthwise_bn(self.depthwise_conv(out)))
        out = self.project_bn(self.project_conv(out))
        
        if self.use_residual:
            out += identity
        
        out = self.dropout(out)
        return out

class StudentEfficientNetB4(nn.Module):
    def __init__(self, num_classes=6, dropout_rate=0.5):
        super(StudentEfficientNetB4, self).__init__()
        self.stem_conv = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.stem_bn = nn.BatchNorm2d(32)
        
        self.blocks = nn.Sequential(
            SimplifiedMBConvBlock(32, 16, expand_ratio=1, dropout_rate=dropout_rate),
            SimplifiedMBConvBlock(16, 24, expand_ratio=6, stride=2, dropout_rate=dropout_rate),
            SimplifiedMBConvBlock(24, 40, expand_ratio=6, stride=2, dropout_rate=dropout_rate),
            SimplifiedMBConvBlock(40, 80, expand_ratio=6, stride=2, dropout_rate=dropout_rate),
            SimplifiedMBConvBlock(80, 112, expand_ratio=6, stride=1, dropout_rate=dropout_rate),
            SimplifiedMBConvBlock(112, 350, expand_ratio=6, stride=1, dropout_rate=dropout_rate),
            SimplifiedMBConvBlock(350,572, expand_ratio=6, stride=1, dropout_rate=dropout_rate),
            SimplifiedMBConvBlock(572, 1020, expand_ratio=6, stride=1, dropout_rate=dropout_rate),
            # SimplifiedMBConvBlock(820, 1020, expand_ratio=6, stride=1, dropout_rate=dropout_rate),



        )
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(1020, num_classes)

    def forward(self, x):
        x = F.relu6(self.stem_bn(self.stem_conv(x)))
        x = self.blocks(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x