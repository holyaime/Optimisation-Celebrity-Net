import torch
import torch.nn as nn
import timm

class ModifiedCelebrityCosinu(nn.Module):
    def __init__(self, num_classes: int = 6, pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model("efficientnet_b4", pretrained=pretrained)
        self.backbone.classifier = nn.Linear(
            in_features=1792, out_features=num_classes, bias=True
        )
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone.forward_features(x)
        x = self.global_pool(x)
        flattened_conv_output = torch.flatten(x, 1)
        x = self.backbone.classifier(flattened_conv_output)
        return x, flattened_conv_output