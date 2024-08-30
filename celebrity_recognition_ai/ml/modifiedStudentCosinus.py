import torch
import torch.nn as nn
import timm
class ModifiedStudentNetboCosinu(nn.Module):
    def __init__(self, num_classes: int = 6, pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model("efficientnet_b0", pretrained=pretrained)
        self.backbone.classifier = nn.Linear(
            in_features=1280, out_features=num_classes, bias=True
        )
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.projection = nn.Linear(1280, 1792)  # Projection layer to match teacher dimensions

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone.forward_features(x)
        x = self.global_pool(x)
        flattened_conv_output = torch.flatten(x, 1)
        x = self.backbone.classifier(flattened_conv_output)
        flattened_conv_output_proj = self.projection(flattened_conv_output)  # Project to match teacher's dimension

        return x, flattened_conv_output_proj
