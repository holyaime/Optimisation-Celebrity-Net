# -*- coding: utf-8 -*-
import timm  # type: ignore
import torch

class CelebrityNet(torch.nn.Module):
    def __init__(self, num_classes: int = 6, pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model("efficientnet_b4", pretrained=pretrained)

        self.backbone.classifier = torch.nn.Linear(
            in_features=1792, out_features=num_classes, bias=True
        )
      

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        #x = self.dropout(x)
        return x
   