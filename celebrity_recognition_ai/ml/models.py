# -*- coding: utf-8 -*-
import timm
import torch.nn as nn


class CelebrityNet(nn.Module):
    def __init__(self, num_classes=6, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model("efficientnet_b4", pretrained=pretrained)
        self.backbone.classifier = nn.Linear(
            in_features=1792, out_features=num_classes, bias=True
        )

    def forward(self, x):
        x = self.backbone(x)
        return x
