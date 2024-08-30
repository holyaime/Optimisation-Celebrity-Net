# -*- coding: utf-8 -*-
import timm  # type: ignore
import torch
import torch.nn as nn

class Mobilenet_100(nn.Module):
    def __init__(self, num_classes: int = 6, backbone_name: str = "mobilenetv2_100", pretrained: bool = True, dropout_rate: float = 0.5):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained)
        
        # Ajouter des Dropout après certaines couches spécifiques
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),  # Dropout avant la couche finale
            nn.Linear(in_features=1280, out_features=num_classes, bias=True)
        )
        self.dropout = nn.Dropout(p=dropout_rate)  # Dropout général après le global pooling

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone.forward_features(x)  # Extraire les caractéristiques de base
        x = x.mean([2, 3])  # Global average pooling, si nécessaire
        x = self.dropout(x)  # Ajouter un Dropout avant la classification finale
        x = self.backbone.classifier(x)  # Classifier
        return x
   


