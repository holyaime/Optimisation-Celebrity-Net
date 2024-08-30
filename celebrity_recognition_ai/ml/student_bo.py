# -*- coding: utf-8 -*-
import timm  # type: ignore
import torch
import torch.nn as nn

class StudentNetbo(torch.nn.Module):
    def __init__(self, num_classes: int = 6, backbone_name: str = "efficientnet_b0", pretrained: bool = True, dropout_rate: float = 0.5):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained)
        
        # Obtenir le nombre de caractéristiques en sortie de la dernière couche convolutive
        in_features = self.backbone.classifier.in_features
        
        # Ajouter des couches de Dropout et remplacer le classificateur
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        return x