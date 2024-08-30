# -*- coding: utf-8 -*-
import timm  # type: ignore
import torch.nn as nn
import torch

class CelebrityNet(nn.Module):
    def __init__(self, num_classes: int = 6, backbone_name: str ="efficientnet_b4", pretrained: bool = True):
        #dropout_rate: float = 0.2
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained)
        
          # Remplacer le classificateur pour correspondre au nombre de classes désirées
        self.backbone.classifier = nn.Linear(
            in_features=1792, out_features=num_classes, bias=True
        )

        # # Ajouter une couche pour adapter les caractéristiques intermédiaires
        # self.feature_adaptation = nn.Conv2d(
        # in_channels=1280, out_channels=1792, kernel_size=1
        # )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        #x = self.dropout(x)
        return x
    # def get_intermediate_features(self, x: torch.Tensor) -> torch.Tensor:
    #     # Extraire les caractéristiques intermédiaires
    #     features = self.backbone.forward_features(x)
    #     # Adapter les dimensions des caractéristiques
    #     adapted_features = self.feature_adaptation(features)
    #     return adapted_features
