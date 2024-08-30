# -*- coding: utf-8 -*-
import timm  # type: ignore
import torch
import torch.nn as nn
class ModifiedCelebrityRegressor(torch.nn.Module):
    def __init__(self, num_classes: int = 6, pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model("efficientnet_b4", pretrained=pretrained)

        # Modifier la dernière couche pour s'adapter au nombre de classes
        self.backbone.classifier = torch.nn.Linear(
            in_features=1792, out_features=num_classes, bias=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Passer l'entrée à travers toutes les couches convolutives
        x = self.backbone.forward_features(x)

        # Capturer la carte de caractéristiques ici, après les convolutions et les pooling layers
        conv_feature_map = x
        
        # Utiliser ces caractéristiques pour obtenir les logits
        x = self.backbone.forward_head(conv_feature_map, pre_logits=False)

        # Retourner les logits et la carte de caractéristiques
        return x, conv_feature_map
