import timm  # type: ignore
import torch
import torch.nn as nn

class modifiedMobilenet_100Reg(nn.Module):
    def __init__(self, num_classes: int = 6, backbone_name: str = "mobilenetv2_100", pretrained: bool = True, dropout_rate: float = 0.5):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained)
        
        # Ajouter des Dropout après certaines couches spécifiques
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),  # Dropout avant la couche finale
            nn.Linear(in_features=1280, out_features=num_classes, bias=True)
        )
        self.dropout = nn.Dropout(p=dropout_rate)  # Dropout général après le global pooling

         # Ajuster le régresseur pour accepter le nombre correct de canaux
        self.regressor = nn.Sequential(
            nn.Conv2d(1280, 350, kernel_size=3, padding=1),  # Corriger les canaux d'entrée et sortie
            nn.ReLU(),  # Ajouter l'activation ReLU
            nn.Conv2d(350, 1792, kernel_size=1)  # Couche pour aligner les canaux à 1792
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone.forward_features(x)  # Extraire les caractéristiques de base
        regressor_output = self.regressor(x)  # Appliquer la régression
        x = x.mean([2, 3])  # Global average pooling, si nécessaire
        x = self.dropout(x)  # Ajouter un Dropout avant la classification finale
        x = self.backbone.classifier(x)  # Classifier
        return x, regressor_output
