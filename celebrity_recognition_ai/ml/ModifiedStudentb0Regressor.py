import timm  # type: ignore
import torch
import torch.nn as nn

class ModifiedStudentNetboRegressor(torch.nn.Module):
    def __init__(self, num_classes: int = 6, backbone_name: str = "efficientnet_b0", pretrained: bool = True, dropout_rate: float = 0.5):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained)

        # Modifier la dernière couche pour s'adapter au nombre de classes
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(self.backbone.classifier.in_features, num_classes)
        )
        
        # Ajuster le régresseur pour accepter le nombre correct de canaux
        self.regressor = nn.Sequential(
            nn.Conv2d(1280, 448, kernel_size=3, padding=1),  # Corriger les canaux d'entrée et sortie
            nn.ReLU(),  # Ajouter l'activation ReLU
            nn.Conv2d(448, 1792, kernel_size=1)  # Couche pour aligner les canaux à 1792
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Obtenir les caractéristiques intermédiaires du modèle de base
        x = self.backbone.forward_features(x)
        
        # Appliquer le régresseur sur cette carte de caractéristiques
        regressor_output = self.regressor(x)
        
        # Obtenir les logits à partir des caractéristiques
        logits = self.backbone.forward_head(x, pre_logits=False)

        # Retourner les logits et la sortie du régresseur
        return logits, regressor_output
