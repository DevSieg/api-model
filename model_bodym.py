import torch, torch.nn as nn
from torchvision.models import resnet101
from torchvision.models import ResNet101_Weights  # cambia aquí a ResNet50


class BMnetLite(nn.Module):
    """
    ResNet101 que recibe 2 canales (front+side) + MLP con height(+weight).
    Devuelve 15 medidas normalizadas.
    """
    def __init__(self, out_dim=15, use_weight=True, pretrained=True):
        super().__init__()
        # Cargar ResNet18 (intenta con pesos ImageNet; si falla, sin pesos)
        try:
            from torchvision.models import ResNet18_Weights
            weights = ResNet101_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = resnet101(weights=weights)
        except Exception:
            backbone = resnet101(weights=None)

        w = backbone.conv1.weight                      # [64,3,7,7]
        backbone.conv1 = nn.Conv2d(2, 64, 7, 2, 3, bias=False)
        with torch.no_grad():
            # inicializa promediando los 3 canales a 2
            backbone.conv1.weight.copy_(w.mean(1, keepdim=True)[:, :2, :, :])

        feat_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone

        in_scalars = 2 if use_weight else 1
        self.head = nn.Sequential(
            nn.Linear(feat_dim + in_scalars, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, out_dim)
        )
        self.use_weight = use_weight

    def forward(self, img, scalars):
        z = self.backbone(img)                 # [B, feat ]
        if not self.use_weight:
            scalars = scalars[:, :1]
        x = torch.cat([z, scalars], dim=1)
        return self.head(x)
