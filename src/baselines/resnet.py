import torch
import torch.nn as nn
import torchvision.models as models
from typing import Dict, Optional


class ResNetBaseline(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.backbone = models.resnet50(pretrained=pretrained)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 1)

    def forward(self, x: torch.Tensor, freq_cached: Optional[torch.Tensor] = None,
                sobel_cached: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        logit = self.backbone(x)
        return {
            "logit": logit,
            "prob": torch.sigmoid(logit),
        }
