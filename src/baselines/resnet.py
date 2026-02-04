import torch
import torch.nn as nn
import torchvision.models as models
from typing import Dict, Optional
import os
from pathlib import Path

os.environ['TORCH_HOME'] = str(Path(__file__).parent.parent.parent / 'data' / 'torch_cache')


class ResNetBaseline(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        if pretrained:
            self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        else:
            self.backbone = models.resnet50(weights=None)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 1)

    def forward(self, x: torch.Tensor, freq_cached: Optional[torch.Tensor] = None,
                sobel_cached: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        logit = self.backbone(x)
        return {
            "logit": logit,
            "prob": torch.sigmoid(logit),
        }
