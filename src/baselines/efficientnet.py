import torch
import torch.nn as nn
import timm
from typing import Dict, Optional


class EfficientNetBaseline(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model("tf_efficientnet_b0", pretrained=pretrained, num_classes=1)

    def forward(self, x: torch.Tensor, freq_cached: Optional[torch.Tensor] = None,
                sobel_cached: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        logit = self.backbone(x)
        return {
            "logit": logit,
            "prob": torch.sigmoid(logit),
        }
