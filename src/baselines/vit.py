import torch
import torch.nn as nn
import timm
from typing import Dict, Optional
import os
from pathlib import Path

os.environ['TORCH_HOME'] = str(Path(__file__).parent.parent.parent / 'data' / 'torch_cache')
os.environ['TIMM_CACHE_DIR'] = str(Path(__file__).parent.parent.parent / 'data' / 'torch_cache' / 'hub')


class ViTBaseline(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model("vit_small_patch16_224", pretrained=pretrained, num_classes=1)

    def forward(self, x: torch.Tensor, freq_cached: Optional[torch.Tensor] = None,
                sobel_cached: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        logit = self.backbone(x)
        return {
            "logit": logit,
            "prob": torch.sigmoid(logit),
        }
