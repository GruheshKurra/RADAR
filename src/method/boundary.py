import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


def _init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class BoundaryArtifactDetector(nn.Module):
    def __init__(self, embed_dim: int = 384, evidence_dim: int = 64):
        super().__init__()

        self.edge_encoder = nn.Sequential(
            nn.Conv2d(1, 32, 7, stride=4, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )

        self.patch_processor = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(inplace=True),
        )

        self.fusion = nn.Sequential(
            nn.Linear(128 + embed_dim // 2, evidence_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(evidence_dim * 2, evidence_dim),
        )

        self.classifier = nn.Linear(evidence_dim, 1)

        self.apply(_init_weights)

    def forward(self, image: torch.Tensor, patch_features: torch.Tensor,
                sobel_cached: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        if sobel_cached is None:
            gray = 0.299 * image[:, 0] + 0.587 * image[:, 1] + 0.114 * image[:, 2]
            gray = gray.unsqueeze(1)

            sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                                  dtype=image.dtype, device=image.device).view(1, 1, 3, 3)
            sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                                  dtype=image.dtype, device=image.device).view(1, 1, 3, 3)

            grad_x = F.conv2d(gray, sobel_x, padding=1)
            grad_y = F.conv2d(gray, sobel_y, padding=1)
            sobel_cached = torch.sqrt(grad_x**2 + grad_y**2)

            batch_size = sobel_cached.shape[0]
            for i in range(batch_size):
                max_val = sobel_cached[i].max()
                if max_val > 1e-8:
                    sobel_cached[i] = sobel_cached[i] / max_val

        edges = sobel_cached.unsqueeze(1) if sobel_cached.dim() == 3 else sobel_cached
        edge_feat = self.edge_encoder(edges).flatten(1)

        patch_mean = patch_features.mean(dim=1)
        patch_max = patch_features.max(dim=1)[0]
        patch_aggregated = torch.cat([patch_mean, patch_max], dim=1)
        patch_feat = self.patch_processor(patch_aggregated)

        fused = torch.cat([edge_feat, patch_feat], dim=1)
        evidence = self.fusion(fused)
        logit = self.classifier(evidence)

        return {"evidence": evidence, "logit": logit, "score": torch.sigmoid(logit)}
