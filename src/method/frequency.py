import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


RGB_TO_GRAY_WEIGHTS = (0.299, 0.587, 0.114)


def _init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


def compute_frequency_spectrum(image: torch.Tensor, target_size: int = 112) -> torch.Tensor:
    gray = RGB_TO_GRAY_WEIGHTS[0] * image[:, 0] + RGB_TO_GRAY_WEIGHTS[1] * image[:, 1] + RGB_TO_GRAY_WEIGHTS[2] * image[:, 2]

    if gray.shape[-1] != target_size:
        gray = F.interpolate(gray.unsqueeze(1), size=(target_size, target_size),
                            mode='bilinear', align_corners=False).squeeze(1)

    fft = torch.fft.fft2(gray)
    fft_shifted = torch.fft.fftshift(fft)
    magnitude = torch.abs(fft_shifted)

    H, W = magnitude.shape[-2:]
    center_y, center_x = H // 2, W // 2
    y, x = torch.meshgrid(torch.arange(H, device=gray.device),
                         torch.arange(W, device=gray.device), indexing='ij')
    dist = torch.sqrt((x - center_x)**2 + (y - center_y)**2)
    cutoff = min(H, W) / 8
    high_pass = torch.sigmoid((dist - cutoff) / 10)

    magnitude = magnitude * high_pass.unsqueeze(0)

    freq_magnitude = torch.log1p(magnitude)
    min_vals = freq_magnitude.amin(dim=(1, 2), keepdim=True)
    max_vals = freq_magnitude.amax(dim=(1, 2), keepdim=True)
    denom = (max_vals - min_vals).clamp_min(1e-8)
    freq_magnitude = (freq_magnitude - min_vals) / denom

    return freq_magnitude


class FrequencyArtifactDetector(nn.Module):
    def __init__(self, embed_dim: int = 384, evidence_dim: int = 64, fft_size: int = 112):
        super().__init__()
        self.fft_size = fft_size

        self.freq_encoder = nn.Sequential(
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

        self.cls_processor = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )

        self.fusion = nn.Sequential(
            nn.Linear(128 + embed_dim // 2, evidence_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(evidence_dim * 2, evidence_dim),
        )

        self.classifier = nn.Linear(evidence_dim, 1)

        self.apply(_init_weights)

    def forward(self, raw_image: torch.Tensor, cls_token: torch.Tensor,
                freq_cached: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        if freq_cached is None:
            freq_cached = compute_frequency_spectrum(raw_image, target_size=self.fft_size)

        magnitude = freq_cached.unsqueeze(1) if freq_cached.dim() == 3 else freq_cached
        if magnitude.shape[-1] != self.fft_size:
            magnitude = F.interpolate(magnitude, size=(self.fft_size, self.fft_size),
                                    mode="bilinear", align_corners=False)

        freq_feat = self.freq_encoder(magnitude).flatten(1)
        cls_feat = self.cls_processor(cls_token)

        fused = torch.cat([freq_feat, cls_feat], dim=1)
        evidence = self.fusion(fused)
        logit = self.classifier(evidence)

        return {"evidence": evidence, "logit": logit, "score": torch.sigmoid(logit)}
