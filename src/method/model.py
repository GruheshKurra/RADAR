import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, Optional
import timm


@dataclass
class RADARConfig:
    img_size: int = 224
    patch_size: int = 16
    embed_dim: int = 384
    evidence_dim: int = 64
    reasoning_iterations: int = 3
    reasoning_heads: int = 4
    fft_size: int = 112
    dropout: float = 0.1


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
    """
    Compute frequency spectrum using FFT with high-pass filter.

    This is the canonical frequency extraction for AADM. The same logic is used
    for on-the-fly computation and should match preprocessing when applicable.

    Args:
        image: RGB image tensor (B, 3, H, W) with ImageNet-style normalization
            (per-channel mean subtraction and std division) as used in dataset preprocessing.
        target_size: Size to resize for FFT computation

    Returns:
        Normalized frequency magnitude (B, H, W)
    """
    # Convert to grayscale
    gray = 0.299 * image[:, 0] + 0.587 * image[:, 1] + 0.114 * image[:, 2]

    # Resize if needed
    if gray.shape[-1] != target_size:
        gray = F.interpolate(gray.unsqueeze(1), size=(target_size, target_size),
                            mode='bilinear', align_corners=False).squeeze(1)

    # FFT and shift to center
    fft = torch.fft.fft2(gray)
    fft_shifted = torch.fft.fftshift(fft)
    magnitude = torch.abs(fft_shifted)

    # Apply high-pass filter (emphasize high frequencies)
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


class BoundaryArtifactDetector(nn.Module):
    """
    Boundary Artifact Detection Module (BADM).

    Processes Sobel edge maps (1-channel grayscale) through a CNN encoder.
    Note: Sobel is computed as single-channel gradient magnitude, then used directly
    without replication to RGB channels. The edge_encoder is designed for 1-channel input.
    """
    def __init__(self, embed_dim: int = 384, evidence_dim: int = 64):
        super().__init__()

        # Edge encoder: takes 1-channel Sobel edge map
        self.edge_encoder = nn.Sequential(
            nn.Conv2d(1, 32, 7, stride=4, padding=3),  # 1-channel input (Sobel magnitude)
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
            # Compute grayscale using luminance formula
            gray = 0.299 * image[:, 0] + 0.587 * image[:, 1] + 0.114 * image[:, 2]
            gray = gray.unsqueeze(1)

            # Compute Sobel edge detection
            sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                                  dtype=image.dtype, device=image.device).view(1, 1, 3, 3)
            sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                                  dtype=image.dtype, device=image.device).view(1, 1, 3, 3)

            grad_x = F.conv2d(gray, sobel_x, padding=1)
            grad_y = F.conv2d(gray, sobel_y, padding=1)
            sobel_cached = torch.sqrt(grad_x**2 + grad_y**2)
            sobel_cached = sobel_cached / (sobel_cached.max() + 1e-8)
            # Keep as 1-channel (B, 1, H, W) - no replication to 3 channels

        # Ensure sobel_cached is 4D (B, 1, H, W)
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
        # Compute frequency spectrum if not cached
        if freq_cached is None:
            freq_cached = compute_frequency_spectrum(raw_image, target_size=self.fft_size)

        # Ensure correct shape (B, 1, H, W)
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


class EvidenceCrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

        self.apply(_init_weights)

    def forward(self, query: torch.Tensor, evidence: torch.Tensor):
        B = query.shape[0]
        num_evidence = evidence.shape[1]

        Q = self.q_proj(query).view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(evidence).view(B, num_evidence, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(evidence).view(B, num_evidence, self.num_heads, self.head_dim).transpose(1, 2)

        attn = (Q @ K.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ V).transpose(1, 2).contiguous().view(B, 1, -1)
        out = self.out_proj(out.squeeze(1))

        return out, attn.squeeze(1).mean(dim=1)


class EvidenceRefinementModule(nn.Module):
    """
    Evidence Refinement Module (ERM).

    Uses iterative cross-attention and GRU to refine evidence from BADM and AADM.
    Fixed to always expect 2 evidence sources (BADM + AADM) for architectural consistency.
    """
    def __init__(self, evidence_dim: int = 64, num_heads: int = 4,
                 num_iterations: int = 3, dropout: float = 0.1):
        super().__init__()
        self.num_iterations = num_iterations
        self.evidence_dim = evidence_dim

        # Fixed initialization: always assume 2 evidence sources (BADM + AADM)
        self.init_proj = nn.Sequential(
            nn.Linear(self.evidence_dim * 2, self.evidence_dim),
            nn.ReLU(inplace=True),
        )

        self.attention = EvidenceCrossAttention(evidence_dim, num_heads, dropout)
        self.gru = nn.GRUCell(evidence_dim, evidence_dim)
        self.predictor = nn.Linear(evidence_dim, 1)

        self.apply(_init_weights)

    def forward(self, *evidence_list: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass with fixed 2 evidence sources.

        Args:
            evidence_list: Tuple of evidence tensors (BADM, AADM)
        """
        if len(evidence_list) != 2:
            raise ValueError(f"ERM expects exactly 2 evidence sources, got {len(evidence_list)}")

        B = evidence_list[0].shape[0]
        device = evidence_list[0].device
        evidence_stack = torch.stack(evidence_list, dim=1)

        # Initialize hidden state from concatenated evidence
        h = self.init_proj(torch.cat(evidence_list, dim=1))

        iteration_logits = torch.zeros(B, self.num_iterations, 1, device=device)
        iteration_probs = torch.zeros(B, self.num_iterations, 1, device=device)
        attention_history = torch.zeros(B, self.num_iterations, 2, device=device)

        for t in range(self.num_iterations):
            context, attn_weights = self.attention(h, evidence_stack)
            attention_history[:, t] = attn_weights

            h = self.gru(context, h)
            logit = self.predictor(h)

            iteration_logits[:, t] = logit
            iteration_probs[:, t] = torch.sigmoid(logit)

        final_logit = iteration_logits[:, -1]

        convergence_delta = torch.tensor(0.0, device=device)
        if self.num_iterations > 1:
            prob_diff = torch.abs(iteration_probs[:, -1] - iteration_probs[:, -2])
            convergence_delta = prob_diff.mean()

        return {
            "final_logit": final_logit,
            "iteration_logits": iteration_logits.squeeze(-1),
            "iteration_probs": iteration_probs.squeeze(-1),
            "attention_history": attention_history,
            "convergence_delta": convergence_delta,
        }


class RADAR(nn.Module):
    def __init__(self, config: RADARConfig):
        super().__init__()
        self.config = config

        self.encoder = timm.create_model(
            "vit_small_patch16_224",
            pretrained=True,
            num_classes=0,
        )

        self.badm = BoundaryArtifactDetector(config.embed_dim, config.evidence_dim)
        self.aadm = FrequencyArtifactDetector(
            config.embed_dim, config.evidence_dim, config.fft_size
        )

        self.reasoning = EvidenceRefinementModule(
            config.evidence_dim, config.reasoning_heads,
            config.reasoning_iterations, config.dropout
        )

        self.external_classifier = nn.Sequential(
            nn.Linear(config.evidence_dim * 2, config.evidence_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout),
            nn.Linear(config.evidence_dim, 1)
        )
        self.external_classifier.apply(_init_weights)

    def forward(self, x: torch.Tensor, freq_cached: Optional[torch.Tensor] = None,
                sobel_cached: Optional[torch.Tensor] = None, use_badm: bool = True,
                use_aadm: bool = True) -> Dict[str, torch.Tensor]:
        B = x.shape[0]
        device = x.device

        vit_out = self.encoder.forward_features(x)
        cls_token = vit_out[:, 0]
        patch_features = vit_out[:, 1:]

        badm_out = self.badm(x, patch_features, sobel_cached) if use_badm else None
        aadm_out = self.aadm(x, cls_token, freq_cached) if use_aadm else None

        # Reasoning module requires BOTH evidence sources (BADM + AADM)
        # For ablations, we skip reasoning and use zero outputs
        if use_badm and use_aadm and badm_out is not None and aadm_out is not None:
            # Full model: use reasoning module with both evidence sources
            evidence_list = [badm_out["evidence"], aadm_out["evidence"]]
            reasoning_out = self.reasoning(*evidence_list)

            external_input = torch.cat(evidence_list, dim=1)
            external_logit = self.external_classifier(external_input)
            final_logit = (reasoning_out["final_logit"] + external_logit) / 2
        else:
            # Ablation mode: return zero outputs
            reasoning_out = {
                "final_logit": torch.zeros(B, 1, device=device),
                "iteration_logits": torch.zeros(B, self.config.reasoning_iterations, device=device),
                "iteration_probs": torch.zeros(B, self.config.reasoning_iterations, device=device),
                "attention_history": torch.zeros(B, self.config.reasoning_iterations, 2, device=device),
                "convergence_delta": torch.tensor(0.0, device=device),
            }
            external_logit = torch.zeros(B, 1, device=device)
            final_logit = torch.zeros(B, 1, device=device)

        return {
            "logit": final_logit,
            "prob": torch.sigmoid(final_logit),
            "external_logit": external_logit,
            "badm_logit": badm_out["logit"] if badm_out is not None else torch.zeros(B, 1, device=device),
            "badm_score": badm_out["score"] if badm_out is not None else torch.zeros(B, 1, device=device),
            "badm_evidence": badm_out["evidence"] if badm_out is not None else torch.zeros(B, self.config.evidence_dim, device=device),
            "aadm_logit": aadm_out["logit"] if aadm_out is not None else torch.zeros(B, 1, device=device),
            "aadm_score": aadm_out["score"] if aadm_out is not None else torch.zeros(B, 1, device=device),
            "aadm_evidence": aadm_out["evidence"] if aadm_out is not None else torch.zeros(B, self.config.evidence_dim, device=device),
            "attention_history": reasoning_out["attention_history"],
            "iteration_logits": reasoning_out["iteration_logits"],
            "iteration_probs": reasoning_out["iteration_probs"],
            "convergence_delta": reasoning_out["convergence_delta"],
        }
