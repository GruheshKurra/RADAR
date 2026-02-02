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
    reasoning_iterations: int = 2
    reasoning_heads: int = 4
    fft_size: int = 112
    freq_cutoff_divisor: int = 8
    dropout: float = 0.1
    use_dct: bool = True
    prediction_feedback: bool = True


class BoundaryArtifactDetector(nn.Module):
    def __init__(self, embed_dim: int = 384, evidence_dim: int = 64):
        super().__init__()

        self.edge_encoder = nn.Sequential(
            nn.Conv2d(3, 32, 7, stride=4, padding=3),
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

    def forward(self, image: torch.Tensor, patch_features: torch.Tensor,
                sobel_cached: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        assert sobel_cached is not None, "sobel_cached must be provided during training. Use preprocessing."

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
    def __init__(self, embed_dim: int = 384, evidence_dim: int = 64,
                 freq_cutoff_divisor: int = 8, fft_size: int = 112, use_dct: bool = True):
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

    def forward(self, raw_image: torch.Tensor, cls_token: torch.Tensor,
                freq_cached: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        assert freq_cached is not None, "freq_cached must be provided during training. Use preprocessing."

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

    def forward(self, query: torch.Tensor, evidence: torch.Tensor):
        B = query.shape[0]

        Q = self.q_proj(query).view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(evidence).view(B, 2, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(evidence).view(B, 2, self.num_heads, self.head_dim).transpose(1, 2)

        attn = (Q @ K.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ V).transpose(1, 2).contiguous().view(B, 1, -1)
        out = self.out_proj(out.squeeze(1))

        return out, attn.squeeze(1).mean(dim=1)


class EvidenceRefinementModule(nn.Module):
    def __init__(self, evidence_dim: int = 64, num_heads: int = 4,
                 num_iterations: int = 3, dropout: float = 0.1,
                 prediction_feedback: bool = True):
        super().__init__()
        self.num_iterations = num_iterations
        self.prediction_feedback = prediction_feedback

        self.init_proj = nn.Sequential(
            nn.Linear(evidence_dim * 2, evidence_dim),
            nn.ReLU(inplace=True),
        )

        self.attention = EvidenceCrossAttention(evidence_dim, num_heads, dropout)
        self.gru = nn.GRUCell(evidence_dim, evidence_dim)
        self.predictor = nn.Linear(evidence_dim, 1)

        if prediction_feedback:
            self.feedback_proj = nn.Linear(1, evidence_dim)

    def forward(self, badm_evidence: torch.Tensor,
                aadm_evidence: torch.Tensor) -> Dict[str, torch.Tensor]:
        B = badm_evidence.shape[0]
        device = badm_evidence.device
        evidence_stack = torch.stack([badm_evidence, aadm_evidence], dim=1)

        h = self.init_proj(torch.cat([badm_evidence, aadm_evidence], dim=1))

        iteration_logits = torch.zeros(B, self.num_iterations, 1, device=device)
        iteration_probs = torch.zeros(B, self.num_iterations, 1, device=device)
        attention_history = torch.zeros(B, self.num_iterations, 2, device=device)

        for t in range(self.num_iterations):
            context, attn_weights = self.attention(h, evidence_stack)
            attention_history[:, t] = attn_weights

            if self.prediction_feedback and t > 0:
                feedback = self.feedback_proj(iteration_logits[:, t-1])
                context = context + feedback

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
            config.embed_dim, config.evidence_dim,
            config.freq_cutoff_divisor, config.fft_size, config.use_dct
        )

        self.reasoning = EvidenceRefinementModule(
            config.evidence_dim, config.reasoning_heads,
            config.reasoning_iterations, config.dropout,
            config.prediction_feedback
        )

        self.classifier = nn.Linear(config.evidence_dim * 2, 1)

    def forward(self, x: torch.Tensor, freq_cached: Optional[torch.Tensor] = None,
                sobel_cached: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        vit_out = self.encoder.forward_features(x)
        cls_token = vit_out[:, 0]
        patch_features = vit_out[:, 1:]

        badm_out = self.badm(x, patch_features, sobel_cached)
        aadm_out = self.aadm(x, cls_token, freq_cached)

        evidence_concat = torch.cat([badm_out["evidence"], aadm_out["evidence"]], dim=1)

        reasoning_out = self.reasoning(badm_out["evidence"], aadm_out["evidence"])
        external_logit = self.classifier(evidence_concat)
        main_logit = (reasoning_out["final_logit"] + external_logit) / 2

        return {
            "logit": main_logit,
            "prob": torch.sigmoid(main_logit),
            "badm_logit": badm_out["logit"],
            "badm_score": badm_out["score"],
            "badm_evidence": badm_out["evidence"],
            "aadm_logit": aadm_out["logit"],
            "aadm_score": aadm_out["score"],
            "aadm_evidence": aadm_out["evidence"],
            "attention_history": reasoning_out["attention_history"],
            "iteration_logits": reasoning_out["iteration_logits"],
            "iteration_probs": reasoning_out["iteration_probs"],
            "convergence_delta": reasoning_out["convergence_delta"],
        }
