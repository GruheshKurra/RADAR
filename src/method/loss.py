import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
from dataclasses import dataclass


@dataclass
class LossConfig:
    """
    Configuration for RADAR composite loss function.

    Attributes:
        lambda_main: Weight for main classification loss
        lambda_branch: Weight for branch supervision losses (BADM + AADM)
        lambda_orthogonal: Weight for evidence orthogonality constraint
        lambda_deep_supervision: Weight for iteration-wise deep supervision
        label_smoothing: Label smoothing factor for BCE loss
        orthogonality_margin: Margin for hinge-style orthogonality loss.
            See RADARLoss documentation for mathematical justification.
    """
    lambda_main: float = 1.0
    lambda_branch: float = 0.3
    lambda_orthogonal: float = 0.1
    lambda_deep_supervision: float = 0.05
    label_smoothing: float = 0.1
    orthogonality_margin: float = 0.1


class RADARLoss(nn.Module):
    """
    RADAR composite loss function with proper ablation support.

    Loss Components:
        1. Main Loss: BCE on final logit
        2. Branch Loss: BCE on BADM/AADM individual classifiers
        3. Orthogonality Loss: Encourages evidence disentanglement
        4. Deep Supervision: Weighted BCE on reasoning iterations

    Orthogonality Loss (Mathematical Justification):
        We use a hinge-style orthogonality constraint rather than direct
        cosine-squared minimization:

            L_orth = ReLU(|cos(e_badm, e_aadm)| - margin)

        Mathematical rationale:
        - Direct minimization (cos² → 0) is too strict and can collapse
          representations to degenerate solutions
        - Hinge formulation allows "soft orthogonality": evidence vectors
          can have small correlations (|cos| < margin) without penalty
        - margin = 0.1 allows ~6° tolerance from true orthogonality
          (arccos(0.1) ≈ 84.3°)

        Alternatives considered:
        - Cosine-squared: L = cos²(θ) — too aggressive, causes instability
        - Dot-product: L = |e1 · e2| — scale-dependent, requires normalization
        - Gram-Schmidt: Explicit orthogonalization — destroys gradients

        The hinge formulation provides stable training while effectively
        encouraging the two evidence branches to capture complementary
        (non-redundant) artifact information.

    Args:
        config: Loss configuration with weights for each component
    """
    def __init__(self, config: LossConfig):
        super().__init__()
        self.config = config
        self.bce = nn.BCEWithLogitsLoss(reduction='mean')

    def forward(self, outputs: Dict[str, torch.Tensor], labels: torch.Tensor,
                use_badm: bool = True, use_aadm: bool = True) -> Dict[str, torch.Tensor]:
        """
        Compute losses with proper masking for ablation studies.

        Args:
            outputs: Model outputs dictionary
            labels: Ground truth labels
            use_badm: Whether BADM is enabled (for proper loss masking)
            use_aadm: Whether AADM is enabled (for proper loss masking)

        Returns:
            Dictionary with:
                - total: Weighted sum of all losses
                - main: Main classification loss
                - branch: Branch supervision loss (if both branches active)
                - orthogonal: Evidence orthogonality loss (if both branches active)
                - deep_supervision: Iteration-wise supervision loss
        """
        labels_float = labels.float().unsqueeze(1)

        if self.config.label_smoothing > 0:
            labels_smoothed = labels_float * (1 - self.config.label_smoothing) + 0.5 * self.config.label_smoothing
        else:
            labels_smoothed = labels_float

        loss_main = self.bce(outputs["logit"], labels_smoothed)

        # Branch loss: only apply if both branches are enabled
        loss_branch = torch.tensor(0.0, device=labels.device)
        if use_badm and use_aadm:
            loss_branch = 0.5 * (
                self.bce(outputs["badm_logit"], labels_smoothed) +
                self.bce(outputs["aadm_logit"], labels_smoothed)
            )
        elif use_badm:
            loss_branch = self.bce(outputs["badm_logit"], labels_smoothed)
        elif use_aadm:
            loss_branch = self.bce(outputs["aadm_logit"], labels_smoothed)

        # Orthogonality loss: only apply if both branches are enabled
        # Uses hinge-style formulation: ReLU(|cos_sim| - margin)
        # See class docstring for mathematical justification
        loss_orthogonal = torch.tensor(0.0, device=labels.device)
        if use_badm and use_aadm:
            badm_norm = torch.norm(outputs["badm_evidence"], dim=1, keepdim=True)
            aadm_norm = torch.norm(outputs["aadm_evidence"], dim=1, keepdim=True)

            if (badm_norm > 1e-6).all() and (aadm_norm > 1e-6).all():
                badm_ev = outputs["badm_evidence"] / (badm_norm + 1e-8)
                aadm_ev = outputs["aadm_evidence"] / (aadm_norm + 1e-8)
                cosine_sim = (badm_ev * aadm_ev).sum(dim=1)
                # Hinge loss: only penalize when |cos| > margin
                loss_orthogonal = F.relu(torch.abs(cosine_sim) - self.config.orthogonality_margin).mean()

        loss_deep_supervision = torch.tensor(0.0, device=labels.device)
        if outputs["iteration_logits"].shape[1] >= 1:
            num_iters = outputs["iteration_logits"].shape[1]
            # Progressive weights: later iterations weighted more heavily
            weights = torch.linspace(1.0 / num_iters, 1.0, num_iters, device=labels.device)
            weights = weights / weights.sum()

            labels_expanded = labels_smoothed.expand(-1, num_iters).to(device=labels.device)
            iter_losses = F.binary_cross_entropy_with_logits(outputs["iteration_logits"], labels_expanded, reduction='none')
            loss_deep_supervision = (iter_losses.mean(dim=0) * weights).sum()

        total_loss = (
            self.config.lambda_main * loss_main +
            self.config.lambda_branch * loss_branch +
            self.config.lambda_orthogonal * loss_orthogonal +
            self.config.lambda_deep_supervision * loss_deep_supervision
        )

        return {
            "total": total_loss,
            "main": loss_main,
            "branch": loss_branch,
            "orthogonal": loss_orthogonal,
            "deep_supervision": loss_deep_supervision,
        }
