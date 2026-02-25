"""
Loss functions for CDPM-align training.

Implements:
- alignment_loss: multi-scale cosine dissimilarity on guidance-difference embeddings
- diffusion_loss: standard MSE on noise prediction
- CDPMAlignLoss: combined loss module (Ldiff + λ·Lalign)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional

from .numerical_stability import (
    safe_normalize,
    clamp_prediction,
    clamp_loss,
    clamp_cosine_similarity,
)


def _cosine_dissimilarity(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    """
    Per-sample cosine dissimilarity (1 - cosine_sim), averaged over batch.
    Inputs are assumed to be L2-normalised.
    """
    sim = (z1 * z2).sum(dim=-1)
    sim = clamp_cosine_similarity(sim)
    return (1.0 - sim).mean()


def alignment_loss(
    feats_u1: Dict[str, torch.Tensor],
    feats_c1: Dict[str, torch.Tensor],
    feats_u2: Dict[str, torch.Tensor],
    feats_c2: Dict[str, torch.Tensor],
    projector: nn.Module,
) -> torch.Tensor:
    """
    Multi-scale alignment loss on guidance-difference embeddings.

    For each level ℓ:
        Δh_ℓ^(i) = h_cond^(i)[ℓ] - h_uncond^(i)[ℓ]   for views i ∈ {1, 2}
        z_ℓ^(i)  = g_ℓ(GAP(Δh_ℓ^(i)))     (projected + L2-normalised)
        loss_ℓ   = 1 - cos(z_ℓ^(1), z_ℓ^(2))

    Returns the average over levels.
    """
    layer_set = projector.LAYER_SET
    total_loss = torch.tensor(0.0, device=next(iter(feats_u1.values())).device)
    num_layers = 0

    for layer in layer_set:
        if not all(layer in feats for feats in [feats_u1, feats_c1, feats_u2, feats_c2]):
            continue
        if layer not in projector.projectors:
            continue

        delta_h1 = feats_c1[layer] - feats_u1[layer]
        delta_h2 = feats_c2[layer] - feats_u2[layer]

        z1 = projector.pool_feature(delta_h1)
        z2 = projector.pool_feature(delta_h2)

        z1 = projector.projectors[layer](z1)
        z2 = projector.projectors[layer](z2)

        z1 = safe_normalize(z1, dim=-1)
        z2 = safe_normalize(z2, dim=-1)

        layer_loss = clamp_loss(_cosine_dissimilarity(z1, z2), max_val=10.0)
        total_loss = total_loss + layer_loss
        num_layers += 1

    if num_layers == 0:
        return total_loss
    return total_loss / num_layers


def diffusion_loss(eps_pred: torch.Tensor, eps_target: torch.Tensor) -> torch.Tensor:
    """Standard MSE diffusion loss with prediction clamping."""
    eps_pred = clamp_prediction(eps_pred)
    return clamp_loss(F.mse_loss(eps_pred, eps_target))


class CDPMAlignLoss(nn.Module):
    """
    Combined diffusion + alignment loss: L = Ldiff + λ_align · Lalign.

    During the alignment phase, Ldiff is computed over all four noise predictions
    (both timesteps × both conditioning modes) to preserve both generation pathways.
    """

    def __init__(
        self,
        projector: nn.Module,
        lambda_diff: float = 1.0,
        lambda_align: float = 5.0,
    ):
        super().__init__()
        self.projector = projector
        self.lambda_diff = lambda_diff
        self.lambda_align = lambda_align

    def forward(
        self,
        eps_c1: torch.Tensor, eps_u1: torch.Tensor, eps1: torch.Tensor,
        eps_c2: torch.Tensor, eps_u2: torch.Tensor, eps2: torch.Tensor,
        feats_u1: Dict[str, torch.Tensor],
        feats_c1: Dict[str, torch.Tensor],
        feats_u2: Dict[str, torch.Tensor],
        feats_c2: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # Diffusion loss over both timesteps and both conditioning modes
        loss_diff_c = (diffusion_loss(eps_c1, eps1) + diffusion_loss(eps_c2, eps2)) / 2
        loss_diff_u = (diffusion_loss(eps_u1, eps1) + diffusion_loss(eps_u2, eps2)) / 2
        loss_diff = loss_diff_c + loss_diff_u

        # Multi-scale alignment loss
        loss_align = alignment_loss(feats_u1, feats_c1, feats_u2, feats_c2, self.projector)

        total = self.lambda_diff * loss_diff + self.lambda_align * loss_align

        loss_dict = {
            'loss_total': total.detach(),
            'loss_diff': loss_diff.detach(),
            'loss_align': loss_align.detach(),
        }
        return total, loss_dict
