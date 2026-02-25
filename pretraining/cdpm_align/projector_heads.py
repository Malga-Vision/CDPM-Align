"""
MLP Projection heads for CDPM-align multi-scale guidance alignment loss.

Each UNet hierarchy level gets its own projector head: GAP → MLP → L2-norm.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectorHead(nn.Module):
    """Two-layer MLP projector (in_dim → hidden_dim → out_dim, GELU activation)."""

    def __init__(self, in_dim: int, hidden_dim: int = 512, out_dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class MultiLayerProjector(nn.Module):
    """
    Collection of per-level projection heads for CDPM-align.

    Feature layers extracted at four UNet hierarchy levels:
        enc_1_4, enc_1_8, bottleneck, dec_1_8.
    Each level gets its own projector; outputs are L2-normalised.
    """

    LAYER_SET = ["enc_1_4", "enc_1_8", "bottleneck", "dec_1_8"]

    def __init__(self, feature_dims: dict, hidden_dim: int = 512, out_dim: int = 256):
        """
        Args:
            feature_dims: {layer_name: channel_count}, e.g. {'enc_1_4': 256, ...}
            hidden_dim:   Hidden dimension for all projectors.
            out_dim:      Output embedding dimension (shared).
        """
        super().__init__()
        self.feature_dims = feature_dims
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        self.projectors = nn.ModuleDict({
            layer: ProjectorHead(
                in_dim=feature_dims[layer],
                hidden_dim=hidden_dim,
                out_dim=out_dim
            )
            for layer in self.LAYER_SET
            if layer in feature_dims
        })

    def pool_feature(self, feature_map: torch.Tensor) -> torch.Tensor:
        """Global average pooling: [B, C, H, W] → [B, C]."""
        return feature_map.mean(dim=(2, 3))

    def project_delta_h(
        self,
        feats_cond: dict,
        feats_uncond: dict,
        normalize: bool = True
    ) -> dict:
        """
        Compute delta-h = h_cond - h_uncond and project each level.

        Returns dict of L2-normalised embeddings per layer.
        """
        projected = {}
        for layer in self.LAYER_SET:
            if layer not in feats_cond or layer not in feats_uncond:
                continue
            if layer not in self.projectors:
                continue
            delta_h = feats_cond[layer] - feats_uncond[layer]
            pooled = self.pool_feature(delta_h)
            z = self.projectors[layer](pooled)
            if normalize:
                z = F.normalize(z, dim=-1)
            projected[layer] = z
        return projected
