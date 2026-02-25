"""
Loss functions for landmark detection tasks.
"""

import torch
import torch.nn as nn


class CustomNLLLoss(nn.Module):
    """
    Custom Negative Log-Likelihood loss for landmark detection with heatmaps.

    Args:
        reduction (str): Specifies the reduction to apply to the output.
            'none': no reduction will be applied
            'mean': the sum of the output will be divided by the number of elements
            'perkeypoint': the loss will be calculated per keypoint and returned
    """
    def __init__(self, reduction="mean"):
        super(CustomNLLLoss, self).__init__()
        assert reduction in {"mean", "none", "perkeypoint"}
        self.reduction = reduction

    @staticmethod
    def two_d_softmax(x):
        """Apply 2D softmax to heatmaps."""
        exp_y = torch.exp(x)
        return exp_y / torch.sum(exp_y, dim=(2, 3), keepdim=True)

    def forward(self, output, target):
        """
        Forward pass for the NLL loss.

        Args:
            output (torch.Tensor): Predicted heatmaps
            target (torch.Tensor): Target heatmaps

        Returns:
            torch.Tensor: Calculated loss
        """
        # Compute NLL with numerical stability
        nll = -target * torch.log(output.float() + 1e-10)

        if self.reduction == "none":
            return nll
        elif self.reduction == "perkeypoint":
            return torch.sum(nll, dim=(2, 3))
        elif self.reduction == "mean":
            return torch.mean(torch.sum(nll, dim=(2, 3)))
