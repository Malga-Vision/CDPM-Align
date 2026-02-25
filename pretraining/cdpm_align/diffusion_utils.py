"""
Diffusion utilities for CDPM-align training.

Provides forward diffusion (q_sample), DDPM sampling, and DDIM sampling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from typing import Optional, Tuple

from .numerical_stability import (
    get_eps,
    clamp_noise,
    clamp_prediction,
    safe_sqrt,
)


def extract(v: torch.Tensor, t: torch.Tensor, x_shape: tuple) -> torch.Tensor:
    """Extract coefficients at timesteps and reshape for broadcasting."""
    out = torch.gather(v, index=t, dim=0).float()
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))


class GaussianDiffusion(nn.Module):
    """
    Gaussian diffusion process for training and sampling.

    Implements forward diffusion q(x_t | x_0) and reverse sampling (DDPM/DDIM).
    """

    def __init__(self, beta_1: float, beta_T: float, T: int):
        super().__init__()
        self.T = T

        betas = torch.linspace(beta_1, beta_T, T).double()
        alphas = 1.0 - betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:T]

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_bar', alphas_bar)
        self.register_buffer('alphas_bar_prev', alphas_bar_prev)
        self.register_buffer('sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer('sqrt_one_minus_alphas_bar', torch.sqrt(1.0 - alphas_bar))

        self.register_buffer('coeff1', torch.sqrt(1.0 / alphas))
        self.register_buffer('coeff2', self.coeff1 * (1.0 - alphas) / torch.sqrt(1.0 - alphas_bar))
        self.register_buffer('posterior_var', betas * (1.0 - alphas_bar_prev) / (1.0 - alphas_bar))

    def q_sample(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward diffusion: sample x_t from q(x_t | x_0)."""
        if noise is None:
            noise = torch.randn_like(x_0)
        noise = clamp_noise(noise)
        x_t = (
            extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 +
            extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise
        )
        return x_t, noise

    def sample_timesteps(
        self,
        batch_size: int,
        device: torch.device,
        bias_mid: bool = True,
        mid_weight: float = 2.0
    ) -> torch.Tensor:
        """Sample timesteps with optional bias toward mid-range [T/4, 3T/4]."""
        if not bias_mid:
            return torch.randint(0, self.T, (batch_size,), device=device)
        weights = torch.ones(self.T, device=device)
        mid_start, mid_end = self.T // 4, 3 * self.T // 4
        weights[mid_start:mid_end] = mid_weight
        weights = weights / weights.sum()
        return torch.multinomial(weights, batch_size, replacement=True)

    def predict_xt_prev_mean_from_eps(
        self, x_t: torch.Tensor, t: torch.Tensor, eps: torch.Tensor
    ) -> torch.Tensor:
        return (
            extract(self.coeff1, t, x_t.shape) * x_t -
            extract(self.coeff2, t, x_t.shape) * eps
        )

    def p_mean_variance(
        self,
        model: nn.Module,
        x_t: torch.Tensor,
        t: torch.Tensor,
        labels: torch.Tensor,
        labels_uncond: Optional[torch.Tensor] = None,
        w: float = 0.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        var = torch.cat([self.posterior_var[1:2], self.betas[1:]])
        var = extract(var, t, x_t.shape)
        eps = model(x_t, t, labels)
        if w > 0 and labels_uncond is not None:
            eps_uncond = model(x_t, t, labels_uncond)
            eps = (1.0 + w) * eps - w * eps_uncond
        xt_prev_mean = self.predict_xt_prev_mean_from_eps(x_t, t, eps)
        return xt_prev_mean, var

    def ddim_step(
        self,
        model: nn.Module,
        x_t: torch.Tensor,
        t: torch.Tensor,
        t_prev: torch.Tensor,
        labels: torch.Tensor,
        labels_uncond: Optional[torch.Tensor] = None,
        w: float = 0.0,
        eta: float = 0.0
    ) -> torch.Tensor:
        """Single DDIM sampling step."""
        stability_eps = get_eps(x_t.dtype)

        alpha_bar_t = extract(self.alphas_bar, t, x_t.shape)
        alpha_bar_t_prev = (
            extract(self.alphas_bar, t_prev, x_t.shape)
            if t_prev[0] >= 0
            else torch.ones_like(alpha_bar_t)
        )

        eps = model(x_t, t, labels)
        if w > 0 and labels_uncond is not None:
            eps_uncond = model(x_t, t, labels_uncond)
            eps = (1.0 + w) * eps - w * eps_uncond
        eps = clamp_prediction(eps)

        sqrt_alpha_bar_t = safe_sqrt(alpha_bar_t, eps=stability_eps)
        sqrt_one_minus_alpha_bar_t = safe_sqrt(1.0 - alpha_bar_t, eps=stability_eps)
        x0_pred = (x_t - sqrt_one_minus_alpha_bar_t * eps) / sqrt_alpha_bar_t
        x0_pred = torch.clamp(x0_pred, -1.0, 1.0)

        sqrt_alpha_bar_t_prev = safe_sqrt(alpha_bar_t_prev, eps=stability_eps)
        sqrt_one_minus_alpha_bar_t_prev = safe_sqrt(1.0 - alpha_bar_t_prev, eps=stability_eps)

        # sigma computation (eta=0 → deterministic)
        if eta == 0:
            sigma = torch.zeros_like(alpha_bar_t)
        else:
            one_minus_alpha_t = torch.clamp(1.0 - alpha_bar_t, min=stability_eps)
            one_minus_alpha_t_prev = torch.clamp(1.0 - alpha_bar_t_prev, min=stability_eps)
            alpha_ratio = torch.clamp(alpha_bar_t / torch.clamp(alpha_bar_t_prev, min=stability_eps), max=1.0)
            sigma_sq = torch.clamp((one_minus_alpha_t_prev / one_minus_alpha_t) * (1.0 - alpha_ratio), min=0.0)
            sigma = eta * safe_sqrt(sigma_sq)

        direction = safe_sqrt(1.0 - alpha_bar_t_prev - sigma ** 2, eps=stability_eps) * eps
        x_prev = sqrt_alpha_bar_t_prev * x0_pred + direction

        if eta > 0:
            noise = torch.randn_like(x_t)
            x_prev = x_prev + sigma * noise

        return x_prev

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        shape: tuple,
        labels: torch.Tensor,
        labels_uncond: torch.Tensor,
        w: float = 1.0,
        ddim_steps: int = 50,
        device: torch.device = None,
        verbose: bool = True
    ) -> torch.Tensor:
        """Generate samples using DDIM."""
        if device is None:
            device = next(model.parameters()).device

        x_T = torch.randn(shape, device=device)
        x_t = x_T

        step_size = self.T // ddim_steps
        timesteps = list(range(self.T - 1, -1, -step_size))[:ddim_steps]

        iterator = tqdm(timesteps, desc="Sampling") if verbose else timesteps

        for i, t_val in enumerate(iterator):
            t = torch.full((shape[0],), t_val, dtype=torch.long, device=device)
            t_prev_val = timesteps[i + 1] if i + 1 < len(timesteps) else -1
            t_prev = torch.full((shape[0],), t_prev_val, dtype=torch.long, device=device)

            x_t = self.ddim_step(model, x_t, t, t_prev, labels, labels_uncond, w=w, eta=0.0)

        return (x_t + 1.0) / 2.0  # [-1,1] → [0,1]
