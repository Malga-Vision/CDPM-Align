import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


def extract(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    device = t.device
    out = torch.gather(v, index=t, dim=0).float().to(device)
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))


class GaussianDiffusionTrainer(nn.Module):
    def __init__(self, model, beta_1, beta_T, T):
        super().__init__()

        self.model = model
        self.T = T

        self.register_buffer(
            'betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer(
            'sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer(
            'sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))

    def forward(self, x_0, labels, use_checkpoint=False):
        """
        Algorithm 1 with numerical stability improvements for FP16.
        """
        t = torch.randint(self.T, size=(x_0.shape[0], ), device=x_0.device)
        noise = torch.randn_like(x_0)
        
        # Clamp noise to prevent extreme values that can cause NaN in FP16
        noise = torch.clamp(noise, -5.0, 5.0)
        
        x_t = extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 + \
              extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise
        
        # Get model prediction
        pred = self.model(x_t, t, labels, use_checkpoint=use_checkpoint)
        
        # Clamp predictions to prevent extreme values (important for FP16!)
        pred = torch.clamp(pred, -10.0, 10.0)
        
        # Compute loss with numerical stability
        loss = F.mse_loss(pred, noise, reduction='none')
        
        # Clamp loss to prevent extreme values from causing NaN gradients
        loss = torch.clamp(loss, 0.0, 50.0)
        
        return loss


class GaussianDiffusionSampler(nn.Module):
    def __init__(self, model, beta_1, beta_T, T, w = 0.):
        super().__init__()

        self.model = model
        self.T = T
        self.w = w

        self.register_buffer('betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:T]

        self.register_buffer('alphas_bar', alphas_bar)
        self.register_buffer('sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer('sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))
        self.register_buffer('coeff1', torch.sqrt(1. / alphas))
        self.register_buffer('coeff2', self.coeff1 * (1. - alphas) / torch.sqrt(1. - alphas_bar))
        self.register_buffer('posterior_var', self.betas * (1. - alphas_bar_prev) / (1. - alphas_bar))

    def predict_xt_prev_mean_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return extract(self.coeff1, t, x_t.shape) * x_t - extract(self.coeff2, t, x_t.shape) * eps

    def p_mean_variance(self, x_t, t, labels):
        var = torch.cat([self.posterior_var[1:2], self.betas[1:]])
        var = extract(var, t, x_t.shape)
        x_double = torch.cat([x_t, x_t], dim=0)
        t_double = torch.cat([t, t], dim=0)
        labels_double = torch.cat([labels, torch.zeros_like(labels)], dim=0)
        eps_both = self.model(x_double, t_double, labels_double)
        eps, eps_uncond = eps_both.chunk(2, dim=0)
        eps = (1. + self.w) * eps - self.w * eps_uncond
        xt_prev_mean = self.predict_xt_prev_mean_from_eps(x_t, t, eps=eps)
        return xt_prev_mean, var

    def ddim_step(self, x_t, t, t_prev, labels, eta=0.0):
        """
        DDIM sampling step with proper coefficients for arbitrary timestep spacing.
        eta=0: deterministic (DDIM), eta=1: stochastic (DDPM-like)
        """
        # Get alpha values
        alpha_bar_t = extract(self.alphas_bar, t, x_t.shape)
        alpha_bar_t_prev = extract(self.alphas_bar, t_prev, x_t.shape) if t_prev[0] >= 0 else torch.ones_like(alpha_bar_t)
        
        # Predict noise with classifier-free guidance (batched conditional + unconditional)
        x_double = torch.cat([x_t, x_t], dim=0)
        t_double = torch.cat([t, t], dim=0)
        labels_double = torch.cat([labels, torch.zeros_like(labels)], dim=0)
        eps_both = self.model(x_double, t_double, labels_double)
        eps, eps_uncond = eps_both.chunk(2, dim=0)
        eps = (1. + self.w) * eps - self.w * eps_uncond
        
        # DDIM update formula
        # x0 prediction
        x0_pred = (x_t - torch.sqrt(1 - alpha_bar_t) * eps) / torch.sqrt(alpha_bar_t)
        x0_pred = torch.clamp(x0_pred, -1, 1)  # Clamp for stability
        
        # Compute sigma for stochasticity control
        sigma = eta * torch.sqrt((1 - alpha_bar_t_prev) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_t_prev))
        
        # Direction pointing to x_t
        dir_xt = torch.sqrt(1 - alpha_bar_t_prev - sigma**2) * eps
        
        # DDIM sample
        x_t_prev = torch.sqrt(alpha_bar_t_prev) * x0_pred + dir_xt
        
        if eta > 0 and t_prev[0] > 0:
            noise = torch.randn_like(x_t)
            x_t_prev = x_t_prev + sigma * noise
            
        return x_t_prev

    def forward(self, x_t, labels, start_t=None, ddim_steps=None, eta=0.0):
        """
        Sampling with optional DDIM acceleration.
        Args:
            x_t: Initial noise
            labels: Class labels
            start_t: Starting timestep (default: T)
            ddim_steps: Number of DDIM steps (default: None = full DDPM)
            eta: DDIM stochasticity (0=deterministic, 1=DDPM-like)
        """
        if start_t is None:
            start_t = self.T

        if ddim_steps is not None and ddim_steps < start_t:
            # DDIM: use fewer steps with proper update formula
            timesteps = torch.linspace(start_t - 1, 0, ddim_steps + 1, dtype=torch.long).tolist()
            timesteps = [int(t) for t in timesteps]
            
            for i in tqdm(range(len(timesteps) - 1), desc='DDIM Sampling'):
                t_curr = timesteps[i]
                t_prev = timesteps[i + 1]
                
                if t_curr == 0:
                    continue
                    
                t = torch.full((x_t.shape[0],), t_curr, device=x_t.device, dtype=torch.long)
                t_prev_tensor = torch.full((x_t.shape[0],), max(t_prev, 0), device=x_t.device, dtype=torch.long)
                
                x_t = self.ddim_step(x_t, t, t_prev_tensor, labels, eta=eta)
                assert torch.isnan(x_t).int().sum() == 0, "nan in tensor."
        else:
            # Full DDPM sampling
            for time_step in tqdm(reversed(range(1, start_t)), desc='DDPM Sampling'):
                t = torch.full((x_t.shape[0],), time_step, device=x_t.device, dtype=torch.long)
                mean, var = self.p_mean_variance(x_t=x_t, t=t, labels=labels)
                if time_step > 1:
                    noise = torch.randn_like(x_t)
                else:
                    noise = 0
                x_t = mean + torch.sqrt(var) * noise
                assert torch.isnan(x_t).int().sum() == 0, "nan in tensor."
        
        return x_t
    