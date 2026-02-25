""" 
Diffusion model architecture modified from the original code  
https://github.com/dome272/Diffusion-Models-pytorch/tree/main 
and
https://huggingface.co/blog/annotated-diffusion
"""

import torch
import torch.nn as nn
from tqdm import tqdm
import logging
from .unet_blocks import *
from torch.optim import Adam, AdamW, SGD
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy as np
import copy
import os
import torch.nn.functional as F
from .ddpm_unet import *

# ------------------------------------------------------------------------
#                               Noise Schedule
# ------------------------------------------------------------------------

def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 0.0001, 0.9999)

def linear_beta_schedule(timesteps, beta_start=0.0001, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, timesteps)


def quadratic_beta_schedule(timesteps, beta_start=0.0001, beta_end=0.02):
    return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2


def sigmoid_beta_schedule(timesteps, beta_start=0.0001, beta_end=0.02):
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start

# ------------------------------------------------------------------------
#                               DDPM Model
# ------------------------------------------------------------------------

class DDPM:
    def __init__(
        self,
        image_size: int = 256,
        channels: int = 3,
        device = "cuda",
        lr: float = 1e-4,
        optimizer: str = "adam",
        timesteps: int = 1000,
        beta_schedule: str = "linear",
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        unet_channels: list = [1, 2, 4, 8],
        unet_self_condition: bool = True,
        unet_att_res: int = 32,
        unet_att_heads: int = 4,
        unet_res_blocks: int = 4,
        use_ema: bool = False,
    ):

        self.timesteps = timesteps
        self.device = device
        self.image_size = image_size
        self.channels = channels
        self.lr = lr

        self.beta_start = beta_start
        self.beta_end = beta_end

        # define beta schedule
        self.beta = self.prepare_noise_schedule(beta_schedule=beta_schedule).to(device)
        
        # define alphas
        self.alpha = 1.0 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

        # define model
        self.model = Unet(
            dim=image_size,
            channels=channels,
            dim_mults=unet_channels,
            self_condition=unet_self_condition,
            resnet_block_groups=unet_res_blocks,
            att_heads=unet_att_heads,
            att_res=unet_att_res,
            device=self.device
        )
        self.model.to(self.device)

        # define optimizer
        self.optimizer = self.prepare_optimizer(optimizer)
    
        # Initialize EMA
        if use_ema:
            self.ema = EMA(0.995)
            self.ema_model = copy.deepcopy(self.model).eval().requires_grad_(False).to(self.device)
        else:
            self.ema = None
            self.ema_model = None

    def update_ema(self):
        # Update the EMA model parameters
        self.ema.update_model_average(self.ema_model, self.model)
        
    def prepare_optimizer(self, optimizer):
        if optimizer == "adam":
            return Adam(self.model.parameters(), lr=self.lr)
        elif optimizer == "adamw":
            return AdamW(self.model.parameters(), lr=self.lr)
        elif optimizer == "sgd":
            return SGD(self.model.parameters(), lr=self.lr)
        else:
            raise NotImplementedError()
        
    def prepare_noise_schedule(self, beta_schedule):
        if beta_schedule == "linear":
            return linear_beta_schedule(
                timesteps=self.timesteps, beta_start=self.beta_start, beta_end=self.beta_end
            )
        elif beta_schedule == "cosine":
            return cosine_beta_schedule(timesteps=self.timesteps)
        elif beta_schedule == "quadratic":
            return quadratic_beta_schedule(
                timesteps=self.timesteps, beta_start=self.beta_start, beta_end=self.beta_end
            )
        elif beta_schedule == "sigmoid":
            return sigmoid_beta_schedule(
                timesteps=self.timesteps, beta_start=self.beta_start, beta_end=self.beta_end
            )
        else:
            raise NotImplementedError


    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        eps = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * eps, eps
    
    def noise_images_conditioned(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])
        eps = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * eps, eps

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.timesteps, size=(n,)).to(self.device)

    def p_losses(self, x, t, loss_type="l1"):
        
        x_t, noise = self.noise_images(x, t)
        predicted_noise = self.model(x_t, t)

        if loss_type == "l1":
            loss = F.l1_loss(noise, predicted_noise)
        elif loss_type == "l2":
            loss = F.mse_loss(noise, predicted_noise)
        elif loss_type == "huber":
            loss = F.smooth_l1_loss(noise, predicted_noise)
        else:
            raise NotImplementedError()

        return loss


    def sample(self, model, batch_size, timesteps=None, x_cond=None):
        with torch.no_grad():
            if timesteps is None:
                timesteps = self.timesteps

            if x_cond is None:
                x = torch.randn((batch_size, self.channels, self.image_size, self.image_size)).to(self.device)
            else:
                x,_ = self.noise_images_conditioned(x_cond, timesteps-1)

            for i in tqdm(reversed(range(1, timesteps)), desc="Sampling", total=timesteps-1, leave=False):
                t = (torch.ones(batch_size) * i).long().to(self.device)
                predicted_noise = model(x, t)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]

                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise

        return x
    
    @torch.no_grad()
    def save_noising_process_image(self, x_start, filename):

        images = [x_start[0:1]] # Select the first image from the batch
        timesteps_to_visualize = [i for i in range(0, self.timesteps, self.timesteps // 10)]
        
        for t in timesteps_to_visualize:
            t_tensor = torch.tensor([t], device=self.device)
            noised_image, _ = self.noise_images(x_start, t_tensor)
            images.append(noised_image[0:1]) # Select the first image from the batch

        # Create a grid of images
        image_grid = make_grid(torch.cat(images), nrow=len(timesteps_to_visualize) + 1, normalize=False)
        image_grid = image_grid.clamp(0, 1)

        # Convert to numpy array and transpose axes to HWC format for plotting
        np_image_grid = image_grid.cpu().numpy().transpose((1, 2, 0))

        # Plot and save the image
        plt.figure(figsize=(len(images) * 2, 2))
        plt.imshow(np_image_grid)
        plt.axis('off')
        plt.savefig(filename, bbox_inches='tight')
        plt.close()




# ------------------------------------------------------------------------
#                               EMA (Exponential Moving Average)
# ------------------------------------------------------------------------

class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.step = 0

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model, step_start_ema=2000):
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())




