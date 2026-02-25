import os
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from functools import partial
from einops import rearrange, reduce
from einops.layers.torch import Rearrange
from .unet_blocks import *


# ------------------------------------------------------------------------
#                               Unet Model with Time Embeddings
# ------------------------------------------------------------------------

class Unet(nn.Module):
    def __init__(
        self,
        dim,
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        channels=3,
        self_condition=False,
        resnet_block_groups=4,
        att_res=32,
        att_heads=4,
        parallelism=False,  # Flag for model parallelism
        device=None  # Device to use when parallelism is False
    ):
        super().__init__()

        # Parallelism flag
        self.parallelism = parallelism

        # Determine available devices
        available_gpus = torch.cuda.device_count()
        if not self.parallelism:
            # Use the device provided (from config/training)
            self.device_0 = torch.device(device) if device is not None else torch.device('cuda:0' if available_gpus > 0 else 'cpu')
            self.device_1 = self.device_0
        else:
            # Use the training device and the next available GPU (if any)
            self.device_0 = torch.device(device) if device is not None else torch.device('cuda:0' if available_gpus > 0 else 'cpu')
            # Pick the next GPU if available, otherwise fallback to device_0
            if available_gpus > 1:
                # If device_0 is cuda:N, use cuda:(N+1) if available
                if self.device_0.type == 'cuda':
                    next_gpu = (self.device_0.index + 1) if (self.device_0.index is not None and self.device_0.index + 1 < available_gpus) else 0
                    self.device_1 = torch.device(f'cuda:{next_gpu}')
                else:
                    self.device_1 = self.device_0
            else:
                self.device_1 = self.device_0

        # determine dimensions
        self.channels = channels
        self.self_condition = self_condition
        input_channels = channels * (2 if self_condition else 1)

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(input_channels, init_dim, 1, padding=0)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups=resnet_block_groups)

        # time embeddings
        time_dim = dim * 4

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        Residual(
                            PreNorm(dim_in, LinearAttention(dim_in, att_heads, att_res))
                        ),
                        Downsample(dim_in, dim_out)
                        if not is_last
                        else nn.Conv2d(dim_in, dim_out, 3, padding=1),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(
            PreNorm(mid_dim, Attention(mid_dim, att_heads, att_res))
        )
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(
                nn.ModuleList(
                    [
                        block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                        block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                        Residual(
                            PreNorm(
                                dim_out, LinearAttention(dim_out, att_heads, att_res)
                            )
                        ),
                        Upsample(dim_out, dim_in)
                        if not is_last
                        else nn.Conv2d(dim_out, dim_in, 3, padding=1),
                    ]
                )
            )

        self.out_dim = default(out_dim, channels)

        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim=time_dim)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)

        # Move the layers to the appropriate devices based on available GPUs
        if self.parallelism and available_gpus > 1:
            # Move half of the downsampling and upsampling layers to device_1
            half_down_index = len(self.downs) // 2
            half_up_index = len(self.ups) // 2

            for i in range(half_down_index, len(self.downs)):
                self.downs[i] = self.downs[i].to(self.device_1)

            self.mid_block1 = self.mid_block1.to(self.device_1)
            self.mid_attn = self.mid_attn.to(self.device_1)
            self.mid_block2 = self.mid_block2.to(self.device_1)

            for i in range(half_up_index, len(self.ups)):
                self.ups[i] = self.ups[i].to(self.device_1)

    def forward(self, x, time=None, x_self_cond=None, checkpointing=True):
        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim=1)

        x = self.init_conv(x)
        r = x.clone()
        
        # Only compute time embedding if time is provided
        if time is not None:
            t = self.time_mlp(time)
        else:
            t = None

        h = []

        if self.parallelism:
            # Move the input to device_1 if using parallelism
            x = x.to(self.device_1)
            t = t.to(self.device_1) if exists(t) else None
            r = r.to(self.device_1)
            h = [h_i.to(self.device_1) for h_i in h]
            
        # Forward pass with optional checkpointing
        if checkpointing:  
            for i, (block1, block2, attn, downsample) in enumerate(self.downs):
                # Use parallelism to decide which GPU to use
                if self.parallelism and i >= len(self.downs) // 2:
                    x = x.to(self.device_1)
                    t = t.to(self.device_1) if exists(t) else None
                else:
                    x = x.to(self.device_0)
                    t = t.to(self.device_0) if exists(t) else None
                
                # Ensure both x and block1 are on the same device before applying block1
                block1 = block1.to(x.device)
                x = checkpoint(block1, x, t, use_reentrant=False)

                h.append(x)

                block2 = block2.to(x.device)  # Ensure block2 is on the same device
                x = checkpoint(block2, x, t, use_reentrant=False)

                attn = attn.to(x.device)  # Ensure attn is on the same device
                x = checkpoint(attn, x, use_reentrant=False)
                h.append(x)
                
                downsample = downsample.to(x.device)  # Ensure downsample is on the same device
                x = checkpoint(downsample, x, use_reentrant=False)

            if self.parallelism:
                x = x.to(self.device_1)
                t = t.to(self.device_1) if exists(t) else None
                h = [h_i.to(self.device_1) for h_i in h]

            self.mid_block1 = self.mid_block1.to(x.device)
            x = checkpoint(self.mid_block1, x, t, use_reentrant=False)

            self.mid_attn = self.mid_attn.to(x.device)
            x = checkpoint(self.mid_attn, x, use_reentrant=False)

            self.mid_block2 = self.mid_block2.to(x.device)
            x = checkpoint(self.mid_block2, x, t, use_reentrant=False)

            for i, (block1, block2, attn, upsample) in enumerate(self.ups):
                x = torch.cat((x, h.pop()), dim=1)

                # Switch between devices during upsampling
                if self.parallelism and i >= len(self.ups) // 2:
                    x = x.to(self.device_1)
                    t = t.to(self.device_1) if exists(t) else None
                    h = [h_i.to(self.device_1) for h_i in h]
                else:
                    x = x.to(self.device_0)
                    t = t.to(self.device_0) if exists(t) else None
                    h = [h_i.to(self.device_0) for h_i in h]

                block1 = block1.to(x.device)
                x = checkpoint(block1, x, t, use_reentrant=False)
                x = torch.cat((x, h.pop()), dim=1)

                block2 = block2.to(x.device)
                x = checkpoint(block2, x, t, use_reentrant=False)

                attn = attn.to(x.device)
                x = checkpoint(attn, x, use_reentrant=False)

                upsample = upsample.to(x.device)
                x = checkpoint(upsample, x, use_reentrant=False)

        else:
            # Similar logic for non-checkpointing case
            for i, (block1, block2, attn, downsample) in enumerate(self.downs):
                if self.parallelism and i >= len(self.downs) // 2:
                    x = x.to(self.device_1)
                    t = t.to(self.device_1) if exists(t) else None
                    h = [h_i.to(self.device_1) for h_i in h]
                else:
                    x = x.to(self.device_0)
                    t = t.to(self.device_0) if exists(t) else None
                    h = [h_i.to(self.device_0) for h_i in h]

                block1 = block1.to(x.device)
                x = block1(x, t)

                h.append(x)

                block2 = block2.to(x.device)
                x = block2(x, t)

                attn = attn.to(x.device)
                x = attn(x)

                downsample = downsample.to(x.device)
                x = downsample(x)

            if self.parallelism:
                x = x.to(self.device_1)
                t = t.to(self.device_1) if exists(t) else None
                h = [h_i.to(self.device_1) for h_i in h]

            self.mid_block1 = self.mid_block1.to(x.device)
            x = self.mid_block1(x, t)

            self.mid_attn = self.mid_attn.to(x.device)
            x = self.mid_attn(x)

            self.mid_block2 = self.mid_block2.to(x.device)
            x = self.mid_block2(x, t)

            for i, (block1, block2, attn, upsample) in enumerate(self.ups):
                x = torch.cat((x, h.pop()), dim=1)

                if self.parallelism and i >= len(self.ups) // 2:
                    x = x.to(self.device_1)
                    t = t.to(self.device_1) if exists(t) else None
                    h = [h_i.to(self.device_1) for h_i in h]
                else:
                    x = x.to(self.device_0)
                    t = t.to(self.device_0) if exists(t) else None
                    h = [h_i.to(self.device_0) for h_i in h]

                block1 = block1.to(x.device)
                x = block1(x, t)

                x = torch.cat((x, h.pop()), dim=1)

                block2 = block2.to(x.device)
                x = block2(x, t)

                attn = attn.to(x.device)
                x = attn(x)

                upsample = upsample.to(x.device)
                x = upsample(x)

        x = x.to(self.device_0)
        r = r.to(x.device)
        x = torch.cat((x, r), dim=1)
        t = t.to(x.device) if exists(t) else None

        x = self.final_res_block(x, t)
        x = self.final_conv(x)
        return x


def setup_ddpm_unet_model(image_size=256, image_channels=3, num_classes=68, device="cuda:0", pretrained_path=None):
    
    model = Unet(
        dim=512,
        channels=image_channels,
        dim_mults=[1,2,4,8],
        self_condition=True,
        resnet_block_groups=4,
        att_heads=4,
        att_res=32,
        device=device
    ).to(device)  
            
    if pretrained_path is not None and os.path.exists(pretrained_path):
        checkpoint = torch.load(pretrained_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print(f"DDPM pretrained model loaded from {pretrained_path}")

        del checkpoint
    
    # change the number of output channels of the final convolutional layer
    model.final_conv = nn.Conv2d(model.final_conv.in_channels, num_classes, 1)
    
    return model

