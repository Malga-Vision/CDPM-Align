import math
import torch
from torch import nn
from torch.nn import functional as F
from torch import einsum
from einops import rearrange, reduce
from functools import partial

import torch.utils.checkpoint


def drop_connect(x, drop_ratio):
    keep_ratio = 1.0 - drop_ratio
    mask = torch.empty([x.shape[0], 1, 1, 1], dtype=x.dtype, device=x.device)
    mask.bernoulli_(p=keep_ratio)
    x.div_(keep_ratio)
    x.mul_(mask)
    return x


class WeightStandardizedConv2d(nn.Conv2d):
    """
    Weight standardization for better training with GroupNorm.
    https://arxiv.org/abs/1903.10520
    """
    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        weight = self.weight
        # Normalize weights to mean=0, var=1 across spatial dims
        mean = reduce(weight, 'o ... -> o 1 1 1', 'mean')
        var = reduce(weight, 'o ... -> o 1 1 1', partial(torch.var, unbiased=False))
        normalized_weight = (weight - mean) * (var + eps).rsqrt()
        
        return F.conv2d(
            x, normalized_weight, self.bias,
            self.stride, self.padding, self.dilation, self.groups
        )

class TimeEmbedding(nn.Module):
    def __init__(self, T, d_model, dim):
        assert d_model % 2 == 0
        super().__init__()
        emb = torch.arange(0, d_model, step=2) / d_model * math.log(10000)
        emb = torch.exp(-emb)
        pos = torch.arange(T).float()
        emb = pos[:, None] * emb[None, :]
        assert list(emb.shape) == [T, d_model // 2]
        emb = torch.stack([torch.sin(emb), torch.cos(emb)], dim=-1)
        assert list(emb.shape) == [T, d_model // 2, 2]
        emb = emb.view(T, d_model)

        self.timembedding = nn.Sequential(
            nn.Embedding.from_pretrained(emb, freeze=False),
            nn.Linear(d_model, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, t):
        emb = self.timembedding(t)
        return emb


class ConditionalEmbedding(nn.Module):
    def __init__(self, num_labels, d_model, dim):
        assert d_model % 2 == 0
        super().__init__()
        self.condEmbedding = nn.Sequential(
            nn.Embedding(num_embeddings=num_labels + 1, embedding_dim=d_model, padding_idx=0),
            nn.Linear(d_model, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, label):
        emb = self.condEmbedding(label)
        return emb


class DownSample(nn.Module):
    """Lossless downsampling using pixel unshuffle + 1x1 conv."""
    def __init__(self, in_ch):
        super().__init__()
        # Pixel unshuffle rearranges (h, w) -> (h/2, w/2) with 4x channels
        # Then 1x1 conv reduces back to original channel count
        self.conv = nn.Conv2d(in_ch * 4, in_ch, 1)

    def forward(self, x, temb, cemb):
        # Rearrange: b c (h 2) (w 2) -> b (c 4) h w
        x = rearrange(x, 'b c (h p1) (w p2) -> b (c p1 p2) h w', p1=2, p2=2)
        x = self.conv(x)
        return x


class UpSample(nn.Module):
    """Nearest neighbor upsampling + conv (avoids checkerboard artifacts)."""
    def __init__(self, in_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv = nn.Conv2d(in_ch, in_ch, 3, stride=1, padding=1)

    def forward(self, x, temb, cemb):
        x = self.up(x)
        x = self.conv(x)
        return x


class LinearAttention(nn.Module):
    """
    Linear attention with O(n) complexity instead of O(n²).
    Includes numerical stability improvements for FP16 training.
    """
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            nn.GroupNorm(1, dim)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h=self.heads), qkv
        )
        
        # Numerical stability: subtract max before softmax to prevent overflow
        q = q - q.amax(dim=-2, keepdim=True).detach()
        k = k - k.amax(dim=-1, keepdim=True).detach()
        
        # Linear attention trick: softmax on different dims to avoid n×n matrix
        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)
        
        # Add epsilon to prevent underflow in FP16 (1e-6 vanishes in half-precision)
        q = q * self.scale + 1e-4
        
        # Compute context: O(d²) instead of O(n²)
        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)
        
        # Apply context: O(n×d) instead of O(n²)
        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h=self.heads, x=h, y=w)
        return self.to_out(out)


class FullAttention(nn.Module):
    """
    Full multi-head self-attention with O(n²) complexity.
    Only use at low resolutions (e.g., 16x16, 32x32) to avoid OOM.
    """
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h=self.heads), qkv
        )
        q = q * self.scale

        # Full attention: O(n²) - only safe at low resolution!
        sim = einsum('b h d i, b h d j -> b h i j', q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = einsum('b h i j, b h d j -> b h i d', attn, v)
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x=h, y=w)
        return self.to_out(out)


class PreNorm(nn.Module):
    """Wrapper that applies GroupNorm before the function."""
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.GroupNorm(1, dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


class Residual(nn.Module):
    """Wrapper that adds residual connection (for attention layers only)."""
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        # Simple residual - no extra args to avoid checkpoint issues
        return self.fn(x) + x



class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, tdim, dropout, attn='linear', attn_heads=4, dim_head=32, groups=8):
        """
        Improved ResBlock with FiLM conditioning, WeightStandardizedConv, and optional attention.
        
        Args:
            attn: 'linear' for LinearAttention (O(n), safe at any resolution),
                  'full' for FullAttention (O(n²), only for low-res),
                  False/None for no attention
            attn_heads: number of attention heads (default 4 for better representation)
            dim_head: dimension per attention head (default 32, higher = more capacity)
            groups: number of groups for GroupNorm (default 8 for stability)
        """
        super().__init__()
        # First block: GroupNorm -> SiLU -> WeightStandardizedConv
        self.norm1 = nn.GroupNorm(groups, in_ch)
        self.act1 = nn.SiLU()
        self.conv1 = WeightStandardizedConv2d(in_ch, out_ch, 3, stride=1, padding=1)
        
        # FiLM conditioning: generates scale and shift for both time and condition
        # Output is 2x because we split into (scale, shift)
        self.temb_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(tdim, out_ch * 2),
        )
        self.cond_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(tdim, out_ch * 2),
        )
        
        # Second block: GroupNorm -> SiLU -> Dropout -> WeightStandardizedConv
        self.norm2 = nn.GroupNorm(groups, out_ch)
        self.act2 = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = WeightStandardizedConv2d(out_ch, out_ch, 3, stride=1, padding=1)
        
        # Shortcut connection
        if in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, 1, stride=1, padding=0)
        else:
            self.shortcut = nn.Identity()
        
        # Choose attention type - use PreNorm but not Residual (we handle residual in forward)
        if attn == 'linear':
            self.attn = PreNorm(out_ch, LinearAttention(out_ch, heads=attn_heads, dim_head=dim_head))
            self.has_attn = True
        elif attn == 'full':
            self.attn = PreNorm(out_ch, FullAttention(out_ch, heads=attn_heads, dim_head=dim_head))
            self.has_attn = True
        else:
            self.attn = nn.Identity()
            self.has_attn = False

    def forward(self, x, temb, cemb):
        # First block
        h = self.norm1(x)
        h = self.act1(h)
        h = self.conv1(h)
        
        # FiLM conditioning for time embedding
        # Clamp scales to prevent FP16 overflow (critical for mixed precision!)
        if temb is not None:
            temb_out = self.temb_proj(temb)[:, :, None, None]
            t_scale, t_shift = temb_out.chunk(2, dim=1)
            t_scale = torch.clamp(t_scale, -4.0, 4.0)  # Prevent extreme scales
            h = h * (1 + t_scale) + t_shift
        
        # FiLM conditioning for class/conditional embedding
        if cemb is not None:
            cemb_out = self.cond_proj(cemb)[:, :, None, None]
            c_scale, c_shift = cemb_out.chunk(2, dim=1)
            c_scale = torch.clamp(c_scale, -4.0, 4.0)  # Prevent extreme scales
            h = h * (1 + c_scale) + c_shift
        
        # Second block
        h = self.norm2(h)
        h = self.act2(h)
        h = self.dropout(h)
        h = self.conv2(h)
        
        # Residual connection
        h = h + self.shortcut(x)
        
        # Apply attention with residual
        if self.has_attn:
            h = h + self.attn(h)
        
        return h


class UNet(nn.Module):
    def __init__(self, T, num_labels, ch, ch_mult, num_res_blocks, dropout, img_channels=3, 
                 attn_heads=4, dim_head=32, use_linear_attn=True):
        """
        Memory-efficient UNet with LinearAttention for high-resolution training.
        
        Args:
            attn_heads: number of attention heads (default 4 for better representation learning)
            dim_head: dimension per attention head (default 32, higher = more capacity but more memory)
            use_linear_attn: if True, uses LinearAttention (O(n)) at all resolutions,
                           if False, no attention except full attention at middle (memory saving mode)
        """
        super().__init__()
        tdim = ch * 4
        self.time_embedding = TimeEmbedding(T, ch, tdim)
        self.cond_embedding = ConditionalEmbedding(num_labels, ch, tdim)
        self.img_channels = img_channels
        self.use_linear_attn = use_linear_attn
        self.attn_heads = attn_heads
        self.head = nn.Conv2d(img_channels, ch, kernel_size=3, stride=1, padding=1)
        self.downblocks = nn.ModuleList()
        chs = [ch]  # record output channel when downsample for upsample
        now_ch = ch
        
        # Build down blocks with LinearAttention at ALL resolutions (memory efficient!)
        attn_mode = 'linear' if use_linear_attn else False
        print(f"Building UNet with {'LinearAttention' if use_linear_attn else 'NO attention'} in down/up blocks")
        
        for i, mult in enumerate(ch_mult):
            out_ch = ch * mult
            for _ in range(num_res_blocks):
                # Use LinearAttention at all resolutions (O(n) - safe!)
                self.downblocks.append(ResBlock(in_ch=now_ch, out_ch=out_ch, tdim=tdim, 
                                               dropout=dropout, attn=attn_mode, attn_heads=attn_heads, 
                                               dim_head=dim_head))
                now_ch = out_ch
                chs.append(now_ch)
            if i != len(ch_mult) - 1:
                self.downblocks.append(DownSample(now_ch))
                chs.append(now_ch)

        # Middle blocks: use FullAttention (O(n²) but at lowest resolution so it's safe)
        self.middleblocks = nn.ModuleList([
            ResBlock(now_ch, now_ch, tdim, dropout, attn='full', attn_heads=attn_heads, dim_head=dim_head),
            ResBlock(now_ch, now_ch, tdim, dropout, attn=False),
        ])
        
        # Upsampling blocks with LinearAttention for symmetric architecture
        self.upblocks = nn.ModuleList()
        for i, mult in reversed(list(enumerate(ch_mult))):
            out_ch = ch * mult
            for _ in range(num_res_blocks + 1):
                # Use LinearAttention in upsampling too (memory efficient!)
                self.upblocks.append(ResBlock(in_ch=chs.pop() + now_ch, out_ch=out_ch, tdim=tdim, 
                                             dropout=dropout, attn=attn_mode, attn_heads=attn_heads, 
                                             dim_head=dim_head))
                now_ch = out_ch
            if i != 0:
                self.upblocks.append(UpSample(now_ch))
        assert len(chs) == 0

        self.tail = nn.Sequential(
            nn.GroupNorm(32, now_ch),
            nn.SiLU(),
            nn.Conv2d(now_ch, img_channels, 3, stride=1, padding=1)
        )
 

    def forward(self, x, t, labels, use_checkpoint: bool = False):
        # Timestep and label embeddings
        temb = self.time_embedding(t)
        cemb = self.cond_embedding(labels)

        # Head
        h = self.head(x)
        hs = [h]

        # Downsampling
        for layer in self.downblocks:
            if isinstance(layer, ResBlock) and use_checkpoint:
                h = torch.utils.checkpoint.checkpoint(layer, h, temb, cemb, use_reentrant=False)
            else:
                h = layer(h, temb, cemb)
            hs.append(h)

        # Middle
        for layer in self.middleblocks:
            if use_checkpoint:
                h = torch.utils.checkpoint.checkpoint(layer, h, temb, cemb, use_reentrant=False)
            else:
                h = layer(h, temb, cemb)

        # Upsampling
        for layer in self.upblocks:
            if isinstance(layer, ResBlock):
                h = torch.cat([h, hs.pop()], dim=1)
                if use_checkpoint:
                    h = torch.utils.checkpoint.checkpoint(layer, h, temb, cemb, use_reentrant=False)
                else:
                    h = layer(h, temb, cemb)
            else:
                h = layer(h, temb, cemb)

        # Tail
        h = self.tail(h)

        assert not hs, "Skip connections list (hs) should be empty after upsampling"
        return h


if __name__ == '__main__':
    batch_size = 8
    model = UNet(
        T=1000, num_labels=10, ch=128, ch_mult=[1, 2, 2, 2],
        num_res_blocks=2, dropout=0.1)
    x = torch.randn(batch_size, 3, 32, 32)
    t = torch.randint(1000, size=[batch_size])
    labels = torch.randint(10, size=[batch_size])
    y = model(x, t, labels)
    print(y.shape)

