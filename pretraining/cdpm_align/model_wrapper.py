"""
UNet Wrapper with Feature Extraction for CDPM-align.

Wraps the pretrained CDPM UNet to add feature extraction capabilities
for the multi-scale guidance alignment training.
"""

import torch
import torch.nn as nn
import torch.utils.checkpoint

from ..cdpm.model import UNet, ResBlock, DownSample, UpSample


class UNetWithFeatureExtraction(nn.Module):
    """
    Wrapper around pretrained UNet that extracts intermediate features.

    Feature extraction points (same as UNetWithFeatures in cdpm_dino):
    - enc_1_4: After 2nd downblock level (1/4 spatial resolution)
    - enc_1_8: After 3rd downblock level (1/8 spatial resolution)
    - bottleneck: After middle blocks
    - dec_1_8: After 1st upblock level (1/8 spatial resolution)
    """

    FEATURE_LAYERS = ["enc_1_4", "enc_1_8", "bottleneck", "dec_1_8"]

    def __init__(self, base_model: UNet, ch: int, ch_mult: list, num_res_blocks: int):
        """
        Args:
            base_model: Pretrained CDPM UNet
            ch: Base channel count
            ch_mult: Channel multipliers per level
            num_res_blocks: Number of res blocks per level
        """
        super().__init__()
        self.base = base_model
        self.ch = ch
        self.ch_mult = ch_mult
        self.num_res_blocks = num_res_blocks

        # Compute feature extraction indices
        self._compute_feature_indices()

    def _compute_feature_indices(self):
        """Compute indices for feature extraction in down/up paths."""
        idx = 0
        self.feature_indices = {}

        for i in range(len(self.ch_mult)):
            for _ in range(self.num_res_blocks):
                idx += 1
            level_end_resblocks = idx - 1

            if i != len(self.ch_mult) - 1:
                idx += 1  # DownSample

            if i == 1:
                self.feature_indices['enc_1_4'] = level_end_resblocks
            elif i == 2:
                self.feature_indices['enc_1_8'] = level_end_resblocks

        self.feature_indices['bottleneck'] = 'middle'

        # dec_1_8 is after first up level
        up_idx = 0
        for i, _ in reversed(list(enumerate(self.ch_mult))):
            for _ in range(self.num_res_blocks + 1):
                up_idx += 1
            level_end = up_idx - 1

            if i == len(self.ch_mult) - 1:
                self.feature_indices['dec_1_8'] = ('up', level_end)
                break

            if i != 0:
                up_idx += 1

    def get_feature_dims(self) -> dict:
        """Returns dict mapping layer names to channel dimensions."""
        return {
            'enc_1_4': self.ch * self.ch_mult[1],
            'enc_1_8': self.ch * self.ch_mult[2],
            'bottleneck': self.ch * self.ch_mult[-1],
            'dec_1_8': self.ch * self.ch_mult[-1],
        }

    def forward(self, x, t, labels, return_features=False, use_checkpoint=False):
        """
        Forward pass with optional feature extraction.

        Args:
            x: Input noisy image [B, C, H, W]
            t: Timesteps [B]
            labels: Integer class labels [B] (0=unconditional, 1+=class)
            return_features: If True, return (eps_pred, features_dict)
            use_checkpoint: Use gradient checkpointing
        """
        # Get embeddings from base model
        temb = self.base.time_embedding(t)
        cemb = self.base.cond_embedding(labels)

        h = self.base.head(x)
        hs = [h]

        features = {} if return_features else None

        # Downsampling
        for idx, layer in enumerate(self.base.downblocks):
            if isinstance(layer, ResBlock) and use_checkpoint:
                h = torch.utils.checkpoint.checkpoint(layer, h, temb, cemb, use_reentrant=False)
            else:
                h = layer(h, temb, cemb)
            hs.append(h)

            if return_features:
                if idx == self.feature_indices['enc_1_4']:
                    features['enc_1_4'] = h
                elif idx == self.feature_indices['enc_1_8']:
                    features['enc_1_8'] = h

        # Middle
        for layer in self.base.middleblocks:
            if use_checkpoint:
                h = torch.utils.checkpoint.checkpoint(layer, h, temb, cemb, use_reentrant=False)
            else:
                h = layer(h, temb, cemb)

        if return_features:
            features['bottleneck'] = h

        # Upsampling
        dec_1_8_idx = self.feature_indices['dec_1_8'][1]

        up_idx = 0
        for layer in self.base.upblocks:
            if isinstance(layer, ResBlock):
                h = torch.cat([h, hs.pop()], dim=1)
                if use_checkpoint:
                    h = torch.utils.checkpoint.checkpoint(layer, h, temb, cemb, use_reentrant=False)
                else:
                    h = layer(h, temb, cemb)

                if return_features and up_idx == dec_1_8_idx:
                    features['dec_1_8'] = h

                up_idx += 1
            else:
                h = layer(h, temb, cemb)

        h = self.base.tail(h)

        assert not hs, "Skip connections list should be empty"

        if return_features:
            return h, features
        return h

    def load_pretrained(self, checkpoint_path: str, device: torch.device):
        """Load pretrained weights from checkpoint."""
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if 'model_state_dict' in ckpt:
            self.base.load_state_dict(ckpt['model_state_dict'])
        else:
            self.base.load_state_dict(ckpt)
        print(f"Loaded pretrained weights from {checkpoint_path}")


def create_model_from_pretrained(
    checkpoint_path: str,
    T: int,
    num_classes: int,
    ch: int,
    ch_mult: list,
    num_res_blocks: int,
    dropout: float,
    img_channels: int,
    attn_heads: int,
    dim_head: int,
    use_linear_attn: bool,
    device: torch.device
) -> UNetWithFeatureExtraction:
    """
    Create model wrapper and load pretrained weights.

    Args:
        checkpoint_path: Path to pretrained checkpoint
        T, num_classes, ch, etc.: Model architecture params
        device: Target device

    Returns:
        UNetWithFeatureExtraction with loaded weights
    """
    # Create base UNet
    base_model = UNet(
        T=T,
        num_labels=num_classes,
        ch=ch,
        ch_mult=ch_mult,
        num_res_blocks=num_res_blocks,
        dropout=dropout,
        img_channels=img_channels,
        attn_heads=attn_heads,
        dim_head=dim_head,
        use_linear_attn=use_linear_attn
    ).to(device)

    # Wrap with feature extraction
    model = UNetWithFeatureExtraction(base_model, ch, ch_mult, num_res_blocks).to(device)

    # Load pretrained weights
    model.load_pretrained(checkpoint_path, device)

    return model


if __name__ == "__main__":
    print("\n=== Testing UNetWithFeatureExtraction ===")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ch = 64
    ch_mult = [1, 2, 2, 2]
    num_res_blocks = 2

    # Create base model
    base_model = UNet(
        T=500,
        num_labels=3,
        ch=ch,
        ch_mult=ch_mult,
        num_res_blocks=num_res_blocks,
        dropout=0.15,
        img_channels=1,
        attn_heads=4,
        dim_head=32,
        use_linear_attn=True
    ).to(device)

    # Wrap it
    model = UNetWithFeatureExtraction(base_model, ch, ch_mult, num_res_blocks).to(device)

    print(f"ch_mult: {model.ch_mult}")
    print(f"num_res_blocks: {model.num_res_blocks}")
    print(f"Feature indices: {model.feature_indices}")
    print(f"Feature dims: {model.get_feature_dims()}")

    # Test forward
    batch_size = 2
    x = torch.randn(batch_size, 1, 128, 128, device=device)
    t = torch.randint(0, 500, (batch_size,), device=device)
    labels = torch.randint(0, 4, (batch_size,), device=device)

    # Without features
    eps_pred = model(x, t, labels, return_features=False)
    print(f"\nWithout features: {eps_pred.shape}")

    # With features
    eps_pred, features = model(x, t, labels, return_features=True)
    print(f"\nWith features:")
    print(f"  eps_pred: {eps_pred.shape}")
    for name, feat in features.items():
        print(f"  {name}: {feat.shape}")

    print("\n=== Test Complete ===")
