"""
Model Factory v2 - Refactored
=============================
Unified model creation factory utilizing registry patterns for modularity 
and reducing code duplication.
"""

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from segmentation_models_pytorch import encoders as smp_encoders
from typing import Dict, Any, Optional, List, Tuple, Type

# Import custom modules
from pretraining.ddpm_divia.ddpm_unet import Unet
from pretraining.cdpm.model import UNet as CDPMUNet, ResBlock

# ============================================================================
# Configuration & Constants
# ============================================================================

DEFAULT_DECODER_CHANNELS = [256, 128, 64, 32, 16]
RANDOM_WEIGHT_ALIASES = {'None', 'none', 'random', None}

# CDPM downstream adapter constants (see ablation in paper)
CDPM_DOWNSTREAM_TIMESTEP = 200   # fixed noise level used for feature extraction

# Registry mapping architecture names to SMP classes
SMP_ARCHITECTURES: Dict[str, Type[nn.Module]] = {
    'unet': smp.Unet,
}

# ============================================================================
# Helper Classes
# ============================================================================

class CDPMDownstreamAdapter(nn.Module):
    """Adapter for CDPM UNet in downstream tasks (unconditional mode, label=0)."""
    def __init__(self, cdpm_unet: nn.Module):
        super().__init__()
        self.cdpm_unet = cdpm_unet

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        device = x.device
        t = torch.full((batch_size,), CDPM_DOWNSTREAM_TIMESTEP, dtype=torch.long, device=device)
        labels = torch.zeros(batch_size, dtype=torch.long, device=device)
        return self.cdpm_unet(x, t, labels)


# ============================================================================
# Utilities
# ============================================================================

def _validate_config(config: Dict, required_keys: List[str], context: str = "") -> None:
    """Ensures nested keys exist in the config dictionary."""
    for key_path in required_keys:
        keys = key_path.split('.')
        value = config
        for key in keys:
            if not isinstance(value, dict) or key not in value:
                raise ValueError(f"[{context}] Missing required config key: '{key_path}'")
            value = value[key]

def _load_weights(model: nn.Module, path: str, device: torch.device, component: str = 'encoder') -> bool:
    """Safely loads weights into a model or specific component.

    Handles two legacy checkpoint formats automatically:
    - CDPM-align checkpoints that pickle a ``Config`` dataclass: retried with
      the dataclass injected so unpickling succeeds.
    - SSL checkpoints whose keys carry a ``backbone.`` prefix: the prefix is
      stripped before loading so SMP encoder weights load directly.
    """
    if not path:
        return False

    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        state_dict = checkpoint.get('model_state_dict', checkpoint)

        # Strip backbone. prefix from SSL checkpoints automatically
        if any(k.startswith('backbone.') for k in state_dict):
            state_dict = {k[len('backbone.'):]: v
                          for k, v in state_dict.items()
                          if k.startswith('backbone.')}
        
        # FIX: Check if component is a valid string before using hasattr
        if component and hasattr(model, component):
            target = getattr(model, component)
        else:
            # If component is None or doesn't exist, load into the root model
            target = model

        # Load state dict - some models (e.g., SMP encoders) return None instead of the tuple
        result = target.load_state_dict(state_dict, strict=False)
        
        # Handle result if available (may be None for some model types like SMP)
        if result is not None:
            missing, unexpected = result
            if missing:
                print(f"Missing keys when loading: {missing[:5]}..." if len(missing) > 5 else missing)
            if unexpected:
                print(f"Unexpected keys when loading: {unexpected[:5]}..." if len(unexpected) > 5 else unexpected)

        
        print(f"Loaded pretrained {component if component else 'weights'} from: {path}")
        del checkpoint
        return True
        
    except Exception as e:
        print(f"Failed to load pretrained {component}: {e}")
        raise e

def _get_smp_weights_name(weight_config: str) -> Optional[str]:
    """Normalizes weight configuration string for SMP."""
    return None if weight_config in RANDOM_WEIGHT_ALIASES else 'imagenet'


def _infer_cdpm_num_labels_from_checkpoint(checkpoint_path: str, device: torch.device) -> int:
    """
    Infer num_labels from a CDPM checkpoint by examining the conditioning embedding weight shape.

    The ConditionalEmbedding creates an embedding of shape [num_labels + 1, d_model] where
    index 0 is the padding index (unconditional). So num_labels = embedding_size - 1.
    """
    if not checkpoint_path:
        return 1  # Default

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = checkpoint.get('model_state_dict', checkpoint)

        # Look for the conditioning embedding weight
        cond_emb_key = 'cond_embedding.condEmbedding.0.weight'
        if cond_emb_key in state_dict:
            embedding_shape = state_dict[cond_emb_key].shape
            num_labels = embedding_shape[0] - 1  # Subtract 1 for padding_idx=0
            print(f"Inferred num_labels={num_labels} from checkpoint (embedding shape: {embedding_shape})")
            del checkpoint
            return num_labels

        del checkpoint
        print(f"Could not find conditioning embedding in checkpoint, using default num_labels=1")
        return 1

    except Exception as e:
        print(f"Warning: Could not infer num_labels from checkpoint: {e}")
        return 1

# ============================================================================
# Core Builders
# ============================================================================

def _build_ddpm_unet(config: Dict, device: torch.device) -> Unet:
    """Constructs the Custom DDPM U-Net."""
    dataset_cfg = config['dataset']
    
    # Handle both DDPM pretraining and downstream task configs
    if 'unet' in config['model']:
        # DDPM pretraining: config['model']['unet'] contains architecture params
        arch_cfg = config['model']['unet']
    else:
        # Downstream task: architecture is a string, use defaults
        arch_cfg = {}
    
    image_size = dataset_cfg['image_size']
    dim = max(image_size) if isinstance(image_size, (list, tuple)) else image_size

    return Unet(
        dim=dim,
        channels=dataset_cfg['image_channels'],
        dim_mults=arch_cfg.get('channel_mults', (1, 2, 4, 8)),
        self_condition=arch_cfg.get('self_condition', True),
        resnet_block_groups=arch_cfg.get('res_blocks', 8),
        att_heads=arch_cfg.get('num_head_channels', 4),
        att_res=arch_cfg.get('attn_res', 32),
        device=device
    ).to(device)


def _build_cdpm_unet(config: Dict, device: torch.device) -> CDPMDownstreamAdapter:
    """Constructs the Conditional DDPM U-Net (CDPM) wrapped for downstream tasks."""
    dataset_cfg = config['dataset']
    pretrained_path = config['model'].get('pretrained_path')

    # Handle both CDPM pretraining and downstream task configs
    if 'unet' in config['model']:
        # CDPM pretraining: config['model']['unet'] contains architecture params
        arch_cfg = config['model']['unet']
    else:
        # Downstream task: use default CDPM architecture
        arch_cfg = {
            'T': 500,
            'ch': 128,
            'ch_mult': [1, 2, 2, 2],
            'num_res_blocks': 2,
            'dropout': 0.1,
            'attn_heads': 4,
            'dim_head': 32,
            'use_linear_attn': True,
        }

    # Infer num_labels from checkpoint if available, otherwise use config or default
    if pretrained_path and 'num_labels' not in arch_cfg:
        num_labels = _infer_cdpm_num_labels_from_checkpoint(pretrained_path, device)
    else:
        num_labels = arch_cfg.get('num_labels', 1)

    cdpm_unet = CDPMUNet(
        T=arch_cfg.get('T', 500),
        num_labels=num_labels,
        ch=arch_cfg.get('ch', 128),
        ch_mult=arch_cfg.get('ch_mult', [1, 2, 2, 2]),
        num_res_blocks=arch_cfg.get('num_res_blocks', 2),
        dropout=arch_cfg.get('dropout', 0.1),
        img_channels=dataset_cfg['image_channels'],
        attn_heads=arch_cfg.get('attn_heads', 4),
        dim_head=arch_cfg.get('dim_head', 32),
        use_linear_attn=arch_cfg.get('use_linear_attn', True)
    ).to(device)

    return CDPMDownstreamAdapter(cdpm_unet).to(device)


def _build_smp_model(
    arch_name: str,
    encoder_name: str,
    encoder_weights: Optional[str],
    in_channels: int,
    num_classes: int,
    decoder_channels: Optional[List[int]] = None
) -> nn.Module:
    """Generic builder for Segmentation Models Pytorch architectures."""
    clean_name = arch_name.lower().replace('smp_', '')

    if clean_name not in SMP_ARCHITECTURES:
        raise ValueError(f"Unsupported SMP architecture: {clean_name}. "
                         f"Available: {list(SMP_ARCHITECTURES.keys())}")

    model_cls = SMP_ARCHITECTURES[clean_name]

    kwargs = {
        'encoder_name': encoder_name,
        'encoder_weights': encoder_weights,
        'in_channels': in_channels,
        'classes': num_classes,
        'decoder_channels': decoder_channels or DEFAULT_DECODER_CHANNELS,
    }

    return model_cls(**kwargs)

# ============================================================================
# Task Factories
# ============================================================================

def create_ddpm_pretraining_model(config: Dict, device: torch.device) -> nn.Module:
    """Factory for DDPM Pretraining."""
    print("Creating DDPM Pretraining Model...")
    return _build_ddpm_unet(config, device)


def _create_downstream_model(config: Dict, device: torch.device, num_outputs: int, task_name: str) -> nn.Module:
    """Shared internal factory for downstream landmark detection models."""
    _validate_config(config, ['model.architecture', 'dataset.image_channels'], context=task_name)

    arch = config['model']['architecture']

    # Special Case: DDPM U-Net used for downstream task
    if arch.lower() in ['ddpm_unet', 'ddpm']:
        print(f"[{task_name}] Using DDPM U-Net architecture.")
        model = _build_ddpm_unet(config, device)
    elif arch.lower() in ['cdpm_unet', 'cdpm']:
        print(f"[{task_name}] Using CDPM U-Net architecture.")
        model = _build_cdpm_unet(config, device)
    else:
        encoder = config['model'].get('encoder')
        if not encoder:
            raise ValueError(f"[{task_name}] 'model.encoder' is required for SMP architecture '{arch}'")
        print(f"[{task_name}] Using SMP architecture: {arch} with encoder={encoder}")
        # Standard Case: SMP Models
        model = _build_smp_model(
            arch_name=arch,
            encoder_name=encoder,
            encoder_weights=_get_smp_weights_name(config['model'].get('encoder_weights')),
            in_channels=config['dataset']['image_channels'],
            num_classes=num_outputs,
            decoder_channels=config['model'].get('decoder_channels')
        )
    # --- Load Pretrained Weights if specified ---
    if config['model'].get('pretrained_path'):
        if arch.lower() in ['ddpm_unet', 'ddpm']:
            _load_weights(model, config['model']['pretrained_path'], device, component=None)
        elif arch.lower() in ['cdpm_unet', 'cdpm']:
            # Load weights into the wrapped CDPM UNet
            _load_weights(model.cdpm_unet, config['model']['pretrained_path'], device, component=None)
        else:
            _load_weights(model, config['model']['pretrained_path'], device, component='encoder')

    # --- Adapt output layer for task-specific num_outputs ---
    if arch.lower() in ['cdpm_unet', 'cdpm']:
        # CDPM uses 'tail' instead of 'final_conv' - replace the last layer
        old_tail = model.cdpm_unet.tail
        model.cdpm_unet.tail = nn.Sequential(
            old_tail[0],  # GroupNorm
            old_tail[1],  # Swish
            nn.Conv2d(old_tail[2].in_channels, num_outputs, 3, stride=1, padding=1)
        )
    elif arch.lower() in ['ddpm_unet', 'ddpm']:
        # DDPM has final_conv
        model.final_conv = nn.Conv2d(model.final_conv.in_channels, num_outputs, 1)
    # SMP models are already created with the correct num_classes, no need to modify

    encoder = config['model'].get('encoder')
    print(f"[{task_name}] Created {arch}" + (f" encoder={encoder}" if encoder else ""))
    return model.to(device)


def create_landmarks_model(config: Dict, device: torch.device, num_landmarks: int) -> nn.Module:
    """Create a model for landmark detection."""
    return _create_downstream_model(config, device, num_landmarks, "Landmarks")