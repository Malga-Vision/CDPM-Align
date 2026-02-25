"""
Pretraining modules for CDPM-align (generative) and SSL baselines.
"""

# CDPM base pretraining
from .cdpm.run import main as run_cdpm

# CDPM-align finetuning
from .cdpm_align.run_finetune import main as run_cdpm_align

# SSL baselines (DINO, MoCo v3, SimCLR v2)
from .ssl.ssl_train import train_ssl_model
from .ssl.builders import DINO, MoCoV3, SimCLRv2

__all__ = [
    'run_cdpm',
    'run_cdpm_align',
    'train_ssl_model',
    'DINO', 'MoCoV3', 'SimCLRv2',
]
