"""SSL backbone builders used as baselines in the paper (DINO, MoCo v3, SimCLR v2)."""
from .dino import DINO
from .mocov3 import MoCoV3
from .simclrv2 import SimCLRv2

__all__ = ['DINO', 'MoCoV3', 'SimCLRv2']

