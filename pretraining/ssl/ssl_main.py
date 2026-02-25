"""
Main SSL training pipeline.

Supports three SSL methods used as baselines in the paper:
- DINO
- MoCo v3
- SimCLR v2

Backbones:
- torchvision models (resnet18, resnet34, resnet50, resnet101, etc.)
- segmentation_models_pytorch (smp) encoders for medical imaging
"""

# Standard library imports
import os
import sys

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

import signal
import tempfile

# Third-party imports
import torch
import torch.nn as nn
import numpy as np
import torchvision
from tqdm import tqdm

# SMP for medical imaging backbones
try:
    import segmentation_models_pytorch as smp
    from segmentation_models_pytorch import encoders as smp_encoders
    SMP_AVAILABLE = True
except ImportError:
    SMP_AVAILABLE = False
    print("Warning: segmentation_models_pytorch not available. Install with: pip install segmentation_models_pytorch")

# Local imports
from utils.utils import set_seed, load_and_update_config_from_args, print_config_params
from utils.setup import setup_logging, print_system_info, setup_device, setup_optimizer
from utils.main_script_utils import create_standard_parser, initialize_experiment, add_pretraining_args
from datasets.data_functions import setup_data_from_config
from pretraining.ssl.ssl_train import train_ssl_model
from pretraining.ssl.builders import DINO, MoCoV3, SimCLRv2





class SMPBackboneWrapper(nn.Module):
    """
    Wrapper to use SMP encoders as backbones for SSL.
    SMP encoders output a list of features at different scales.
    This wrapper takes the final feature map and applies global average pooling.
    """
    def __init__(self, encoder, feature_size):
        super().__init__()
        self.encoder = encoder
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.feature_size = feature_size
    
    def forward(self, x):
        # SMP encoder returns list of features at different scales
        features = self.encoder(x)
        # Take the last (deepest) feature map
        x = features[-1]
        # Global average pooling
        x = self.global_pool(x)
        x = x.flatten(1)
        return x


def build_backbone(backbone_name, image_channels, feature_size=None, use_smp=False):
    """
    Build a backbone model based on the configuration.
    
    Args:
        backbone_name (str): Name of the backbone architecture
        image_channels (int): Number of input image channels
        feature_size (int, optional): Feature size for the output
        use_smp (bool): Whether to use segmentation_models_pytorch encoder
        
    Returns:
        tuple: (backbone, feature_size)
    """
    # Use SMP encoder if requested
    if use_smp and SMP_AVAILABLE:
        return build_smp_backbone(backbone_name, image_channels)
    
        
    elif backbone_name == "resnet101":
        backbone = torchvision.models.resnet101(weights=None)
        if image_channels == 1:
            backbone.conv1 = torch.nn.Conv2d(image_channels, 64, kernel_size=(7, 7), 
                                             stride=(2, 2), padding=(3, 3), bias=False)
        feature_size = backbone.fc.in_features
        backbone.fc = torch.nn.Identity()
        
    else:
        # Try to use SMP encoder as fallback
        if SMP_AVAILABLE:
            print(f"Unknown backbone: {backbone_name}, trying as SMP encoder...")
            return build_smp_backbone(backbone_name, image_channels)
        else:
            raise ValueError(f"Unknown backbone: {backbone_name}")
    
    return backbone, feature_size


def build_smp_backbone(encoder_name, image_channels, encoder_weights=None):
    """
    Build a backbone from segmentation_models_pytorch encoders.
    
    Args:
        encoder_name (str): Name of the SMP encoder (e.g., 'resnet101', 'efficientnet-b4')
        image_channels (int): Number of input channels
        encoder_weights (str, optional): Pretrained weights ('imagenet' or None)
        
    Returns:
        tuple: (backbone, feature_size)
    """
    if not SMP_AVAILABLE:
        raise ImportError("segmentation_models_pytorch is not installed. Install with: pip install segmentation_models_pytorch")
    
    # Get the encoder from SMP
    encoder = smp_encoders.get_encoder(
        name=encoder_name,
        in_channels=image_channels,
        depth=5,
        weights=encoder_weights
    )
    
    # Get the output feature size (from the last encoder stage)
    # SMP encoders have out_channels attribute
    feature_size = encoder.out_channels[-1]
    
    # Wrap the encoder for SSL use
    backbone = SMPBackboneWrapper(encoder, feature_size)
    
    print(f"Built SMP backbone: {encoder_name}, in_channels={image_channels}, feature_size={feature_size}")
    
    return backbone, feature_size


def build_ssl_model(ssl_method, backbone, feature_size, image_size, config=None):
    """
    Build a self-supervised learning model based on the method.
    
    Args:
        ssl_method (str): Name of the SSL method
        backbone (nn.Module): Backbone network
        feature_size (int): Size of the feature vector
        image_size (int or tuple): Size of the input image
        config (dict, optional): Configuration dictionary with SSL-specific parameters
        
    Returns:
        nn.Module: SSL model
    """
    # Get SSL-specific parameters from config if available
    ssl_params = {}
    if config and 'model' in config and 'ssl_params' in config['model']:
        ssl_params = config['model']['ssl_params'].get(ssl_method, {})
    
    # Handle image_size - some methods need tuple, some need int
    if isinstance(image_size, (list, tuple)):
        image_size_int = image_size[0]  # Use first dimension for square
        image_size_tuple = tuple(image_size)
    else:
        image_size_int = image_size
        image_size_tuple = (image_size, image_size)
    
    # Determine mean/std for grayscale vs RGB
    image_channels = config.get('dataset', {}).get('image_channels', 1) if config else 1
    if image_channels == 1:
        mean = (0.5,)
        std = (0.5,)
    else:
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    
    if ssl_method == "mocov3":
        model = MoCoV3(
            backbone, feature_size, 
            projection_dim=ssl_params.get('projection_dim', 256), 
            hidden_dim=ssl_params.get('hidden_dim', 2048), 
            temperature=ssl_params.get('temperature', 0.5), 
            m=ssl_params.get('m', 0.999), 
            image_size=image_size_tuple,
            mean=mean, std=std
        )
    elif ssl_method == "simclrv2":
        model = SimCLRv2(
            backbone, feature_size, 
            projection_dim=ssl_params.get('projection_dim', 128), 
            temperature=ssl_params.get('temperature', 0.5), 
            image_size=image_size_tuple,
            mean=mean, std=std
        )
    elif ssl_method == "dino":
        model = DINO(
            backbone, feature_size, 
            projection_dim=ssl_params.get('projection_dim', 256), 
            hidden_dim=ssl_params.get('hidden_dim', 2048), 
            bottleneck_dim=ssl_params.get('bottleneck_dim', 256), 
            temp_s=ssl_params.get('temp_s', 0.1), 
            temp_t=ssl_params.get('temp_t', 0.5), 
            m=ssl_params.get('m', 0.5), 
            lamda=ssl_params.get('lamda', 0.996), 
            num_crops=ssl_params.get('num_crops', 6),
            image_size=image_size_int,
            mean=mean, std=std
        )
    else:
        raise ValueError(
            f"Unknown SSL method: {ssl_method}. "
            "Available: mocov3, simclrv2, dino"
        )
    
    return model


# ------------------------------------------------------------------------
#                               MAIN
# ------------------------------------------------------------------------
def main():
    """Main function to run SSL training"""
    # Parse arguments and initialize experiment
    parser = create_standard_parser("SSL Pretraining", "./configs/pretraining/ssl_dino.json")
    add_pretraining_args(parser, task='ssl')
    args = parser.parse_args()

    # Initialize experiment (handles config, device, seed, output dir, etc.)
    config, device, output_dir = initialize_experiment(args, task_type='pretraining')

    # Load dataset
    loaders = setup_data_from_config(config, use_train_val=True)
    train_loader = loaders['train_val']

    # Create directory for saving models
    save_model_path = os.path.join(output_dir, "models")
    os.makedirs(save_model_path, exist_ok=True)

    # Determine if using SMP encoder
    use_smp = config['model'].get('use_smp_encoder', False)
    
    # Initialize backbone for self-supervised learning method
    backbone, feature_size = build_backbone(
        config['model']['encoder'], 
        config['dataset']['image_channels'],
        use_smp=use_smp
    )
    
    print(f"Built backbone: {config['model']['encoder']}")
    print(f"  Use SMP: {use_smp}")
    print(f"  Feature size: {feature_size}")
    print(f"  Image channels: {config['dataset']['image_channels']}")
    
    # Initialize SSL model
    model = build_ssl_model(
        config['model']['architecture'], 
        backbone, 
        feature_size, 
        config['dataset']['image_size'],
        config=config
    )
    model = model.to(device)
    
    print(f"\nSSL Method: {config['model']['architecture']}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M")
    
    # Initialize optimizer
    optimizer = setup_optimizer(
        config['training']['optimizer'], 
        model.parameters(), 
        config['training']['learning_rate'],
        weight_decay=config['training'].get('weight_decay', 1e-4)
    )
    
    # Train the model
    train_ssl_model(model, optimizer, train_loader, device, config)


if __name__ == "__main__":
    main()