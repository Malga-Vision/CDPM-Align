"""
Setup and configuration utilities.
"""

import time
import logging
import sys

import torch
import torch.nn as nn
def get_loss_function(loss_function_name: str, **kwargs) -> nn.Module:
    """Factory function to create loss functions by name."""
    loss_function_name = loss_function_name.lower()
    reduction = kwargs.get('reduction', 'mean')

    if loss_function_name in ['nll', 'negloglikelihood']:
        from landmark_detection.landmark_losses import CustomNLLLoss
        return CustomNLLLoss(reduction=reduction)
    elif loss_function_name in ['l1', 'mae']:
        return nn.L1Loss()
    elif loss_function_name in ['l2', 'mse']:
        return nn.MSELoss()
    elif loss_function_name == 'huber':
        return nn.HuberLoss()
    else:
        raise ValueError(f"Unknown loss function: {loss_function_name}")


def setup_loss_function(loss_function_name, **kwargs):
    return get_loss_function(loss_function_name, **kwargs)



def setup_scheduler(scheduler_name, optimizer, training_config=None):
    """Set up the learning rate scheduler based on the configuration.
    
    Args:
        scheduler_name: Name of the scheduler (or None/null for no scheduler)
        optimizer: PyTorch optimizer
        training_config: Training configuration dictionary with scheduler parameters
        
    Returns:
        Scheduler instance or None if scheduler_name is None/null
    """
    # Handle None or "null" scheduler
    if scheduler_name is None or (isinstance(scheduler_name, str) and scheduler_name.lower() in ['none', 'null', '']):
        return None
    
    scheduler_name = scheduler_name.lower()
    
    # Extract config parameters or use defaults
    if training_config is None:
        training_config = {}
    
    if scheduler_name == 'multisteplr':
        milestones = training_config.get('scheduler_milestones', [4, 8, 12, 15, 18, 21])
        gamma = training_config.get('scheduler_gamma', 0.25)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
    
    elif scheduler_name == 'reducelronplateau':
        factor = training_config.get('scheduler_factor', 0.5)
        patience_val = training_config.get('scheduler_patience', 4)
        min_lr = training_config.get('scheduler_min_lr', 1e-7)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=factor, patience=patience_val, min_lr=min_lr)
    
    elif scheduler_name == 'cosineannealinglr':
        t_max = training_config.get('scheduler_t_max', training_config.get('epochs', 50))
        eta_min = training_config.get('scheduler_eta_min', 1e-6)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=eta_min)
    
    elif scheduler_name == 'exponentiallr':
        gamma = training_config.get('scheduler_gamma', 0.95)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    
    elif scheduler_name in ['warmupcosine', 'cosineannealingwarmrestarts']:
        t_0 = training_config.get('scheduler_t_0', 50)
        t_mult = training_config.get('scheduler_t_mult', 2)
        eta_min = training_config.get('scheduler_eta_min', 1e-6)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=t_0, T_mult=t_mult, eta_min=eta_min
        )
    
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")
    
    return scheduler



def setup_optimizer(optimizer_name, model_parameters, lr, weight_decay=0.01):
    """Set up the optimizer based on the configuration."""
    optimizer_name = optimizer_name.lower()
    if optimizer_name == 'adam':
        optimizer = torch.optim.Adam(model_parameters, lr=lr)
    elif optimizer_name == 'sgd':
        optimizer = torch.optim.SGD(model_parameters, lr=lr, momentum=0.9)
    elif optimizer_name == 'adamw':
        optimizer = torch.optim.AdamW(model_parameters, lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    
    return optimizer

def setup_device(gpu_idx):

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        if gpu_idx is not None and gpu_idx >= 0:
            device = torch.device(f'cuda:{gpu_idx}' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device('cpu')

    print(f"Using device: {device}")

    return device

def setup_logging(base_dir):
    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = f"{base_dir}/{time_str}_experiment.log"

    logging.basicConfig(filename=log_file, format='%(message)s')
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)
    return logger, log_file


def print_system_info():
    print("\n----------------------------------------- SYSTEM INFO -----------------------------------------")
    print(f"Python version: {sys.version}")
    print(f"Pytorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"CUDNN version: {torch.backends.cudnn.version()}")
    
    # Check if MPS is available
    print(f"MPS available: {torch.backends.mps.is_available()}")
    print(f"MPS built: {torch.backends.mps.is_built()}")

    print("------------------------------------------------------------------------------------------------")
