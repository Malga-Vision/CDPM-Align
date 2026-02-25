import os
import tempfile
import torch
import random
import numpy as np
from prettytable import PrettyTable
import json


def safe_save(state, filename):
    """Safely save a model state to a file using atomic rename."""
    temp_filename = None
    try:
        directory = os.path.dirname(filename)
        with tempfile.NamedTemporaryFile(delete=False, dir=directory) as tmp_file:
            temp_filename = tmp_file.name
            torch.save(state, temp_filename)
        os.replace(temp_filename, filename)
    except Exception as e:
        print(f"Error during safe save: {e}")
        if temp_filename and os.path.exists(temp_filename):
            os.remove(temp_filename)


def set_seed(seed):
    """Set the random seed for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True
    random.seed(seed)
    np.random.seed(seed)

def count_parameters(model, show_table=True, show_frozen=False):
    """Count model parameters with optional detailed table."""
    table = PrettyTable(['Module', 'Parameters', 'Trainable'])
    total_params = 0
    trainable_params = 0
    frozen_params = 0
    
    for name, parameter in model.named_parameters():
        params = parameter.numel()
        total_params += params
        
        if parameter.requires_grad:
            trainable_params += params
            table.add_row([name, f"{params:,}", "✓"])
        else:
            frozen_params += params
            if show_frozen:
                table.add_row([name, f"{params:,}", "✗"])
    
    if show_table:
        print(table)
        print(f"\nTotal Parameters: {total_params:,} ({total_params/1e6:.2f}M)")
        print(f"Trainable: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
        print(f"Frozen: {frozen_params:,} ({frozen_params/1e6:.2f}M)")
    
    return table, total_params, trainable_params, frozen_params


def count_trainable_parameters(model):
    """Count only trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_all_parameters(model):
    """Count all parameters in a model (trainable + frozen)."""
    return sum(p.numel() for p in model.parameters())


def get_model_parameters_info(model, model_name="Model", show_details=False):
    """Get comprehensive parameter information for a model."""
    _, total, trainable, frozen = count_parameters(model, show_table=show_details, show_frozen=show_details)
    
    return {
        'model_name': model_name,
        'total_params': total,
        'trainable_params': trainable,
        'frozen_params': frozen,
        'total_M': total / 1e6,
        'trainable_M': trainable / 1e6,
        'frozen_M': frozen / 1e6
    }


def clean_cache():
    """Clean up the GPU cache to free memory."""
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

def save_model(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    save_dir: str,
    epoch: int,
    best_val_loss: float = None,
    epochs_without_improvement: int = 0,
    history: dict = None,
    is_best: bool = False,
    filename: str = None,
    iteration: int = None,
    loss: float = None
) -> str:
    """Save model checkpoint with all necessary components."""
    os.makedirs(save_dir, exist_ok=True)
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
        'epoch': epoch,
        'best_val_loss': best_val_loss,
        'epochs_without_improvement': epochs_without_improvement,
        'history': history or {},
        'iteration': iteration,
        'loss': loss
    }
    
    # Generate filename if not provided
    if filename is None:
        if is_best:
            filename = "best_checkpoint.pt"
        else:
            filename = f"checkpoint_epoch{epoch}"
            if iteration is not None:
                filename += f"_iter{iteration}"
            filename += ".pt"
    
    checkpoint_path = os.path.join(save_dir, filename)
    safe_save(checkpoint, checkpoint_path)
    
    # Always save best model separately
    if is_best:
        safe_save(checkpoint, os.path.join(save_dir, "best_checkpoint.pt"))
    
    return checkpoint_path


def load_model(
    load_path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer = None,
    scheduler: torch.optim.lr_scheduler._LRScheduler = None,
    device: torch.device = None
) -> dict:
    """Load model checkpoint and restore training state."""
    checkpoint = torch.load(load_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler is not None and checkpoint.get('scheduler_state_dict') is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    data = {
        'epoch': checkpoint.get('epoch', 0) + 1,
        'best_val_loss': checkpoint.get('best_val_loss', float('inf')),
        'epochs_without_improvement': checkpoint.get('epochs_without_improvement', 0),
        'history': checkpoint.get('history', {})
    }

    del checkpoint
    return data


class EarlyStopping:
    """Early stops training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=10, delta=0, save_dir=None, verbose=True, filename=None):
        self.patience = patience
        self.counter = 0
        self.best_val_loss = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.delta = delta
        self.save_dir = save_dir
        self.verbose = verbose
        self.filename = filename

    def __call__(self, val_loss, model, optimizer, scheduler, epoch, iteration=None, history=None):

        if self.best_val_loss is None or val_loss < self.best_val_loss - self.delta:
            self.best_val_loss = val_loss
            self.counter = 0
            if self.verbose:
                print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')
            self.val_loss_min = val_loss
            
            # Save the best model
            save_model(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                save_dir=self.save_dir,
                epoch=epoch,
                best_val_loss=val_loss,
                epochs_without_improvement=self.counter,
                history=history,
                is_best=True,
                filename=self.filename,
                iteration=iteration,
                loss=val_loss
            )
        else:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True

def get_pretrain_type_from_path(architecture: str, pretrained_path, encoder_weights=None) -> str:
    """Classify pretrained weight source from architecture name and weight path.

    Returns one of: 'cdpm_pretrained', 'ddpm_pretrained', an SSL method name,
    'ssl_pretrained', 'imagenet', 'no_pretrain', or 'random'.
    """
    arch = architecture.lower()
    if arch in ('cdpm', 'cdpm_unet'):
        return 'cdpm_pretrained' if pretrained_path else 'no_pretrain'
    if arch in ('ddpm', 'ddpm_unet'):
        return 'ddpm_pretrained' if pretrained_path else 'no_pretrain'
    if pretrained_path:
        ssl_methods = ["mocov3", "mocov2", "moco", "simclrv2", "simclr",
                       "dino", "barlow_twins", "byol", "simsiam", "swav"]
        path_lower = pretrained_path.lower()
        found = next((m for m in ssl_methods
                      if f'/{m}/' in path_lower or f'/{m}_' in path_lower), None)
        return found or 'ssl_pretrained'
    if encoder_weights == 'imagenet':
        return 'imagenet'
    return 'random'


def load_config_with_overrides(json_path: str, cli_overrides: dict) -> dict:
    """Load a flat JSON config and apply CLI overrides (non-None values only)."""
    if not json_path or not os.path.exists(json_path):
        raise FileNotFoundError(f"Config file not found: {json_path}")
    with open(json_path) as f:
        config = json.load(f)
    for key, value in cli_overrides.items():
        if value is not None:
            config[key] = value
    return config


def load_and_update_config_from_args(config_path, args):
    """Load config from JSON and update with command-line arguments."""
    try:
        with open(config_path) as f:
            config = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading config file: {e}")
        config = {}

    args_dict = vars(args)
    
    arg_to_config_map = {
        # Common
        'gpu':          ('environment', 'gpu'),
        'seed':         ('environment', 'seed'),
        'dataset':      ('dataset', 'name'),
        'output_dir':   ('environment', 'output_dir'),
        'num_samples': ('dataset', 'training_samples'),
        'resume':       ('training', 'resume'),

        # Pretraining SSL
        'model':        ('model', 'architecture'),
        'backbone':     ('model', 'encoder'),

        # Distillation
        'teacher_path': ('model', 'teacher', 'path'),
        'student':      ('model', 'student', 'name'),
        'epochs':       ('training', 'epochs'),

    }
    
    for key, value in args_dict.items():
        if key == 'config' or value is None:
            continue
            
        if isinstance(value, str):
            if '.' in value and value.replace('.', '', 1).isdigit():
                value = float(value)
            elif value.isdigit():
                value = int(value)
            elif value.lower() in ('true', 'yes'):
                value = True
            elif value.lower() in ('false', 'no'):
                value = False
        if key in arg_to_config_map:
            config_section, config_key = arg_to_config_map[key]
            if config_key:
                if config_section not in config:
                    config[config_section] = {}
                config[config_section][config_key] = value
            else:
                config[config_section] = value
        else:
            config[key] = value

    if 'output_dir' in args_dict and args_dict['output_dir'] is not None:
        if 'environment' not in config:
            config['environment'] = {}
        config['environment']['output_dir'] = args_dict['output_dir']

    if 'environment' in config and 'output_dir' in config['environment']:
        os.makedirs(config['environment']['output_dir'], exist_ok=True)
        
    return config

def print_config_params(config):
    from pprint import pformat
    print("----------------------------------------- CONFIG PARAMS -----------------------------------------")
    print("\n" + pformat(config, indent=4, width=120))
    print("------------------------------------------------------------------------------------------------")

def get_samples_subdirectory(config):
    """Generate a standardized samples subdirectory string from config."""
    training_samples = config['dataset'].get('training_samples', 'all')
    task_type = config.get('task_type', '').lower()
    
    if training_samples == 'all':
        return "samples_all"
    else:
        return f"samples_{training_samples}_total"

def create_experiment_output_path(config):
    """Create a standardized output directory path for experiments."""
    base_output_dir = config['environment']['output_dir']
    dataset_name = config['dataset']['name']
    
    batch_size = config['dataloader']['batch_size']
    grad_acc = config['dataloader'].get('grad_accumulation', 1)
    lr = config['training']['learning_rate']
    loss_function = config['training']['loss_function']
    scheduler = config['training']['scheduler']
    
    task_type = config.get('task_type', 'downstream')
    is_ddpm = 'beta_schedule' in config.get('model', {})
    is_ssl = task_type == 'pretraining' and not is_ddpm
    
    if is_ssl:
        ssl_method = config['model']['architecture']
        encoder = config['model']['encoder']
        use_smp = config['model'].get('use_smp_encoder', False)
        
        lr_str = f"{lr:.3f}".rstrip('0').rstrip('.') if lr >= 0.001 else f"{lr:.0e}"
        encoder_str = f"smp_{encoder}" if use_smp else encoder
        training_params = f"bs{batch_size}x{grad_acc}_lr{lr_str}"
        model_params = f"{ssl_method}_{encoder_str}"
        
        output_path = os.path.join(
            base_output_dir,
            dataset_name,
            model_params,
            training_params
        )
    elif is_ddpm:
        timesteps = config['model']['beta_schedule']['train']['n_timestep']
        schedule = config['model']['beta_schedule']['train']['schedule']
        use_ema = config['model'].get('use_ema', False)
        
        lr_str = f"{lr:.3f}".rstrip('0').rstrip('.') if lr >= 0.001 else f"{lr:.0e}"
        training_params = f"bs{batch_size}x{grad_acc}_lr{lr_str}"
        model_params = f"unet_t{timesteps}_{schedule}"
        training_info = f"{loss_function}_{'ema' if use_ema else 'no_ema'}"
        samples_subdir = get_samples_subdirectory(config)
        
        output_path = os.path.join(
            base_output_dir,
            dataset_name,
            model_params,
            training_params,
            training_info,
            samples_subdir
        )
    else:
        architecture = config['model']['architecture']
        encoder = config['model'].get('encoder')

        encoder_weights = config['model'].get('encoder_weights')
        pretrained_path = config['model'].get('pretrained_path')

        weights = get_pretrain_type_from_path(architecture, pretrained_path, encoder_weights)

        samples_subdir = get_samples_subdirectory(config)
        lr_str = f"{lr:.3f}".rstrip('0').rstrip('.') if lr >= 0.001 else f"{lr:.0e}"
        training_params = f"bs{batch_size}x{grad_acc}_lr{lr_str}"
        training_info = f"{loss_function}_{scheduler}" if scheduler else loss_function
        model_params = f"{architecture}_{encoder}" if encoder else architecture
        
        output_path = os.path.join(
            base_output_dir,
            dataset_name,
            model_params,
            weights,
            training_params,
            training_info,
            samples_subdir
        )
    
    return output_path
