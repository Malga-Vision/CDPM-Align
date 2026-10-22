"""
Shared utilities for main experiment scripts to reduce boilerplate code.
"""

import os
import json
import argparse
from typing import Dict, Any, Optional, Tuple

from .utils import load_and_update_config_from_args, set_seed, print_config_params, create_experiment_output_path
from .setup import print_system_info, setup_device


def create_standard_parser(description: str = "Run experiment", default_config: str = None) -> argparse.ArgumentParser:
    """Create a standardized argument parser with common experiment arguments."""
    parser = argparse.ArgumentParser(description=description)
    
    parser.add_argument('-c', '--config', type=str, default=default_config, 
                       help='Path to config file')
    parser.add_argument('-g', '--gpu', type=int, 
                       help='GPU index')
    parser.add_argument('-s', '--seed', type=int, 
                       help='Random seed')
    parser.add_argument('--dataset', type=str, 
                       help='Dataset name')
    parser.add_argument('--output_dir', type=str, 
                       help='Output directory')
    parser.add_argument('-m', '--mode', type=str, default='train', 
                       choices=['train', 'test'], 
                       help='Mode to run the script in')
    
    return parser


def initialize_experiment(
    args: argparse.Namespace,
    task_type: str,
    create_model_dir: bool = True,
    output_dir_suffix: Optional[str] = None
) -> Tuple[Dict[str, Any], Any, str]:
    """Standard initialization for all experiment scripts. Returns (config, device, output_dir)."""
    config = load_and_update_config_from_args(args.config, args)
    config['task_type'] = task_type
    set_seed(config['environment']['seed'])
    device = setup_device(config['environment']['gpu'])
    print_system_info()
    print_config_params(config)

    output_dir = create_experiment_output_path(config)
    if output_dir_suffix:
        output_dir = os.path.join(output_dir, output_dir_suffix, f"seed{config['environment']['seed']}")
    
    os.makedirs(output_dir, exist_ok=True)
    config['environment']['output_dir'] = output_dir

    config_path = os.path.join(output_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"[ + ] Configuration saved to {config_path}")

    if create_model_dir:
        model_dir = os.path.join(output_dir, 'models')
        os.makedirs(model_dir, exist_ok=True)
    
    print(f"[ + ] Output directory: {output_dir}")
    
    return config, device, output_dir


def add_downstream_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add common arguments for downstream landmark detection tasks."""
    parser.add_argument('-n', '--num_samples', type=str, 
                       help='Number of training samples to use')
    parser.add_argument('--model_path', type=str, 
                       help='Path to model checkpoint for evaluation')
    return parser


def add_pretraining_args(parser: argparse.ArgumentParser, task: str = 'ddpm') -> argparse.ArgumentParser:
    """Add common arguments for pretraining tasks (DDPM, SSL)."""
    parser.add_argument('--resume', action='store_true', default=False,
                       help='Resume training from checkpoint')
    
    if task == 'ssl':
        parser.add_argument('--model', type=str, 
                           help='SSL method name')
        parser.add_argument('--backbone', type=str, 
                           help='Backbone model name')
    
    return parser

