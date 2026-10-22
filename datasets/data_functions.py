"""
Streamlined utility functions for dataset and dataloader creation with unified dataset structure.
"""

import os
import json

from torch.utils.data import DataLoader

# Import unified dataset classes
from .chest_dataset import ChestDataset
from .nih_chest_dataset import NIHChestDataset
from .hand_dataset import HandDataset
from .cephalo_dataset import CephaloDataset
from .combined_dataset import CombinedDataset

# Dataset mappings - unified datasets support multiple tasks through task_type parameter
DATASET_MAPPINGS = {
    'pretraining': {
        # Unified datasets (use task_type='pretraining')
        'chest': lambda **kwargs: ChestDataset(task_type='pretraining', **kwargs),
        'nih': lambda **kwargs: NIHChestDataset(task_type='pretraining', **kwargs),
        'hand': lambda **kwargs: HandDataset(task_type='pretraining', **kwargs),
        'cephalo': lambda **kwargs: CephaloDataset(task_type='pretraining', **kwargs),
        'combined': lambda **kwargs: CombinedDataset(**{k: v for k, v in kwargs.items() if k not in ('task_type', 'channels')}),
        'combined_landmarks': lambda **kwargs: CombinedDataset(**{k: v for k, v in kwargs.items() if k not in ('task_type',)}),
    },
    'landmarks': {
        # Unified datasets (use task_type='landmarks')
        'chest': lambda **kwargs: ChestDataset(task_type='landmarks', **kwargs),
        'hand': lambda **kwargs: HandDataset(task_type='landmarks', **kwargs),
        'cephalo': lambda **kwargs: CephaloDataset(task_type='landmarks', **kwargs),
    },
}

def get_dataset(dataset_name, root_dir, phase, image_size, channels, task_type=None, **kwargs):
    """
    Get the appropriate dataset based on name and task type.
    
    Supports both unified datasets (single class for all tasks) and legacy datasets (separate classes).
    """
    dataset_name = dataset_name.lower()

    if task_type == 'unlabeled':
        task_type = 'pretraining'

    task_type = 'pretraining' if task_type is None else task_type.lower()

    if isinstance(image_size, int):
        image_size = (image_size, image_size)

    if task_type not in DATASET_MAPPINGS:
        raise ValueError(f"Unknown task type: {task_type}. Must be one of: {list(DATASET_MAPPINGS.keys())}")

    task_datasets = DATASET_MAPPINGS[task_type]
    if dataset_name not in task_datasets:
        raise ValueError(f"Dataset '{dataset_name}' not available for task '{task_type}'")

    dataset_factory = task_datasets[dataset_name]

    return dataset_factory(root_dir=root_dir, phase=phase, image_size=image_size, channels=channels, **kwargs)

def create_datasets(config, use_train_val=False):
    dataset_name = config['dataset']['name']
    root_dir = config['dataset']['path']
    image_size = config['dataset']['image_size']
    channels = config['dataset']['image_channels']
    training_samples = config['dataset'].get('training_samples', 'all')
    extra_args = config['dataset'].get('extra_args', {})

    task_type = config.get('task_type', None)

    if task_type and task_type.lower() == 'landmarks':
        sigma = config['dataset'].get('sigma', None)
        if sigma is not None:
            extra_args['sigma'] = sigma

    # Support for combined datasets (datasets list in config)
    if 'datasets' in config['dataset']:
        extra_args['datasets'] = config['dataset']['datasets']

    datasets = {}

    if use_train_val:
        phases = ['train_val', 'test']
    else:
        phases = ['train', 'val', 'test']
    
    for phase in phases:
        datasets[phase] = get_dataset(
            dataset_name=dataset_name,
            root_dir=root_dir,
            phase=phase,
            image_size=image_size,
            channels=channels,
            task_type=task_type,
            **extra_args
        )
      # Limit training samples if specified
    train_phase = 'train_val' if use_train_val else 'train'
    
    if training_samples != 'all' and hasattr(datasets[train_phase], 'image_files'):
        num_samples = int(training_samples)
        if len(datasets[train_phase].image_files) >= num_samples:
            datasets[train_phase].image_files = datasets[train_phase].image_files[:num_samples]
            print(f"Limited {train_phase} dataset to {num_samples} samples")

        indexed_files_path = os.path.join(config.get('environment', {}).get('output_dir', '.'),
                                          f"{datasets[train_phase].phase}_image_files.json")
        with open(indexed_files_path, 'w') as f:
            json.dump(datasets[train_phase].image_files, f)
        print(f"Saved indexed image files for {train_phase} to {indexed_files_path}")
        
    return datasets


def create_dataloaders(datasets, config=None, **kwargs):
    if config is not None:
        batch_size = config['dataloader']['batch_size']
        pin_memory = config['dataloader'].get('pin_memory', True)
        num_workers = config['dataloader'].get('num_workers', 4)
        drop_last = config['dataloader'].get('drop_last', False)
    else:
        batch_size = kwargs.get('batch_size', 8)
        pin_memory = kwargs.get('pin_memory', True)
        num_workers = kwargs.get('num_workers', 4)
        drop_last = kwargs.get('drop_last', False)

    dataloaders = {}
    for phase, dataset in datasets.items():
        if dataset is None:
            dataloaders[phase] = None
            continue

        dataloader_args = {
            'batch_size': batch_size,
            'shuffle': phase.startswith('train'),
            'num_workers': num_workers,
            'pin_memory': pin_memory,
            'drop_last': drop_last if phase.startswith('train') else False
        }

        dataloaders[phase] = DataLoader(dataset, **dataloader_args)

    return dataloaders


def setup_data_from_config(config, use_train_val=False):
    """One-step function to set up datasets and dataloaders from config."""
    datasets = create_datasets(config, use_train_val)
    return create_dataloaders(datasets, config)

if __name__ == "__main__":
    # Example usage
    print("\n==== Testing data_functions.py ====\n")

    test_config = {
        'dataset': {
            'name': 'chest',
            'path': '/path/to/datasets',
            'image_size': (256, 256),
            'image_channels': 1,
            'training_samples': 10,
        },
        'dataloader': {
            'batch_size': 4,
            'num_workers': 2,
            'pin_memory': True,
        },
        'task_type': 'landmarks',
    }

    try:
        dataloaders = setup_data_from_config(test_config, use_train_val=False)
        print(f"Dataloaders created: {', '.join(dataloaders.keys())}")
    except Exception as e:
        print(f"Setup failed: {e}")