"""
Unified Chest X-ray dataset for all tasks (pretraining, landmarks).
"""

import os
import sys

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import numpy as np
from datasets.base_dataset import BaseDataset
from landmark_detection.landmark_utils import get_scale_factor
import albumentations as A
import cv2

_TRAIN_NUM = 195
_VAL_NUM = 34


class ChestDataset(BaseDataset):
    """
    Unified Chest X-ray dataset for:
    - Pretraining (unlabeled images)
    - Landmarks (landmark detection)
    """
    
    def __init__(
        self, 
        root_dir, 
        phase='train', 
        image_size=(512, 512), 
        channels=1,
        task_type='pretraining',
        sigma=5,  # For landmarks
        **kwargs
    ):
        super().__init__(
            root_dir=root_dir, 
            dataset_name='chest',
            phase=phase,
            image_size=image_size,
            channels=channels,
            task_type=task_type,
            sigma=sigma,
            **kwargs
        )
        
        self.sigma = sigma
        
        # Set paths
        self.image_dir = os.path.join(self.dataset_path, 'pngs')
        self.labels_dir = os.path.join(self.dataset_path, 'labels')
        
        # Get all files
        files = [i[:-4] for i in sorted(os.listdir(self.image_dir)) if i.endswith('.png')]

        # Split into train/val/test
        n = len(files)
        train_num, val_num = _TRAIN_NUM, _VAL_NUM
        test_num = n - train_num - val_num
        
        if phase == 'train':
            self.image_files = files[:train_num]
        elif phase in ['val', 'validate']:
            self.image_files = files[train_num:train_num+val_num]
        elif phase == 'train_val':
            self.image_files = files[:train_num+val_num]
        elif phase == 'test':
            self.image_files = files[-test_num:]
        elif phase == 'all':
            self.image_files = files
        else:
            raise ValueError(f"Unknown phase: {phase}")
        
        # Set number of landmarks (for landmarks task)
        if task_type == 'landmarks':
            self.num_landmarks = 6
        
        print(f"Loaded {len(self.image_files)} images for {self.dataset_name} - {phase} - {task_type}")
    
    def read_landmarks(self, image_name, original_size):
        """Read landmarks from the chest dataset."""
        path = os.path.join(self.labels_dir, image_name + '.txt')
        landmarks_coordinates = []
        ratios = []
        
        with open(path, 'r') as f:
            n = int(f.readline())
            for i in range(n):
                ratio = [float(i) for i in f.readline().split()]
                landmark = (ratio[0] * original_size[1], ratio[1] * original_size[0])
                landmarks_coordinates.append((landmark[0], landmark[1]))
                ratios.append(ratio)
                
        return landmarks_coordinates, np.array(ratios)
    
    def __getitem__(self, idx):
        image_name = self.image_files[idx]
        image_path = os.path.join(self.image_dir, f"{image_name}.png")

        image = self.read_image(image_path)

        if self.task_type == 'pretraining':
            transformed = self.get_transforms()(image=image)
            transformed_image = transformed['image']

            return {
                'name': image_name,
                'image': transformed_image.float()
            }
        elif self.task_type == 'landmarks':
            landmarks, ratios = self.read_landmarks(image_name, original_size=(image.shape[0], image.shape[1]))
            physical_scaling_factor = np.array([1, 1])

            return self._get_landmark_item_with_fallback(
                image_name, image, landmarks, physical_scaling_factor
            )

        else:
            raise ValueError(f"Unknown task type: {self.task_type}")

    def _get_simple_augmentations(self):
        """Get simple augmentations for chest X-rays (conservative geometric + reduced intensity)."""
        return [
            A.Affine(
                scale=(0.94, 1.10),
                translate_percent={'x': (-0.08, 0.08), 'y': (-0.08, 0.08)},
                rotate=(-12, 12),
                shear=(-3, 3),
                interpolation=cv2.INTER_LINEAR,
                border_mode=cv2.BORDER_CONSTANT,
                p=0.7
            ),
            A.RandomBrightnessContrast(brightness_limit=0.14, contrast_limit=0.14, p=0.6),
            A.RandomGamma(gamma_limit=(84, 116), p=0.5),
            A.CLAHE(clip_limit=(1, 3), tile_grid_size=(8, 8), p=0.25),
            A.OneOf([
                A.GaussNoise(std_range=(0.025, 0.10), per_channel=True),
                A.GaussianBlur(blur_limit=(3, 5), sigma_limit=(0.5, 1.5)),
            ], p=0.3),
        ]

    def _get_optimal_augmentations(self):
        """Optimal augmentations for chest X-rays (6 sparse landmarks, 195 images).

        Rationale:
        - Stronger scale/rotation due to sparse landmarks (only 6 lung field points)
        - Higher elastic/grid probability - sparse landmarks tolerate more deformation
        - Aggressive intensity augmentations - small dataset (195 images) needs generalization
        - Larger translation range - lung fields have natural anatomical variation
        """
        return [
            A.Affine(
                scale=(0.88, 1.18),
                translate_percent={'x': (-0.12, 0.12), 'y': (-0.12, 0.12)},
                rotate=(-20, 20),
                shear=(-6, 6),
                interpolation=cv2.INTER_LINEAR,
                border_mode=cv2.BORDER_CONSTANT,
                p=0.8
            ),
            A.ElasticTransform(
                alpha=45,
                sigma=9,
                interpolation=cv2.INTER_LINEAR,
                border_mode=cv2.BORDER_CONSTANT,
                p=0.35
            ),
            A.GridDistortion(
                num_steps=5,
                distort_limit=0.13,
                interpolation=cv2.INTER_LINEAR,
                border_mode=cv2.BORDER_CONSTANT,
                p=0.25
            ),
        ]
    
    def _get_advanced_augmentations(self):
        """Advanced augmentations for chest X-rays (most aggressive, may lose landmarks).
        
        Rationale:
        - Wider range of geometric transforms for maximum generalization
        - Higher probability of elastic/grid distortions (may lose landmarks)
        - Stronger intensity augmentations for robustness
        - Fallback mechanism ensures landmarks are preserved when this fails
        """
        return [
            A.Affine(
                scale=(0.85, 1.20),
                translate_percent={'x': (-0.15, 0.15), 'y': (-0.15, 0.15)},
                rotate=(-25, 25),
                shear=(-8, 8),
                interpolation=cv2.INTER_LINEAR,
                border_mode=cv2.BORDER_CONSTANT,
                p=0.8
            ),
            A.ElasticTransform(
                alpha=50,
                sigma=10,
                interpolation=cv2.INTER_LINEAR,
                border_mode=cv2.BORDER_CONSTANT,
                p=0.4
            ),
            A.GridDistortion(
                num_steps=5,
                distort_limit=0.15,
                interpolation=cv2.INTER_LINEAR,
                border_mode=cv2.BORDER_CONSTANT,
                p=0.3
            ),
        ] + self._get_intensity_augmentations()


if __name__ == "__main__":
    from utils.visualizations import visualize_dataset_samples, print_dataset_statistics
    
    # Test pretraining
    print("\n=== Testing Pretraining ===")
    dataset_pretrain = ChestDataset(
        root_dir="/nethome/home/user/Datasets/Landmarks_datasets",
        phase='train',
        task_type='pretraining'
    )
    print_dataset_statistics(dataset_pretrain)
    
    # Test landmarks
    print("\n=== Testing Landmarks ===")
    dataset_landmarks = ChestDataset(
        root_dir="/nethome/home/user/Datasets/Landmarks_datasets",
        phase='train',
        task_type='landmarks',
        sigma=2.0
    )
    print_dataset_statistics(dataset_landmarks)

    
    # Test augmentation fallback strategy and collect statistics
    print("\n=== Testing Augmentation Levels ===")
    dataset_landmarks.reset_augmentation_stats()
    num_test_samples = min(100, len(dataset_landmarks))
    print(f"Testing {num_test_samples} samples to analyze augmentation levels...")
    
    for i in range(num_test_samples):
        _ = dataset_landmarks[i]
    
    stats = dataset_landmarks.get_augmentation_stats()
    total = sum(stats.values())
    
    print("\nAugmentation Level Usage:")
    print(f"  Advanced: {stats['advanced']:4d} ({stats['advanced']/total*100:5.1f}%) - Most aggressive")
    print(f"  Optimal:  {stats['optimal']:4d} ({stats['optimal']/total*100:5.1f}%) - Full augmentations")
    print(f"  Simple:   {stats['simple']:4d} ({stats['simple']/total*100:5.1f}%) - Safe geometric + intensity")
    print(f"  Base:     {stats['base']:4d} ({stats['base']/total*100:5.1f}%) - Preprocessing only")
    print(f"  Total:    {total:4d}")
    print(f"\nFallback rate: {(stats['optimal'] + stats['simple'] + stats['base'])/total*100:.1f}%")
    
    # Visualize samples
    output_dir = "visualizations/chest_unified"
    os.makedirs(output_dir, exist_ok=True)
    
    visualize_dataset_samples(
        dataset_pretrain,
        num_samples=64,
        save_path=os.path.join(output_dir, "pretraining_samples.png"),
        title="Chest Dataset - Pretraining"
    )
    
    visualize_dataset_samples(
        dataset_landmarks,
        num_samples=32,
        save_path=os.path.join(output_dir, "landmarks_samples.png"),
        title="Chest Dataset - Landmarks"
    )
