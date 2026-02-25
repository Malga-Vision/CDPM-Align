"""
Unified Hand X-ray dataset for all tasks (pretraining, landmarks).
"""

import os
import sys

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import numpy as np
import pandas as pd
from datasets.base_dataset import BaseDataset
from landmark_detection.landmark_utils import get_scale_factor
import albumentations as A
import cv2

_TRAIN_END = 550
_VAL_END = 609


class HandDataset(BaseDataset):
    """
    Unified Hand X-ray dataset for:
    - Pretraining (unlabeled images)
    - Landmarks (landmark detection)
    """
    
    def __init__(
        self, 
        root_dir, 
        phase='train', 
        image_size=(512, 368), 
        channels=1,
        task_type='pretraining',
        sigma=5,  # For landmarks
        **kwargs
    ):
        super().__init__(
            root_dir=root_dir, 
            dataset_name='hand',
            phase=phase,
            image_size=image_size,
            channels=channels,
            task_type=task_type,
            sigma=sigma,
            **kwargs
        )
        
        self.sigma = sigma
        
        # Set paths
        self.image_dir = os.path.join(self.dataset_path, 'jpg')
        
        if task_type == 'landmarks':
            self.labels = pd.read_csv(os.path.join(self.dataset_path, 'labels/all.csv'), 
                                      header=None, index_col=0)
            index_set = set(self.labels.index)
            files = [f[:-4] for f in sorted(os.listdir(self.image_dir)) if f.endswith('.jpg')]
            files = [i for i in files if int(i) in index_set]
        else:
            files = [i[:-4] for i in sorted(os.listdir(self.image_dir)) if i.endswith('.jpg')]
        
        # Split into train/val/test
        if phase == 'train':
            self.image_files = files[:_TRAIN_END]
        elif phase in ['val', 'validate']:
            self.image_files = files[_TRAIN_END:_VAL_END]
        elif phase == 'train_val':
            self.image_files = files[:_VAL_END]
        elif phase == 'test':
            self.image_files = files[_VAL_END:]
        elif phase == 'all':
            self.image_files = files
        else:
            raise ValueError(f"Unknown phase: {phase}")
        
        # Set number of landmarks (for landmarks task)
        if task_type == 'landmarks':
            self.num_landmarks = 37
            self.wrist_width_mm = 50  # Assumption: width of the wrist is 50mm
        
        print(f"Loaded {len(self.image_files)} images for {self.dataset_name} - {phase} - {task_type}")
    
    def read_landmarks(self, image_name, original_size):
        """Read landmarks from the hand dataset."""
        li = list(self.labels.loc[int(image_name), :])
        landmarks_coordinates = []
        landmarks_ratios = []
        
        for i in range(0, len(li), 2):
            landmarks_coordinates.append((float(li[i]), float(li[i + 1])))
            ratios = (li[i] / original_size[0], li[i + 1] / original_size[1])
            landmarks_ratios.append(ratios)
            
        return np.array(landmarks_coordinates), np.array(landmarks_ratios)
    
    def __getitem__(self, idx):
        image_name = self.image_files[idx]
        image_path = os.path.join(self.image_dir, f"{image_name}.jpg")
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

            wrist_width_pixels = np.linalg.norm(landmarks[0] - landmarks[4])
            mm_per_pixel = self.wrist_width_mm / wrist_width_pixels
            
            width_scale_factor, height_scale_factor = get_scale_factor(image.shape, self.image_size)
            physical_scaling_factor = np.array([width_scale_factor * mm_per_pixel, height_scale_factor * mm_per_pixel])

            return self._get_landmark_item_with_fallback(
                image_name, image, landmarks, physical_scaling_factor
            )
        
        else:
            raise ValueError(f"Unknown task type: {self.task_type}")
    
    def _get_simple_augmentations(self):
        """Get simple augmentations for hand X-rays (conservative geometric + reduced intensity)."""
        return [
            A.Affine(
                scale=(0.97, 1.07),
                translate_percent={'x': (-0.04, 0.04), 'y': (-0.04, 0.04)},
                rotate=(-5, 5),
                shear=(-2, 2),
                interpolation=cv2.INTER_LINEAR,
                border_mode=cv2.BORDER_CONSTANT,
                p=0.6
            ),
            A.RandomBrightnessContrast(brightness_limit=0.10, contrast_limit=0.10, p=0.5),
            A.RandomGamma(gamma_limit=(92, 108), p=0.4),
            A.CLAHE(clip_limit=(1, 2), tile_grid_size=(8, 8), p=0.15),
            A.OneOf([
                A.GaussNoise(std_range=(0.015, 0.05), per_channel=True),
                A.GaussianBlur(blur_limit=(3, 3), sigma_limit=(0.5, 1.0)),
            ], p=0.15),
        ]
    
    def _get_optimal_augmentations(self):
        """Optimal augmentations for hand X-rays (37 dense landmarks, 550 images).
        
        Rationale:
        - Conservative scale/rotation due to dense landmark distribution (37 landmarks)
        - Reduced elastic/grid distortions to minimize landmark loss
        - Moderate intensity augmentations for robustness without excessive noise
        """
        return [
            A.Affine(
                scale=(0.92, 1.12),
                translate_percent={'x': (-0.08, 0.08), 'y': (-0.08, 0.08)},
                rotate=(-12, 12),
                shear=(-4, 4),
                interpolation=cv2.INTER_LINEAR,
                border_mode=cv2.BORDER_CONSTANT,
                p=0.8
            ),
            A.ElasticTransform(
                alpha=35,
                sigma=8,
                interpolation=cv2.INTER_LINEAR,
                border_mode=cv2.BORDER_CONSTANT,
                p=0.25
            ),
            A.GridDistortion(
                num_steps=5,
                distort_limit=0.10,
                interpolation=cv2.INTER_LINEAR,
                border_mode=cv2.BORDER_CONSTANT,
                p=0.15
            ),
        ]
    
    def _get_advanced_augmentations(self):
        """Advanced augmentations for hand X-rays (most aggressive, may lose landmarks).
        
        Rationale:
        - Wider range of geometric transforms for maximum generalization
        - Higher probability of elastic/grid distortions (may lose landmarks)
        - Stronger intensity augmentations for robustness
        - Fallback mechanism ensures landmarks are preserved when this fails
        """
        return [
            A.Affine(
                scale=(0.9, 1.15),
                translate_percent={'x': (-0.1, 0.1), 'y': (-0.1, 0.1)},
                rotate=(-15, 15),
                shear=(-5, 5),
                interpolation=cv2.INTER_LINEAR,
                border_mode=cv2.BORDER_CONSTANT,
                p=0.8
            ),
            A.ElasticTransform(
                alpha=40,
                sigma=8,
                interpolation=cv2.INTER_LINEAR,
                border_mode=cv2.BORDER_CONSTANT,
                p=0.3
            ),
            A.GridDistortion(
                num_steps=5,
                distort_limit=0.12,
                interpolation=cv2.INTER_LINEAR,
                border_mode=cv2.BORDER_CONSTANT,
                p=0.2
            ),
        ] + self._get_intensity_augmentations()


if __name__ == "__main__":
    from utils.visualizations import visualize_dataset_samples, print_dataset_statistics
    
    # Test pretraining
    print("\n=== Testing Pretraining ===")
    dataset_pretrain = HandDataset(
        root_dir="/nethome/home/user/Datasets/Landmarks_datasets",
        phase='train',
        task_type='pretraining'
    )
    print_dataset_statistics(dataset_pretrain)
    
    # Test landmarks
    print("\n=== Testing Landmarks ===")
    dataset_landmarks = HandDataset(
        root_dir="/nethome/home/user/Datasets/Landmarks_datasets",
        phase='train',
        task_type='landmarks',
        sigma=2.0,
        image_size=(512, 512)
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
    output_dir = "visualizations/hand_unified"
    os.makedirs(output_dir, exist_ok=True)
    
    visualize_dataset_samples(
        dataset_pretrain,
        num_samples=16,
        save_path=os.path.join(output_dir, "pretraining_samples.png"),
        title="Hand Dataset - Pretraining"
    )
    
    visualize_dataset_samples(
        dataset_landmarks,
        num_samples=16,
        save_path=os.path.join(output_dir, "landmarks_samples.png"),
        title="Hand Dataset - Landmarks"
    )
