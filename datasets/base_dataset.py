"""
Unified base dataset class for all medical imaging tasks.
All dataset implementations should inherit from this class.
"""

import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import pydicom
from landmark_detection.landmark_utils import keypoints2heatmaps


class BaseDataset(Dataset):
    """
    Unified base dataset class for all medical imaging tasks.
    
    Supports task types:
    - 'pretraining': Unlabeled data for pretraining (DDPM, SSL)
    - 'landmarks': Landmark detection

    All dataset implementations should inherit from this class and implement
    the necessary methods based on their task type.
    """
    
    def __init__(
        self, 
        root_dir, 
        dataset_name,
        phase='train', 
        image_size=(512, 512), 
        channels=1,
        task_type='pretraining',  # 'pretraining', 'landmarks'
        **kwargs
    ):
        """
        Initialize the base dataset.
        
        Args:
            root_dir (str): Root directory containing the dataset
            dataset_name (str): Name of the dataset folder
            phase (str): 'train', 'val', 'test', 'train_val', or 'all'
            image_size (tuple): Target image size (height, width)
            channels (int): Number of image channels (1 for grayscale, 3 for RGB)
            task_type (str): Type of task ('pretraining', 'landmarks')
            **kwargs: Additional arguments for specific dataset implementations
        """
        self.root_dir = root_dir
        self.dataset_path = os.path.join(root_dir, dataset_name)
        self.dataset_name = dataset_name
        self.phase = phase
        # Handle image_size as tuple, list, or single int
        if isinstance(image_size, (tuple, list)):
            self.image_size = tuple(image_size)
        else:
            self.image_size = (image_size, image_size)
        self.channels = channels
        self.task_type = task_type
        
        # Store kwargs for subclass use
        self.kwargs = kwargs
        
        # Setup image paths - to be implemented by child classes
        self.image_files = []
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        """
        Get an item from the dataset.
        
        Returns:
            dict: A dictionary containing task-specific data:
                - For all tasks: 'name', 'image'
                - For landmarks: 'landmarks', 'target' (heatmaps), 'physical_scaling_factor'
        """
        # To be implemented by child classes
        raise NotImplementedError("Subclasses must implement __getitem__")
    
    def read_image(self, image_path):
        """
        Read and process an image (including DICOM) from disk,
        inverting DICOM pixel intensities.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            np.ndarray: Processed image as numpy array
        """
        ext = os.path.splitext(image_path)[1].lower()

        if ext == '.dcm':
            ds = pydicom.dcmread(image_path)
            image_np = ds.pixel_array.astype(np.float32)
            image_np = image_np.max() - image_np

            if self.channels == 3:
                if image_np.ndim == 2:
                    image_np = np.stack([image_np] * 3, axis=-1)
            elif self.channels == 1:
                if image_np.ndim == 2:
                    image_np = np.expand_dims(image_np, axis=2)
            else:
                raise ValueError(f"Channels must be either 1 or 3, but got {self.channels}")

        else:
            if self.channels == 3:
                img = Image.open(image_path).convert('RGB')
                image_np = np.array(img, dtype=np.uint8)

            elif self.channels == 1:
                img = Image.open(image_path).convert('L')
                image_np = np.array(img, dtype=np.uint8)
                image_np = np.expand_dims(image_np, axis=2)
            else:
                raise ValueError(f"Channels must be either 1 or 3, but got {self.channels}")

        return image_np
    
    def get_transforms(self, level='advanced'):
        """
        Get task-specific transforms for the dataset.
        
        This method provides a unified interface for getting transforms based on:
        - Task type (pretraining, landmarks, etc.)
        - Phase (train, val, test)
        - Augmentation level for landmarks (base, simple, optimal, advanced)
        
        Args:
            level: Augmentation level for landmarks task:
                - 'base': Only preprocessing (resize, pad, normalize)
                - 'simple': Conservative geometric + intensity (safe)
                - 'optimal': Best balance for the dataset
                - 'advanced': Most aggressive augmentations (default, may lose landmarks)
        
        Returns:
            A.Compose: Albumentations transforms
        
        Note:
            Subclasses should implement _get_simple_augmentations(), 
            _get_optimal_augmentations(), and _get_advanced_augmentations() 
            for landmarks tasks. For other task types, _get_pretraining_augmentations() 
            can be overridden.
        """
        preprocessing = self._get_preprocessing_transforms()
        
        if self.task_type == 'pretraining':
            if self.phase in ['train', 'train_val']:
                augmentations = self._get_pretraining_augmentations() + preprocessing
                return A.Compose(augmentations)
            return A.Compose(preprocessing)
        
        elif self.task_type == 'landmarks':
            keypoint_params = A.KeypointParams(format='xy', remove_invisible=False)
            
            # BASE: Only preprocessing - no augmentations
            if level == 'base':
                return A.Compose(preprocessing, keypoint_params=keypoint_params)
            
            # SIMPLE: Conservative geometric + reduced intensity
            if level == 'simple':
                augmentations = self._get_simple_augmentations() + preprocessing
                return A.Compose(augmentations, keypoint_params=keypoint_params)
            
            # OPTIMAL: Best balance for the dataset
            if level == 'optimal':
                augmentations = self._get_optimal_augmentations() + preprocessing
                return A.Compose(augmentations, keypoint_params=keypoint_params)
            
            # ADVANCED: Most aggressive augmentations (may lose landmarks)
            if level == 'advanced':
                augmentations = self._get_advanced_augmentations() + preprocessing
                return A.Compose(augmentations, keypoint_params=keypoint_params)
            
            raise ValueError(f"Unknown augmentation level: {level}. Use 'base', 'simple', 'optimal', or 'advanced'.")

        raise ValueError(f"Unknown task_type: {self.task_type}. Use 'pretraining' or 'landmarks'.")
    
    # === Methods for landmark detection support ===
    
    def read_landmarks(self, image_name, image_shape):
        """
        Read landmarks for landmark detection tasks.
        To be implemented by child classes that support landmarks.
        
        Args:
            image_name (str): Image name/identifier
            image_shape (tuple): Shape of the image
            
        Returns:
            tuple: (landmarks, additional_info)
                - landmarks: list of (x, y) coordinates
                - additional_info: any other data needed
        """
        raise NotImplementedError("Subclasses must implement read_landmarks for landmark tasks")
    
    # === Augmentation stats tracking ===
    
    def get_augmentation_stats(self):
        """Get statistics on which augmentation levels were used."""
        if not hasattr(self, '_augmentation_stats'):
            return {'base': 0, 'simple': 0, 'optimal': 0, 'advanced': 0}
        return self._augmentation_stats.copy()
    
    def reset_augmentation_stats(self):
        """Reset augmentation statistics."""
        self._augmentation_stats = {'base': 0, 'simple': 0, 'optimal': 0, 'advanced': 0}
    
    # === Common transform methods ===
    
    def _get_preprocessing_transforms(self):
        """Get common preprocessing transforms (resize, pad, normalize, to tensor)."""
        return [
            A.LongestMaxSize(max_size=max(self.image_size)),
            A.PadIfNeeded(
                min_height=self.image_size[0],
                min_width=self.image_size[1],
                border_mode=cv2.BORDER_CONSTANT
            ),
            A.Normalize(normalization='min_max'),
            ToTensorV2()
        ]
    
    # === Hooks for dataset-specific augmentations ===
    
    def _get_pretraining_augmentations(self):
        """
        Get augmentations for pretraining task.
        Default implementation - can be overridden by subclasses.
        """
        return [
            A.Affine(
                scale=(0.95, 1.05), 
                translate_percent=(-0.03, 0.03), 
                rotate=(-5, 5), 
                shear=0, 
                p=0.3
            ),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
            A.RandomGamma(gamma_limit=(90, 110), p=0.3),
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 3), p=1.0),
                A.GaussNoise(std_range=(0.01, 0.05), p=1.0),
            ], p=0.2),
        ]
    
    def _get_intensity_augmentations(self):
        """Get intensity-based augmentations (safe for landmarks)."""
        return [
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
            A.RandomGamma(gamma_limit=(80, 120), p=0.6),
            A.CLAHE(clip_limit=(1, 4), tile_grid_size=(8, 8), p=0.3),
            A.OneOf([
                A.GaussNoise(std_range=(0.04, 0.15), per_channel=True),
                A.GaussianBlur(blur_limit=(3, 5), sigma_limit=(0.5, 1.5)),
            ], p=0.4),
        ]
    
    def _get_simple_augmentations(self):
        """
        Get simple augmentations for landmarks task.
        Must be implemented by subclasses that use landmarks.
        """
        raise NotImplementedError("Subclasses must implement _get_simple_augmentations for landmarks")
    
    def _get_optimal_augmentations(self):
        """
        Get optimal augmentations for landmarks task.
        Must be implemented by subclasses that use landmarks.
        """
        raise NotImplementedError("Subclasses must implement _get_optimal_augmentations for landmarks")
    
    def _get_advanced_augmentations(self):
        """
        Get advanced augmentations for landmarks task.
        Must be implemented by subclasses that use landmarks.
        """
        raise NotImplementedError("Subclasses must implement _get_advanced_augmentations for landmarks")
    
    def _get_landmark_item_with_fallback(self, image_name, image, landmarks, physical_scaling_factor):
        """
        Generic landmark item processing with augmentation fallback.
        
        This method handles the augmentation fallback strategy:
        - Try 'advanced' augmentations first (most aggressive)
        - Fallback to 'optimal' if advanced fails to preserve landmarks
        - Fallback to 'simple' if optimal also fails
        - Fallback to 'base' (preprocessing only) if simple also fails
        
        Args:
            image_name: Name of the image
            image: Input image array
            landmarks: Landmark coordinates
            physical_scaling_factor: Physical scaling factor for landmarks
            
        Returns:
            dict: Processed item with image, landmarks, target, etc.
        """
        if self.phase in ['train', 'train_val']:
            augmentation_levels = ['advanced', 'optimal', 'simple', 'base']
        else:
            augmentation_levels = ['base']

        transformed_image = None
        transformed_landmarks = None

        for level in augmentation_levels:
            transform = self.get_transforms(level=level)
            transformed = transform(image=image, keypoints=landmarks)
            transformed_landmarks = transformed['keypoints']

            if len(transformed_landmarks) == self.num_landmarks:
                transformed_image = transformed['image']
                if not hasattr(self, '_augmentation_stats'):
                    self._augmentation_stats = {'base': 0, 'simple': 0, 'optimal': 0, 'advanced': 0}
                self._augmentation_stats[level] += 1
                break

        if transformed_image is None or len(transformed_landmarks) != self.num_landmarks:
            raise RuntimeError(
                f"Failed to preserve all {self.num_landmarks} landmarks for {image_name} "
                f"even with base transforms. Got {len(transformed_landmarks)} landmarks."
            )

        transformed_landmarks = np.array(transformed_landmarks)
        transformed_landmarks[:, 0] = np.clip(transformed_landmarks[:, 0], 0, self.image_size[1] - 1)
        transformed_landmarks[:, 1] = np.clip(transformed_landmarks[:, 1], 0, self.image_size[0] - 1)

        heatmaps = keypoints2heatmaps(transformed_landmarks, img_size=self.image_size, sigma=self.sigma)
        
        return {
            'name': image_name,
            'image': transformed_image.float(),
            'landmarks': torch.from_numpy(transformed_landmarks).double(),
            'target': torch.from_numpy(np.stack(heatmaps)).float(),
            'physical_scaling_factor': torch.from_numpy(physical_scaling_factor).double()
        }
