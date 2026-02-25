"""
NIH Chest X-ray dataset for pretraining tasks.
Dataset source: https://www.kaggle.com/datasets/nih-chest-xrays/data
Contains 112,120 X-ray images from 30,805 unique patients.
"""

import os
import sys

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from datasets.base_dataset import BaseDataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

NIH_CHEST_NUM_DIRS = 12


class NIHChestDataset(BaseDataset):
    """
    NIH Chest X-ray dataset for pretraining (unlabeled images).

    The dataset contains 112,120 frontal-view chest X-ray images from
    30,805 unique patients, used for diffusion model pretraining.
    """

    def __init__(
        self,
        root_dir,
        phase='train',
        image_size=(512, 512),
        channels=1,
        task_type='pretraining',
        max_samples=None,
        **kwargs
    ):
        # Detect whether root_dir is the dataset itself or a parent directory
        if os.path.exists(os.path.join(root_dir, 'Data_Entry_2017.csv')):
            dataset_name = '.'
        else:
            dataset_name = 'NIH-Chest-Xray'
        
        super().__init__(
            root_dir=root_dir, 
            dataset_name=dataset_name,
            phase=phase,
            image_size=image_size,
            channels=channels,
            task_type=task_type,
            **kwargs
        )
        
        self.max_samples = max_samples

        metadata_path = os.path.join(self.dataset_path, 'Data_Entry_2017.csv')
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(
                f"Metadata file not found at {metadata_path}. "
                f"Please ensure the NIH Chest X-ray dataset is properly extracted."
            )
        
        self.metadata = pd.read_csv(metadata_path)

        self.image_dirs = []
        for i in range(1, NIH_CHEST_NUM_DIRS + 1):
            img_dir = os.path.join(self.dataset_path, f'images_{i:03d}', 'images')
            if os.path.exists(img_dir):
                self.image_dirs.append(img_dir)
        
        if not self.image_dirs:
            raise FileNotFoundError(
                f"No image directories found at {self.dataset_path}. "
                f"Please ensure the dataset is properly extracted."
            )

        self._build_image_index()

        available_images = set(self.image_index.keys())
        all_images = self.metadata['Image Index'].tolist()
        self.valid_images = [img for img in all_images if img in available_images]

        self._create_patient_splits()

        if phase == 'train':
            self.image_files = self.train_images
        elif phase in ['val', 'validate']:
            self.image_files = self.val_images
        elif phase == 'train_val':
            self.image_files = self.train_images + self.val_images
        elif phase == 'test':
            self.image_files = self.test_images
        elif phase == 'all':
            self.image_files = self.valid_images
        else:
            raise ValueError(f"Unknown phase: {phase}")

        if max_samples is not None and max_samples < len(self.image_files):
            self.image_files = self.image_files[:max_samples]
        
        print(f"Loaded {len(self.image_files)} images for {self.dataset_name} - {phase} - {task_type}")
    
    def _build_image_index(self):
        """Build a mapping from image filename to full path."""
        print("Building image index...")
        self.image_index = {}
        for idx, img_dir in enumerate(self.image_dirs, 1):
            if os.path.exists(img_dir):
                print(f"  Scanning directory {idx}/{len(self.image_dirs)}: {os.path.basename(os.path.dirname(img_dir))}")
                for img_file in os.listdir(img_dir):
                    if img_file.endswith('.png'):
                        self.image_index[img_file] = os.path.join(img_dir, img_file)
        print(f"  Found {len(self.image_index)} images total")
    
    def _create_patient_splits(self):
        """Create patient-based 70/10/20 train/val/test splits (no data leakage)."""
        patient_ids = self.metadata['Patient ID'].unique()
        rng = np.random.default_rng(42)
        rng.shuffle(patient_ids)
        
        # Split patients
        n_patients = len(patient_ids)
        train_end = int(0.7 * n_patients)
        val_end = int(0.8 * n_patients)
        
        train_patients = set(patient_ids[:train_end])
        val_patients = set(patient_ids[train_end:val_end])
        test_patients = set(patient_ids[val_end:])

        self.train_images = []
        self.val_images = []
        self.test_images = []

        image_to_patient = dict(zip(self.metadata['Image Index'], self.metadata['Patient ID']))
        
        for img_name in self.valid_images:
            patient_id = image_to_patient.get(img_name)
            if patient_id is None:
                continue
            
            if patient_id in train_patients:
                self.train_images.append(img_name)
            elif patient_id in val_patients:
                self.val_images.append(img_name)
            else:
                self.test_images.append(img_name)
    
    def __getitem__(self, idx, _retries=0):
        image_name = self.image_files[idx]
        image_path = self.image_index[image_name]

        try:
            image = self.read_image(image_path)
        except Exception as e:
            if _retries >= 10:
                raise RuntimeError(
                    f"NIHChestDataset: failed to load item after 10 retries "
                    f"(last idx={idx})"
                ) from e
            print(f"Warning: Failed to load {image_path}: {e}. Trying next image...")
            return self.__getitem__((idx + 1) % len(self), _retries + 1)
        
        transformed = self.get_transforms()(image=image)
        transformed_image = transformed['image']

        return {
            'name': image_name,
            'image': transformed_image.float()
        }
    
    def get_transforms(self):
        """Get preprocessing transforms for NIH chest X-rays."""
        return A.Compose([
            A.LongestMaxSize(max_size=max(self.image_size)),
            A.PadIfNeeded(
                min_height=self.image_size[0],
                min_width=self.image_size[1],
                border_mode=cv2.BORDER_CONSTANT
            ),
            A.Normalize(normalization='min_max'),
            ToTensorV2()
        ])



class NIHCDPMDataset(Dataset):
    """NIH Chest X-ray wrapper for CDPM training (single-class, class_label=0)."""

    def __init__(
        self,
        root_dir,
        phase='train',
        image_size=(512, 512),
        max_samples=None,
        **kwargs
    ):
        self.num_classes = 1  # Single class for unconditional generation

        self._dataset = NIHChestDataset(
            root_dir=root_dir,
            phase=phase,
            image_size=image_size,
            channels=1,
            task_type='pretraining',
            max_samples=max_samples,
            **kwargs
        )

        print(f"NIH CDPM dataset: {len(self._dataset)} chest X-ray images, {self.num_classes} class")

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx):
        sample = self._dataset[idx]
        sample['class_label'] = 0
        return sample


if __name__ == "__main__":
    from utils.visualizations import print_dataset_statistics

    print("\n=== Testing NIH Chest Pretraining ===")
    dataset = NIHChestDataset(
        root_dir="/data05/user",
        phase='train',
        task_type='pretraining',
        max_samples=100,
    )
    print_dataset_statistics(dataset)

    sample = dataset[0]
    print(f"Sample keys: {sample.keys()}")
    print(f"Image shape: {sample['image'].shape}")
    print(f"Image range: [{sample['image'].min():.3f}, {sample['image'].max():.3f}]")
