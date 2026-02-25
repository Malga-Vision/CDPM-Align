"""
Combined dataset for multi-anatomy X-ray pretraining (CDPM, SSL, etc.).

Merges chest, hand, and cephalo datasets into a single ConcatDataset with
class labels for conditional generation or flexible dataset selection.
"""

from torch.utils.data import ConcatDataset
from .chest_dataset import ChestDataset
from .hand_dataset import HandDataset
from .cephalo_dataset import CephaloDataset


# Registry of available anatomy datasets
_ANATOMY_DATASETS = {
    'chest': ChestDataset,
    'hand': HandDataset,
    'cephalo': CephaloDataset,
}


class CombinedDataset(ConcatDataset):
    """
    Combines multiple anatomy datasets for pretraining.

    Supports flexible dataset selection (any subset of chest/hand/cephalo)
    and assigns integer class labels for conditional generation.

    Class labels are assigned in order of the `datasets` tuple:
        datasets=('chest', 'hand', 'cephalo')  →  0, 1, 2
        datasets=('hand',)                     →  0
    """

    def __init__(
        self,
        root_dir,
        phase='train',
        image_size=(256, 256),
        channels=1,
        task_type='pretraining',
        datasets=('chest', 'hand', 'cephalo'),
        **kwargs,
    ):
        """
        Args:
            root_dir: Root directory containing all datasets.
            phase: 'train', 'val', 'test', 'train_val', or 'all'.
            image_size: Target image size (height, width).
            channels: Number of image channels (1 for grayscale).
            task_type: Task type passed to individual datasets.
            datasets: Tuple of dataset names to include.
            **kwargs: Additional arguments forwarded to each dataset.
        """
        loaded_datasets = []
        dataset_info = []

        for name in datasets:
            key = name.lower()
            if key not in _ANATOMY_DATASETS:
                print(f"Warning: Unknown dataset '{name}', skipping...")
                continue

            DatasetClass = _ANATOMY_DATASETS[key]
            try:
                ds = DatasetClass(
                    root_dir=root_dir,
                    phase=phase,
                    image_size=image_size,
                    channels=channels,
                    task_type=task_type,
                    **kwargs,
                )
                setattr(self, f'_{key}_dataset', ds)
                loaded_datasets.append(ds)
                dataset_info.append(f"{len(ds)} {key}")
            except Exception as e:
                print(f"Warning: Could not load {name} dataset: {e}")
                continue

        if not loaded_datasets:
            raise ValueError(f"No datasets could be loaded from {datasets}")

        self.num_classes = len(loaded_datasets)
        self._datasets_list = loaded_datasets

        # Pre-compute cumulative sizes for fast class-label lookup
        self._cumulative_sizes_for_labels = []
        total = 0
        for ds in loaded_datasets:
            total += len(ds)
            self._cumulative_sizes_for_labels.append(total)

        super().__init__(loaded_datasets)

        print(
            f"CombinedDataset ({phase}): "
            + " + ".join(dataset_info)
            + f" = {len(self)} total images"
        )

    # ------------------------------------------------------------------
    # Class-label helpers
    # ------------------------------------------------------------------

    def _get_class_label(self, idx):
        """Return the integer class label for a global index."""
        for class_idx, cum_size in enumerate(self._cumulative_sizes_for_labels):
            if idx < cum_size:
                return class_idx
        return len(self._cumulative_sizes_for_labels) - 1

    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        sample['class_label'] = self._get_class_label(idx)
        return sample


