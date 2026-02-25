"""
Shared evaluation utilities for pretraining modules.
"""

import math
import os

import torch
from torch.utils.data import DataLoader
from torchmetrics.image.fid import FrechetInceptionDistance


@torch.no_grad()
def compute_per_class_fid(
    num_classes,
    dataset,
    generate_fn,
    fid_path,
    device,
    num_images=256,
    batch_size=16,
    ddim_steps=50,
):
    """
    Compute per-class FID by collecting real images and generating fake ones.

    Args:
        num_classes: Number of classes.
        dataset: Dataset with 'class_label' and 'image' fields.
        generate_fn: Callable (class_idx, current_batch_size, device) -> Tensor[B,C,H,W]
                     Returns a batch of generated images in [0, 1].
        fid_path: File path to write FID scores.
        device: torch.device.
        num_images: Number of images per class for FID computation.
        batch_size: Batch size for generation and real-image feeding.
        ddim_steps: Passed through for logging only (generation handled by generate_fn).
    """
    print(f"\n{'='*60}")
    print("COMPUTING PER-CLASS FID")
    print(f"{'='*60}")

    num_batches = math.ceil(num_images / batch_size)

    # Collect real images per class
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=2)
    real_by_class = {c: [] for c in range(num_classes)}
    real_counts = {c: 0 for c in range(num_classes)}

    for batch in dataloader:
        for c in range(num_classes):
            if real_counts[c] < num_images:
                mask = batch['class_label'] == c
                if mask.any():
                    imgs = batch['image'][mask]
                    need = num_images - real_counts[c]
                    real_by_class[c].append(imgs[:need])
                    real_counts[c] += imgs[:need].shape[0]
        if all(real_counts[c] >= num_images for c in range(num_classes)):
            break

    fid_scores = {}

    for class_idx in range(num_classes):
        print(f"\nClass {class_idx}: Computing FID ({num_images} images)...")

        fid_metric = FrechetInceptionDistance(normalize=True).to(device)

        # Feed real images batch-by-batch
        real_imgs = torch.cat(real_by_class[class_idx], dim=0)[:num_images]
        for i in range(0, real_imgs.shape[0], batch_size):
            batch_real = real_imgs[i:i + batch_size].to(device)
            batch_real = torch.clamp(batch_real, 0.0, 1.0)
            if batch_real.shape[1] == 1:
                batch_real = batch_real.repeat(1, 3, 1, 1)
            fid_metric.update(batch_real, real=True)
            del batch_real

        # Generate fake images and feed to FID
        for batch_idx in range(num_batches):
            current_batch_size = min(batch_size, num_images - batch_idx * batch_size)
            generated = generate_fn(class_idx, current_batch_size, device)
            generated = torch.clamp(generated, 0.0, 1.0)
            if generated.shape[1] == 1:
                generated = generated.repeat(1, 3, 1, 1)
            fid_metric.update(generated, real=False)
            del generated

        score = fid_metric.compute().item()
        fid_scores[class_idx] = score
        print(f"  Class {class_idx} FID: {score:.2f}")
        del fid_metric

    # Summary
    mean_fid = sum(fid_scores.values()) / len(fid_scores)
    print(f"\n{'─'*60}")
    for c, s in fid_scores.items():
        print(f"  Class {c}: FID = {s:.2f}")
    print(f"  Mean FID: {mean_fid:.2f}")
    print(f"{'='*60}\n")

    # Save to file
    os.makedirs(os.path.dirname(fid_path), exist_ok=True)
    with open(fid_path, 'w') as f:
        f.write("Per-Class FID Scores (End of Training)\n")
        f.write(f"Images per class: {num_images}, DDIM steps: {ddim_steps}\n\n")
        for c, s in fid_scores.items():
            f.write(f"Class {c}: {s:.4f}\n")
        f.write(f"\nMean FID: {mean_fid:.4f}\n")
    print(f"FID scores saved to: {fid_path}")

    del real_by_class
