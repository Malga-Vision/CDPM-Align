"""
Visualization utilities for training and dataset inspection.
"""

import os
import math
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import torch


# ============================================================================
# Unified Training History Plotting
# ============================================================================

def plot_training_history(history, save_path=None, task_type='general', metrics_to_plot=None):
    """
    Unified training history plotter for all downstream tasks.

    Args:
        history (dict): Training history with keys like 'train_loss', 'val_loss', 'val_metrics'
        save_path (str, optional): Path to save the plot
        task_type (str): Type of task ('landmarks' or 'general')
        metrics_to_plot (list, optional): Specific metrics to plot. If None, plots all available.

    Returns:
        fig: matplotlib figure object
    """
    # Determine metrics based on task type if not specified
    if metrics_to_plot is None:
        if task_type == 'landmarks':
            metrics_to_plot = ['mre', 'sdr', 'accuracy']
        else:
            # General case: detect all available val_ metrics
            metrics_to_plot = [k.replace('val_', '') for k in history.keys()
                             if k.startswith('val_') and k != 'val_loss']

    # Filter metrics that actually exist in history
    available_metrics = [m for m in metrics_to_plot
                        if f'val_{m}' in history or f'train_{m}' in history]

    # Determine subplot layout
    num_plots = 1 + len(available_metrics)  # Loss + metrics

    if num_plots <= 2:
        rows, cols = 1, num_plots
        figsize = (7 * cols, 5)
    elif num_plots <= 4:
        rows, cols = 2, 2
        figsize = (14, 10)
    elif num_plots <= 6:
        rows, cols = 2, 3
        figsize = (18, 10)
    else:
        rows, cols = 3, 3
        figsize = (18, 14)

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if num_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

    # Plot 1: Loss
    ax = axes[0]
    epochs = range(1, len(history['train_loss']) + 1)
    ax.plot(epochs, history['train_loss'], 'o-', label='Train Loss', linewidth=2, markersize=4)
    if 'val_loss' in history:
        ax.plot(epochs, history['val_loss'], 's-', label='Val Loss', linewidth=2, markersize=4)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Plot metrics
    for idx, metric in enumerate(available_metrics, start=1):
        if idx >= len(axes):
            break

        ax = axes[idx]

        # Plot validation metric
        if f'val_{metric}' in history:
            ax.plot(epochs, history[f'val_{metric}'], 'o-',
                   label=f'Val {metric.upper()}', linewidth=2, markersize=4)

        # Plot training metric if available
        if f'train_{metric}' in history:
            ax.plot(epochs, history[f'train_{metric}'], 's-',
                   label=f'Train {metric.upper()}', linewidth=2, markersize=4, alpha=0.7)

        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel(metric.upper(), fontsize=12)
        ax.set_title(f'{metric.upper()} Over Epochs', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(num_plots, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved training history to {save_path}")

    return fig


def plot_metrics_bar(metrics_dict, save_path=None, title='Metrics', figsize=(12, 6)):
    """
    Create a bar plot for metrics (works for any task).

    Args:
        metrics_dict (dict): Dictionary of metric names and values
        save_path (str, optional): Path to save the plot
        title (str): Title for the plot
        figsize (tuple): Figure size

    Returns:
        fig: matplotlib figure object
    """
    fig = plt.figure(figsize=figsize)

    metric_names = list(metrics_dict.keys())
    metric_values = list(metrics_dict.values())

    colors = plt.cm.Set3(np.linspace(0, 1, len(metric_names)))
    bars = plt.bar(metric_names, metric_values, color=colors, alpha=0.7, edgecolor='black')

    plt.xlabel('Metrics', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=45, ha='right')

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved metrics bar plot to {save_path}")

    return fig


def plot_metrics_radar(metrics_dict, save_path=None, title='Performance Radar',
                       ylim=(0, 1), figsize=(8, 8)):
    """
    Create a radar/spider plot for metrics.

    Args:
        metrics_dict (dict): Dictionary of metric names and values
        save_path (str, optional): Path to save the plot
        title (str): Title for the plot
        ylim (tuple): Y-axis limits
        figsize (tuple): Figure size

    Returns:
        fig: matplotlib figure object
    """
    metric_names = list(metrics_dict.keys())
    metric_values = list(metrics_dict.values())

    # Number of metrics
    num_vars = len(metric_names)

    # Compute angle for each metric
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

    # Complete the circle
    metric_values += metric_values[:1]
    angles += angles[:1]

    # Create radar plot
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection='polar'))

    # Plot data
    ax.plot(angles, metric_values, 'o-', linewidth=2, color='#1f77b4', markersize=8)
    ax.fill(angles, metric_values, alpha=0.25, color='#1f77b4')

    # Fix axis to go in the right order and start at 12 o'clock
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # Draw axis lines for each metric
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_names, size=10)

    # Set y limits
    ax.set_ylim(ylim)

    # Add gridlines
    ax.grid(True, linestyle='--', alpha=0.7)

    plt.title(title, size=14, fontweight='bold', pad=20)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved radar plot to {save_path}")

    return fig


# ============================================================================
# Dataset Visualization Utilities
# ============================================================================

def visualize_dataset_samples(dataset, num_samples=16, save_path=None, title=None):
    """
    Visualize a grid of dataset samples with their labels.

    Args:
        dataset: Dataset object that returns dictionaries with 'image' and optional 'target'
        num_samples (int): Number of samples to visualize
        save_path (str, optional): Path to save the visualization
        title (str, optional): Title for the visualization
    """
    # Ensure num_samples is a perfect square for grid layout
    grid_size = int(np.ceil(np.sqrt(num_samples)))
    num_samples = min(grid_size * grid_size, len(dataset))

    # Create figure
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(grid_size * 3, grid_size * 3))
    if grid_size == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    # Get random samples
    indices = np.random.choice(len(dataset), num_samples, replace=False)

    for idx, ax in enumerate(axes[:num_samples]):
        try:
            # Get sample
            sample = dataset[indices[idx]]
            image = sample['image']
            landmarks = sample.get('landmarks', None)
            target = sample.get('target', None)
            name = sample.get('name', f"Sample {indices[idx]}")
            label_fields = sample.get('label_fields', None)

            # Convert image to numpy and handle different channel formats
            if isinstance(image, torch.Tensor):
                image = image.cpu().numpy()
            if image.shape[0] in [1, 3]:  # CHW format
                image = np.transpose(image, (1, 2, 0))
            if image.shape[-1] == 1:
                image = np.squeeze(image)

            # Display image
            ax.imshow(image, cmap='gray' if image.ndim == 2 else None)

            # Handle landmarks if available
            if landmarks is not None:
                if isinstance(landmarks, torch.Tensor):
                    landmarks = landmarks.cpu().numpy()

                # Plot landmarks with labels if available
                for i, landmark in enumerate(landmarks):
                    # Use label field name if available, otherwise use index
                    label = label_fields[i] if label_fields is not None and i < len(label_fields) else f"{i+1}"
                    ax.plot(landmark[0], landmark[1], 'o', color='red', markersize=0.5)
                    ax.text(landmark[0]-50, landmark[1]+30, label, fontsize=2, color='yellow', bbox=dict(facecolor='black', alpha=0.5, boxstyle='round,pad=0.2'))

            # Handle landmark heatmap targets for visualization
            elif target is not None:
                if isinstance(target, torch.Tensor):
                    target = target.cpu().numpy()

                # Landmark heatmaps case
                if target.ndim == 3 and target.shape[0] > 1:  # Multiple heatmaps
                    # Define colormap for heatmaps
                    cmap_yellow = LinearSegmentedColormap.from_list(
                        'my_cmap_yellow', [(0, (0, 0, 0, 0)), (1, 'yellow')], N=256)
                    # Fuse heatmaps using maximum for visualization
                    fused_heatmap = np.max(target, axis=0)
                    ax.imshow(fused_heatmap, cmap=cmap_yellow, alpha=0.7)
                    # Plot landmark points if available
                    if landmarks is not None:
                        for i, landmark in enumerate(landmarks):
                            ax.plot(landmark[0], landmark[1], 'o', color='red', markersize=4)
                    ax.set_title(name, fontsize=9)
                else:
                    ax.set_title(name, fontsize=9)
            else:
                ax.set_title(name, fontsize=9)
            ax.axis('off')

        except Exception as e:
            print(f"Error visualizing sample {indices[idx]}: {e}")
            ax.text(0.5, 0.5, f"Error: {str(e)[:50]}...",
                   horizontalalignment='center', verticalalignment='center')
            ax.axis('off')

    # Hide unused subplots
    for idx in range(num_samples, len(axes)):
        axes[idx].axis('off')

    # Add overall title
    if title:
        fig.suptitle(title, fontsize=16, y=0.98)

    plt.tight_layout()

    # Save if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Saved visualization to {save_path}")

    plt.close(fig)
    return fig


def visualize_entire_dataset(dataset, output_dir, samples_per_page=16, max_pages=None, random_samples=False):
    """
    Visualize all samples in the dataset, organized into multiple pages.

    Args:
        dataset: Dataset object containing samples
        output_dir: Directory to save the visualization pages
        samples_per_page: Number of samples to show per page
        max_pages: Maximum number of pages to generate (None for all)
        random_samples: Whether to select random samples or sequential samples
    """
    os.makedirs(output_dir, exist_ok=True)

    total_samples = len(dataset)
    total_pages = math.ceil(total_samples / samples_per_page)

    if max_pages is not None:
        total_pages = min(total_pages, max_pages)

    print(f"Generating {total_pages} pages of visualizations ({samples_per_page} samples per page)...")

    if random_samples:
        # Generate random sample indices for all pages
        all_indices = np.random.permutation(total_samples)
    else:
        # Use sequential indices
        all_indices = np.arange(total_samples)

    for page in range(total_pages):
        start_idx = page * samples_per_page
        end_idx = min((page + 1) * samples_per_page, total_samples)

        # Get indices for this page
        page_indices = all_indices[start_idx:end_idx]

        # Create a subset view of the dataset with these specific indices
        subset = torch.utils.data.Subset(dataset, page_indices.tolist())

        # Generate the visualization for this page
        save_path = os.path.join(output_dir, f"samples_page_{page+1:03d}.png")

        visualize_dataset_samples(
            subset,
            num_samples=len(subset),
            save_path=save_path,
            title=f"Dataset Samples (Page {page+1}/{total_pages})"
        )

        print(f"Generated page {page+1}/{total_pages}")

    print(f"Visualization complete. All pages saved to {output_dir}")


def print_dataset_statistics(dataset):
    """
    Print statistics about the dataset.

    Args:
        dataset: Dataset object
    """
    print("\nDataset Statistics:")
    print(f"Number of samples: {len(dataset)}")

    # Get a sample to analyze
    sample = dataset[0]

    # Image statistics
    image = sample['image']
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()

    print("\nImage Statistics:")
    print(f"Shape: {image.shape}")
    print(f"Data type: {image.dtype}")
    print(f"Value range: [{image.min():.3f}, {image.max():.3f}]")
    print(f"Mean: {image.mean():.3f}")
    print(f"Std: {image.std():.3f}")

    # Landmark statistics if present
    if 'landmarks' in sample:
        landmarks = sample['landmarks']
        if isinstance(landmarks, torch.Tensor):
            landmarks = landmarks.cpu().numpy()

        print("\nLandmark Statistics:")
        print(f"Number of landmarks: {len(landmarks)}")
        print(f"Shape: {landmarks.shape}")
        print(f"Data type: {landmarks.dtype}")
        print(f"X range: [{landmarks[:, 0].min():.3f}, {landmarks[:, 0].max():.3f}]")
        print(f"Y range: [{landmarks[:, 1].min():.3f}, {landmarks[:, 1].max():.3f}]")
