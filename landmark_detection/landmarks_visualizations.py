"""
Comprehensive visualization functions for landmark detection tasks.
Includes training monitoring, debugging, and publication-quality visualizations.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from landmark_detection.landmark_utils import compute_euclidean_distance


def _to_display_image(img):
    """Convert a CHW or HWC numpy array to a display-ready 2-D or HWC array in [0, 1]."""
    if img.ndim == 3 and img.shape[0] in [1, 3]:
        img = np.transpose(img, (1, 2, 0))
    if img.ndim == 3 and img.shape[-1] == 1:
        img = img.squeeze(-1)
    if img.max() > 1.0:
        img = img / 255.0
    return np.clip(img, 0, 1)


def plot_training_curves(history, save_dir):
    """
    Plot comprehensive training curves for monitoring.
    
    Args:
        history (dict): Training history with keys: train_loss, val_loss, val_mre, val_ere
        save_dir (str): Directory to save plots
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss curves
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # Metrics curves (MRE, ERE)
    axes[1].plot(epochs, history['val_mre'], 'g-', label='MRE (mm)', linewidth=2)
    if 'val_ere' in history:
        axes[1].plot(epochs, history['val_ere'], 'm--', label='ERE (mm)', linewidth=2, alpha=0.7)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Radial Error (mm)', fontsize=12)
    axes[1].set_title('Validation Landmark Errors', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'training_curves.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Training curves saved to {save_path}")


def plot_validation_predictions(images, gt_landmarks, pred_landmarks, heatmaps, names, 
                                scaling_factors, save_dir, epoch, num_samples=4, 
                                landmark_labels=None, threshold=2.0):
    """
    Plot validation predictions during training for monitoring.
    Uses the same 2-row grid visualization as hip training code:
    - Row 1: Heatmaps overlaid on images using 'inferno' colormap
    - Row 2: Landmarks with color-coded predictions (green ≤threshold, red >threshold)
    
    Args:
        images (ndarray): Images (N, C, H, W)
        gt_landmarks (ndarray): Ground truth landmarks (N, num_landmarks, 2)
        pred_landmarks (ndarray): Predicted landmarks (N, num_landmarks, 2)
        heatmaps (ndarray): Predicted heatmaps (N, num_landmarks, H, W)
        names (list): Image names
        scaling_factors (ndarray): Physical scaling factors
        save_dir (str): Directory to save plots
        epoch (int): Current epoch number
        num_samples (int): Number of samples to visualize
        landmark_labels (list): Landmark labels
        threshold (float): Error threshold for coloring (in mm)
    """
    val_vis_dir = os.path.join(save_dir, 'validation_predictions')
    os.makedirs(val_vis_dir, exist_ok=True)
    
    num_samples = min(num_samples, len(images))
    num_landmarks = gt_landmarks.shape[1]
    
    # Create a single grid figure with all samples
    cols = min(num_samples, 4)  # Max 4 columns
    fig, axes = plt.subplots(2, cols, figsize=(4*cols, 8))
    if cols == 1:
        axes = axes.reshape(2, 1)
    
    for idx in range(num_samples):
        if idx >= cols:
            break
            
        img = images[idx]
        gt_lms = gt_landmarks[idx]
        pred_lms = pred_landmarks[idx]
        scale = scaling_factors[idx]
        
        img = _to_display_image(img)

        # ===============================
        # FIRST ROW: HEATMAPS
        # ===============================
        ax_heatmap = axes[0, idx]
        ax_heatmap.imshow(img, cmap='gray')
        
        # Normalize and overlay heatmaps using 'inferno' colormap (matching hip training code)
        normalized_heatmaps = heatmaps[idx] / np.max(heatmaps[idx], axis=(1, 2), keepdims=True)
        squashed_heatmaps = np.max(normalized_heatmaps, axis=0)
        ax_heatmap.imshow(squashed_heatmaps, cmap='inferno', alpha=0.6)
        
        # Extract sample name for title
        sample_name = os.path.basename(str(names[idx])) if names is not None else f'Sample {idx+1}'
        ax_heatmap.set_title(f'{sample_name}', fontsize=10)
        ax_heatmap.axis('off')
        
        # ===============================
        # SECOND ROW: LANDMARKS
        # ===============================
        ax_landmarks = axes[1, idx]
        ax_landmarks.imshow(img, cmap='gray')
        
        # Compute errors and separate predictions by threshold
        good_predictions = []
        bad_predictions = []
        
        for j in range(num_landmarks):
            error_mm = compute_euclidean_distance(pred_lms[j] * scale, gt_lms[j] * scale)
            if error_mm <= threshold:
                good_predictions.append(pred_lms[j])
            else:
                bad_predictions.append(pred_lms[j])
        
        good_predictions = np.array(good_predictions) if len(good_predictions) > 0 else np.empty((0, 2))
        bad_predictions = np.array(bad_predictions) if len(bad_predictions) > 0 else np.empty((0, 2))
        
        # Plot ground truth landmarks in blue (zorder=1, bottom layer)
        ax_landmarks.scatter(gt_lms[:, 0], gt_lms[:, 1], 
                           color='blue', s=20, label='Ground Truth', zorder=1)
        
        # Plot good predictions (≤threshold) in green (zorder=2)
        if len(good_predictions) > 0:
            ax_landmarks.scatter(good_predictions[:, 0], good_predictions[:, 1], 
                               color='green', s=20, label=f'Pred (≤{threshold}mm)', zorder=2)
        
        # Plot bad predictions (>threshold) in red (zorder=2)
        if len(bad_predictions) > 0:
            ax_landmarks.scatter(bad_predictions[:, 0], bad_predictions[:, 1], 
                               color='red', s=20, label=f'Pred (>{threshold}mm)', zorder=2)
        
        # Draw yellow connection lines between ground truth and predictions (zorder=3, top layer)
        for j in range(num_landmarks):
            ax_landmarks.plot([gt_lms[j, 0], pred_lms[j, 0]],
                            [gt_lms[j, 1], pred_lms[j, 1]],
                            color='yellow', alpha=0.6, linewidth=1.5, zorder=3)
        
        ax_landmarks.set_title(f'{sample_name}', fontsize=10)
        ax_landmarks.axis('off')
        
        # Add legend only to the first subplot of landmarks row
        if idx == 0:
            ax_landmarks.legend(fontsize=8, loc='upper right')
    
    # Hide empty subplots if num_samples < cols
    for idx in range(num_samples, cols):
        axes[0, idx].axis('off')
        axes[1, idx].axis('off')
    
    plt.suptitle(f'Validation - Epoch {epoch}', fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    save_path = os.path.join(val_vis_dir, f'epoch_{epoch:03d}_grid.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Validation predictions (epoch {epoch}) saved to {save_path}")


def plot_test_prediction_grid(images, gt_landmarks, pred_landmarks, heatmaps, names, 
                              scaling_factors, save_dir, landmark_labels=None, 
                              threshold=2.0, max_samples=16):
    """
    Create a comprehensive grid showing test predictions with heatmaps.
    Publication-quality visualization for debugging and analysis.
    
    Args:
        images (ndarray): Test images (N, C, H, W)
        gt_landmarks (ndarray): Ground truth landmarks (N, num_landmarks, 2)
        pred_landmarks (ndarray): Predicted landmarks (N, num_landmarks, 2)
        heatmaps (ndarray): Predicted heatmaps (N, num_landmarks, H, W)
        names (list): Image names
        scaling_factors (ndarray): Physical scaling factors
        save_dir (str): Directory to save plots
        landmark_labels (list): Landmark labels
        threshold (float): Error threshold in mm
        max_samples (int): Maximum samples to show
    """
    test_vis_dir = os.path.join(save_dir, 'test_visualizations')
    os.makedirs(test_vis_dir, exist_ok=True)
    
    num_samples = min(max_samples, len(images))
    cols = 4
    rows = (num_samples + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle('Test Predictions - Landmark Detection', fontsize=16, fontweight='bold', y=0.995)
    
    for idx in range(num_samples):
        row = idx // cols
        col = idx % cols
        ax = axes[row, col]
        
        img = images[idx]
        gt_lms = gt_landmarks[idx]
        pred_lms = pred_landmarks[idx]
        scale = scaling_factors[idx]
        
        img = _to_display_image(img)

        # Show image
        if img.ndim == 2:
            ax.imshow(img, cmap='gray', alpha=0.9)
        else:
            ax.imshow(img, alpha=0.9)
        
        # Plot landmarks
        errors = []
        for j, (gt_lm, pred_lm) in enumerate(zip(gt_lms, pred_lms)):
            error_mm = compute_euclidean_distance(pred_lm * scale, gt_lm * scale)
            errors.append(error_mm)
            color = 'lime' if error_mm <= threshold else 'red'
            
            ax.scatter(gt_lm[0], gt_lm[1], c=color, s=50, marker='o', 
                      alpha=0.6, edgecolors='white', linewidths=1)
            ax.scatter(pred_lm[0], pred_lm[1], c=color, s=50, marker='x', 
                      alpha=0.8, linewidths=2)
        
        mean_error = np.mean(errors)
        max_error = np.max(errors)
        title_color = 'green' if mean_error <= threshold else 'red'
        ax.set_title(f'{names[idx]}\nMean: {mean_error:.2f}mm | Max: {max_error:.2f}mm', 
                    fontsize=9, color=title_color, fontweight='bold')
        ax.axis('off')
    
    # Hide empty subplots
    for idx in range(num_samples, rows * cols):
        row = idx // cols
        col = idx % cols
        axes[row, col].axis('off')
    
    # Add legend
    legend_elements = [
        mpatches.Patch(color='lime', label=f'Good (≤{threshold}mm)'),
        mpatches.Patch(color='red', label=f'Poor (>{threshold}mm)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='black', 
                  markersize=8, label='Ground Truth'),
        plt.Line2D([0], [0], marker='x', color='black', markersize=8, 
                  label='Prediction', linestyle='None')
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=4, 
              bbox_to_anchor=(0.5, -0.02), fontsize=11, frameon=True)
    
    plt.tight_layout(rect=[0, 0.02, 1, 0.99])
    save_path = os.path.join(test_vis_dir, 'test_prediction_grid.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Test prediction grid saved to {save_path}")


def plot_error_distribution(radial_errors, per_landmark_errors, save_dir, 
                            landmark_labels=None, threshold=2.0):
    """
    Plot error distribution analysis for publication and debugging.
    
    Args:
        radial_errors (ndarray): All radial errors (mm)
        per_landmark_errors (list): List of errors per landmark
        save_dir (str): Directory to save plots
        landmark_labels (list): Landmark labels
        threshold (float): Clinical threshold
    """
    test_vis_dir = os.path.join(save_dir, 'test_visualizations')
    os.makedirs(test_vis_dir, exist_ok=True)
    
    fig = plt.figure(figsize=(16, 5))
    gs = GridSpec(1, 3, figure=fig)
    
    # 1. Overall error histogram
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(radial_errors, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    ax1.axvline(threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold ({threshold}mm)')
    ax1.axvline(np.median(radial_errors), color='green', linestyle='--', linewidth=2, label=f'Median ({np.median(radial_errors):.2f}mm)')
    ax1.set_xlabel('Radial Error (mm)', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Overall Error Distribution', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 2. Cumulative distribution (SDR)
    ax2 = fig.add_subplot(gs[0, 1])
    sorted_errors = np.sort(radial_errors)
    cumulative = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors) * 100
    ax2.plot(sorted_errors, cumulative, linewidth=2, color='darkblue')
    ax2.axvline(threshold, color='red', linestyle='--', linewidth=2, alpha=0.7)
    
    # Mark key thresholds
    for thresh in [2.0, 2.5, 3.0, 4.0]:
        if thresh <= sorted_errors.max():
            sdr = (sorted_errors <= thresh).mean() * 100
            ax2.axhline(sdr, color='gray', linestyle=':', alpha=0.5, linewidth=1)
            ax2.text(thresh, sdr + 2, f'{thresh}mm: {sdr:.1f}%', fontsize=9)
    
    ax2.set_xlabel('Radial Error (mm)', fontsize=12)
    ax2.set_ylabel('Success Detection Rate (%)', fontsize=12)
    ax2.set_title('Cumulative Error Distribution (SDR)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 105])
    
    # 3. Per-landmark error boxplot
    ax3 = fig.add_subplot(gs[0, 2])
    num_landmarks = len(per_landmark_errors)
    
    bp = ax3.boxplot(per_landmark_errors, labels=landmark_labels or [f'LM{i+1}' for i in range(num_landmarks)],
                     patch_artist=True, showmeans=True)
    
    # Color boxes
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
        patch.set_alpha(0.7)
    
    ax3.axhline(threshold, color='red', linestyle='--', linewidth=2, alpha=0.7, label=f'Threshold ({threshold}mm)')
    ax3.set_ylabel('Radial Error (mm)', fontsize=12)
    ax3.set_xlabel('Landmark', fontsize=12)
    ax3.set_title('Per-Landmark Error Distribution', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3, axis='y')
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    save_path = os.path.join(test_vis_dir, 'error_distribution.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Error distribution saved to {save_path}")


def plot_per_landmark_metrics(per_landmark_mre, per_landmark_std, save_dir, 
                              landmark_labels=None, threshold=2.0):
    """
    Plot per-landmark performance metrics as bar chart.
    
    Args:
        per_landmark_mre (list): Mean radial error per landmark
        per_landmark_std (list): Std of radial error per landmark
        save_dir (str): Directory to save plots
        landmark_labels (list): Landmark labels
        threshold (float): Clinical threshold
    """
    test_vis_dir = os.path.join(save_dir, 'test_visualizations')
    os.makedirs(test_vis_dir, exist_ok=True)
    
    num_landmarks = len(per_landmark_mre)
    x = np.arange(num_landmarks)
    labels = landmark_labels or [f'LM{i+1}' for i in range(num_landmarks)]
    
    fig, ax = plt.subplots(figsize=(max(10, num_landmarks * 0.6), 6))
    
    # Create bars
    colors = ['green' if mre <= threshold else 'orange' if mre <= threshold * 1.5 else 'red' 
              for mre in per_landmark_mre]
    bars = ax.bar(x, per_landmark_mre, yerr=per_landmark_std, color=colors, alpha=0.7, 
                  capsize=5, edgecolor='black', linewidth=1.2)
    
    # Add threshold line
    ax.axhline(threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold ({threshold}mm)')
    
    # Add value labels on bars
    for i, (bar, mre, std) in enumerate(zip(bars, per_landmark_mre, per_landmark_std)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.1,
                f'{mre:.2f}±{std:.2f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_ylabel('Radial Error (mm)', fontsize=13, fontweight='bold')
    ax.set_xlabel('Landmark', fontsize=13, fontweight='bold')
    ax.set_title('Per-Landmark Mean Radial Error', fontsize=15, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    save_path = os.path.join(test_vis_dir, 'per_landmark_metrics.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Per-landmark metrics saved to {save_path}")


def plot_test_samples_grid(images, gt_landmarks, pred_landmarks, heatmaps, names,
                          scaling_factors, save_dir, threshold=2.0):
    """
    Create a comprehensive grid showing ALL test samples with 2 rows per sample:
    - Row 1: Heatmaps overlaid on images using 'inferno' colormap
    - Row 2: Landmarks with color-coded predictions (green ≤threshold, red >threshold)
    
    Args:
        images (ndarray): Test images (N, C, H, W)
        gt_landmarks (ndarray): Ground truth landmarks (N, num_landmarks, 2)
        pred_landmarks (ndarray): Predicted landmarks (N, num_landmarks, 2)
        heatmaps (ndarray): Predicted heatmaps (N, num_landmarks, H, W)
        names (list): Image names
        scaling_factors (ndarray): Physical scaling factors
        save_dir (str): Directory to save plots
        threshold (float): Error threshold in mm
    """
    test_vis_dir = os.path.join(save_dir, 'test_visualizations')
    os.makedirs(test_vis_dir, exist_ok=True)
    
    num_samples = len(images)
    num_landmarks = gt_landmarks.shape[1]
    
    # Determine grid layout (max 4 columns)
    cols = min(num_samples, 4)
    rows = 2 * ((num_samples + cols - 1) // cols)  # 2 rows per sample batch
    
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
    
    # Handle single row/column cases
    if rows == 2 and cols == 1:
        axes = axes.reshape(2, 1)
    elif rows == 2:
        axes = axes.reshape(2, cols)
    elif cols == 1:
        axes = axes.reshape(rows, 1)
    
    fig.suptitle('Test Set - All Samples (Heatmaps + Landmarks)', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    for idx in range(num_samples):
        col_idx = idx % cols
        row_block = (idx // cols) * 2  # Each sample occupies 2 rows
        
        img = images[idx]
        gt_lms = gt_landmarks[idx]
        pred_lms = pred_landmarks[idx]
        scale = scaling_factors[idx]
        
        img = _to_display_image(img)

        # ===============================
        # FIRST ROW: HEATMAPS
        # ===============================
        ax_heatmap = axes[row_block, col_idx] if rows > 1 else axes[col_idx]
        ax_heatmap.imshow(img, cmap='gray')
        
        # Normalize and overlay heatmaps using 'inferno' colormap
        normalized_heatmaps = heatmaps[idx] / np.max(heatmaps[idx], axis=(1, 2), keepdims=True)
        squashed_heatmaps = np.max(normalized_heatmaps, axis=0)
        ax_heatmap.imshow(squashed_heatmaps, cmap='inferno', alpha=0.6)
        
        # Extract sample name for title
        sample_name = os.path.basename(str(names[idx]))
        ax_heatmap.set_title(f'{sample_name}\nHeatmaps', fontsize=10, fontweight='bold')
        ax_heatmap.axis('off')
        
        # ===============================
        # SECOND ROW: LANDMARKS
        # ===============================
        ax_landmarks = axes[row_block + 1, col_idx] if rows > 1 else axes[col_idx]
        ax_landmarks.imshow(img, cmap='gray')
        
        # Compute errors and separate predictions by threshold
        good_predictions = []
        bad_predictions = []
        
        for j in range(num_landmarks):
            error_mm = compute_euclidean_distance(pred_lms[j] * scale, gt_lms[j] * scale)
            if error_mm <= threshold:
                good_predictions.append(pred_lms[j])
            else:
                bad_predictions.append(pred_lms[j])
        
        good_predictions = np.array(good_predictions) if len(good_predictions) > 0 else np.empty((0, 2))
        bad_predictions = np.array(bad_predictions) if len(bad_predictions) > 0 else np.empty((0, 2))
        
        # Plot ground truth landmarks in blue (zorder=1, bottom layer)
        ax_landmarks.scatter(gt_lms[:, 0], gt_lms[:, 1], 
                           color='blue', s=20, label='Ground Truth', zorder=1)
        
        # Plot good predictions (≤threshold) in green (zorder=2)
        if len(good_predictions) > 0:
            ax_landmarks.scatter(good_predictions[:, 0], good_predictions[:, 1], 
                               color='green', s=20, label=f'Pred (≤{threshold}mm)', zorder=2)
        
        # Plot bad predictions (>threshold) in red (zorder=2)
        if len(bad_predictions) > 0:
            ax_landmarks.scatter(bad_predictions[:, 0], bad_predictions[:, 1], 
                               color='red', s=20, label=f'Pred (>{threshold}mm)', zorder=2)
        
        # Draw yellow connection lines between ground truth and predictions (zorder=3, top layer)
        for j in range(num_landmarks):
            ax_landmarks.plot([gt_lms[j, 0], pred_lms[j, 0]],
                            [gt_lms[j, 1], pred_lms[j, 1]],
                            color='yellow', alpha=0.6, linewidth=1.5, zorder=3)
        
        # Compute mean error for this sample
        sample_errors = [compute_euclidean_distance(pred_lms[j] * scale, gt_lms[j] * scale) 
                        for j in range(num_landmarks)]
        mean_error = np.mean(sample_errors)
        
        title_color = 'green' if mean_error <= threshold else 'red'
        ax_landmarks.set_title(f'{sample_name}\nMean Error: {mean_error:.2f}mm', 
                             fontsize=10, fontweight='bold', color=title_color)
        ax_landmarks.axis('off')
        
        # Add legend only to the first subplot of landmarks row
        if idx == 0:
            ax_landmarks.legend(fontsize=8, loc='upper right')
    
    # Hide empty subplots if num_samples doesn't fill the grid
    total_cells = (rows // 2) * cols
    for idx in range(num_samples, total_cells):
        col_idx = idx % cols
        row_block = (idx // cols) * 2
        axes[row_block, col_idx].axis('off')
        axes[row_block + 1, col_idx].axis('off')
    
    plt.tight_layout(rect=[0, 0.01, 1, 0.99])
    
    save_path = os.path.join(test_vis_dir, 'test_all_samples_grid.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Test samples grid (all samples) saved to {save_path}")


def plot_landmarks_test_results(images, gt_landmarks, pred_landmarks, heatmaps, names, 
                                scaling_factors, save_dir, num_landmarks, radial_errors,
                                per_landmark_errors, labels=None, threshold=2.0):
    """
    Generate all test result visualizations.
    
    Args:
        images (ndarray): Test images
        gt_landmarks (ndarray): Ground truth landmarks
        pred_landmarks (ndarray): Predicted landmarks
        heatmaps (ndarray): Predicted heatmaps
        names (list): Image names
        scaling_factors (ndarray): Physical scaling factors
        save_dir (str): Directory to save plots
        num_landmarks (int): Number of landmarks
        radial_errors (ndarray): All radial errors
        per_landmark_errors (list): Errors per landmark
        labels (list): Landmark labels
        threshold (float): Error threshold
    """
    print("\nGenerating landmark detection visualizations...")
    
    # 1. Complete test samples grid with heatmaps and landmarks (ALL samples)
    plot_test_samples_grid(images, gt_landmarks, pred_landmarks, heatmaps, names,
                          scaling_factors, save_dir, threshold)
    
    # 2. Prediction grid with heatmaps (subset for quick overview)
    plot_test_prediction_grid(images, gt_landmarks, pred_landmarks, heatmaps, names,
                              scaling_factors, save_dir, labels, threshold, max_samples=16)
    
    # 3. Error distribution analysis
    plot_error_distribution(radial_errors, per_landmark_errors, save_dir, labels, threshold)
    
    # 4. Per-landmark performance metrics
    per_landmark_mre = [np.mean(errors) for errors in per_landmark_errors]
    per_landmark_std = [np.std(errors) for errors in per_landmark_errors]
    plot_per_landmark_metrics(per_landmark_mre, per_landmark_std, save_dir, labels, threshold)
    
    print("✓ All landmark visualizations completed")


