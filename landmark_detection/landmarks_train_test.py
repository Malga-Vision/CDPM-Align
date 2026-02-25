"""
Landmark detection training and evaluation functions.
"""

import os
import numpy as np
import torch
from torch import nn
from tqdm.auto import tqdm
from utils.utils import EarlyStopping, save_model, load_model, clean_cache
from landmark_detection.landmark_utils import (
    extract_landmark, compute_euclidean_distance, compute_mean_std,
    get_mode_probability, compute_sdr, calculate_ere, compute_p95
)
from landmark_detection.landmark_losses import CustomNLLLoss
from landmark_detection.landmarks_visualizations import (
    plot_training_curves,
    plot_validation_predictions,
    plot_landmarks_test_results,
)

def train_step(model, device, dataloader, loss_fn, optimizer, gradient_accumulation_steps=1):
    """Train model for one epoch."""
    model.train()
    train_loss = 0.0
    sigma = dataloader.dataset.sigma
    use_argmax = sigma <= 1

    ## --------------------------------------------- ##

    use_cross_entropy = isinstance(loss_fn, nn.CrossEntropyLoss)
    use_custom_nll = isinstance(loss_fn, CustomNLLLoss)

    for batch, data in enumerate(tqdm(dataloader, desc="Training", leave=False)):
        X = data['image'].to(device)
        y = data['target'].to(device)

        y_pred = model(X)

        if use_cross_entropy:
            loss = loss_fn(y_pred, y)
        elif use_custom_nll:
            y_pred = loss_fn.two_d_softmax(y_pred)
            loss = loss_fn(y_pred, y)
        
        loss = loss / gradient_accumulation_steps
        loss.backward()
        train_loss += loss.item()

        if ((batch + 1) % gradient_accumulation_steps == 0) or (batch + 1 == len(dataloader)):
            optimizer.step()
            optimizer.zero_grad()

    train_loss /= len(dataloader)
    return train_loss


def validate_step(model, device, dataloader, loss_fn, save_visualizations=False, save_dir=None, epoch=None):
    """Validate model performance and optionally save visualizations."""
    model.eval()
    val_loss = 0.0
    all_radial_errors = []
    all_expected_radial_errors = []

    sigma = getattr(dataloader.dataset, 'sigma', 1)
    use_argmax = sigma <= 1

    # Collect data for visualization if needed
    if save_visualizations:
        collected_images = []
        collected_gt_landmarks = []
        collected_pred_landmarks = []
        collected_heatmaps = []
        collected_names = []
        collected_scaling_factors = []

    use_cross_entropy = isinstance(loss_fn, nn.CrossEntropyLoss)
    use_custom_nll = isinstance(loss_fn, CustomNLLLoss)

    with torch.no_grad():
        for data in tqdm(dataloader, desc="Validating", leave=False):
            X = data['image'].to(device)
            y = data['target'].to(device)
            gt_landmarks = data['landmarks']
            scaling_factors = data['physical_scaling_factor']
            names = data.get('name', [f"img_{i}" for i in range(len(X))])

            y_pred = model(X)

            if use_cross_entropy:
                loss = loss_fn(y_pred, y)
            elif use_custom_nll:
                y_pred = loss_fn.two_d_softmax(y_pred)
                loss = loss_fn(y_pred, y)
            else:
                loss = loss_fn(y_pred, y)
            
            val_loss += loss.item()
            
            # Collect first batch for visualization
            if save_visualizations and len(collected_images) == 0:
                collected_images.append(X.cpu().numpy())
                collected_heatmaps.append(y_pred.cpu().numpy())
                collected_names.extend(names)
                collected_scaling_factors.append(scaling_factors.cpu().numpy())
                collected_gt_landmarks.append(gt_landmarks.cpu().numpy())
                collected_pred_landmarks.append([])

            # Compute errors
            y_pred_np = y_pred.detach().cpu().numpy()
            gt_landmarks_np = gt_landmarks.cpu().numpy()
            scaling_factors_np = scaling_factors.cpu().numpy()
            
            for i in range(y_pred.shape[0]):
                img_pred_landmarks = []
                
                for j in range(y_pred.shape[1]):
                    pred_landmark = extract_landmark(y_pred_np[i, j], use_argmax)
                    gt_landmark = gt_landmarks_np[i, j]
                    scale = scaling_factors_np[i]

                    pred_scaled = pred_landmark * scale
                    gt_scaled = gt_landmark * scale
                    
                    distance = compute_euclidean_distance(pred_scaled, gt_scaled)
                    all_radial_errors.append(distance)
                    
                    #expected_distance = calculate_ere(y_pred_np[i, j], pred_scaled, scale)
                    expected_distance = 0
                    all_expected_radial_errors.append(expected_distance)
                    
                    # Collect landmarks for visualization (for the first batch only)
                    if save_visualizations and len(collected_images) == 1:
                        img_pred_landmarks.append(pred_landmark)
          
                # Append landmarks for this image (for the first batch only)
                if save_visualizations and len(collected_images) == 1:
                    collected_pred_landmarks[0].append(img_pred_landmarks)

    val_loss /= len(dataloader)
    mre, _ = compute_mean_std(all_radial_errors)
    ere, _ = compute_mean_std(all_expected_radial_errors)

    # Generate validation visualization
    if save_visualizations and save_dir is not None:
        images = collected_images[0]
        gt_lms = collected_gt_landmarks[0]
        pred_lms = np.array(collected_pred_landmarks[0])
        heatmaps = collected_heatmaps[0]
        scales = collected_scaling_factors[0]
        
        plot_validation_predictions(
            images, gt_lms, pred_lms, heatmaps, collected_names,
            scales, save_dir, epoch, num_samples=4, threshold=2.0
        )

    return val_loss, mre, ere


def train_and_validate(model, device, train_dataloader, val_dataloader, optimizer, scheduler, loss_fn,
                       epochs=10, save_path=None, patience=10, gradient_accumulation_steps=1,
                       continue_training=False):
    """Train and validate landmark detection model."""
    if save_path:
        os.makedirs(save_path, exist_ok=True)
    
    early_stopping = EarlyStopping(patience=patience, save_dir=save_path, verbose=False)
    
    start_epoch = 0
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_mre': [],
        'val_ere': []
    }
    
    if continue_training and save_path and os.path.exists(os.path.join(save_path, "last_checkpoint.pt")):
        checkpoint_info = load_model(os.path.join(save_path, "last_checkpoint.pt"), model, optimizer, scheduler, device)
        start_epoch = checkpoint_info.get('epoch', 0)
        early_stopping.best_val_loss = checkpoint_info.get('best_val_loss', float('inf'))
        history = checkpoint_info.get('history', history)
        print(f"Resumed training from epoch {start_epoch}")
    
    model = model.to(device)
 
    epoch_pbar = tqdm(range(start_epoch, epochs), initial=start_epoch, total=epochs, desc="Training")
    for epoch in epoch_pbar:
        
        train_loss = train_step(model, device, train_dataloader, loss_fn, optimizer, gradient_accumulation_steps)
        
        # Save validation visualizations every 20 epochs and on last epoch
        save_vis = (epoch % 20 == 0) or (epoch == epochs - 1)
        val_loss, val_mre, val_ere = validate_step(
            model, device, val_dataloader, loss_fn, 
            save_visualizations=save_vis, save_dir=save_path, epoch=epoch+1
        )
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_mre'].append(val_mre)
        history['val_ere'].append(val_ere)
        
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss)
        elif scheduler is not None:
            scheduler.step()
        
        epoch_pbar.set_postfix({
            "Train Loss": f"{train_loss:.4f}",
            "Val Loss": f"{val_loss:.4f}",
            "MRE": f"{val_mre:.2f}",
            "ERE": f"{val_ere:.2f}",
            "LR": f"{optimizer.param_groups[0]['lr']:.2e}"
        })
        
        if save_path:
            save_model(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                save_dir=save_path,
                epoch=epoch,
                history=history,
                filename="last_checkpoint.pt"
            )
        
        if early_stopping:
            early_stopping(val_loss, model, optimizer, scheduler, epoch, history=history)
            if early_stopping.early_stop:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

        #clean_cache()
    
    # Plot training curves at the end
    if save_path:
        plot_training_curves(history, save_path)
    
    return model, history


## -----------------------------------------------------------------------------------------------------------------##
##                                                  EVALUATION PART                                                 ##
## -----------------------------------------------------------------------------------------------------------------##

def _compute_landmark_metrics(metrics, num_landmarks, radial_errors_array, expected_radial_errors_array):
    """Compute aggregate and per-landmark statistics from collected per-sample errors."""
    per_landmark_mre = [np.mean(errors) for errors in metrics['per_landmark_errors']]
    per_landmark_std = [np.std(errors) for errors in metrics['per_landmark_errors']]
    sdr_statistics, thresholds = compute_sdr(radial_errors_array)
    p95 = compute_p95(radial_errors_array)
    per_landmark_p95 = [compute_p95(np.array(e)) for e in metrics['per_landmark_errors']]

    return {
        'loss': np.mean(metrics['losses']),
        'mre': radial_errors_array.mean(),
        'mre_std': radial_errors_array.std(),
        'ere': expected_radial_errors_array.mean(),
        'ere_std': expected_radial_errors_array.std(),
        'mode_probs': np.array(metrics['mode_probabilities']),
        'pixel_sizes': np.array(metrics['pixel_size']),
        'per_landmark_mre': per_landmark_mre,
        'per_landmark_std': per_landmark_std,
        'sdr': dict(zip(thresholds, sdr_statistics)),
        'p95': p95,
        'per_landmark_p95': per_landmark_p95,
        'radial_errors': radial_errors_array,
    }


def evaluate(model, device, test_dataloader, loss_fn, num_landmarks, save_dir):
    """
    Evaluate landmark detection model.

    Args:
        model: PyTorch model to evaluate
        device: Device to run evaluation on
        test_dataloader: DataLoader for test data
        loss_fn: Loss function
        num_landmarks: Number of landmarks
        save_dir: Directory to save results

    Returns:
        Dictionary containing all evaluation metrics
    """
    print("\nEvaluating landmarks")
    
    model = model.to(device)
    model.eval()

    sigma = getattr(test_dataloader.dataset, 'sigma', 1)
    use_argmax = sigma <= 1
    
    metrics = {
        'losses': [],
        'radial_errors': [],
        'expected_radial_errors': [],
        'mode_probabilities': [],
        'pixel_size': [],
        'per_landmark_errors': [[] for _ in range(num_landmarks)]
    }
    
    # Always collect visualization data
    collected_images = []
    collected_gt_landmarks = []
    collected_pred_landmarks = []
    collected_heatmaps = []
    collected_names = []
    collected_scaling_factors = []
    collected_labels = []
    
    # Determine loss function type
    use_cross_entropy = isinstance(loss_fn, nn.CrossEntropyLoss)
    use_custom_nll = isinstance(loss_fn, CustomNLLLoss)

    with torch.no_grad():
        for data in tqdm(test_dataloader, desc="Evaluating"):
            X = data['image'].to(device)
            y = data['target'].to(device) 
            ground_truth_landmarks = data['landmarks']
            scaling_factors = data['physical_scaling_factor']
            names = data['name']
            label_fields = data.get('label_fields', None)

            y_pred = model(X)

            # Compute loss
            if use_cross_entropy:
                loss = loss_fn(y_pred, y)
            elif use_custom_nll:
                y_pred = loss_fn.two_d_softmax(y_pred)
                loss = loss_fn(y_pred, y)
            else:
                loss = loss_fn(y_pred, y)
            
            metrics['losses'].append(loss.item())
            
            # Always collect for visualization
            collected_images.append(X.cpu().numpy())
            collected_heatmaps.append(y_pred.cpu().numpy())
            collected_names.extend(names)
            collected_scaling_factors.append(scaling_factors.cpu().numpy())
            collected_gt_landmarks.append(ground_truth_landmarks.cpu().numpy())
            if label_fields is not None:
                collected_labels.extend(label_fields)

            batch_size = y_pred.shape[0]
            y_pred_np = y_pred.detach().cpu().numpy()
            gt_landmarks_np = ground_truth_landmarks.cpu().numpy()
            scaling_factors_np = scaling_factors.cpu().numpy()
            
            batch_pred_landmarks = []
            
            for i in range(batch_size):
                img_pred_landmarks = np.zeros((num_landmarks, 2))
                
                for j in range(num_landmarks):
                    pred_heatmap = y_pred_np[i, j]
                    pred_landmark = extract_landmark(pred_heatmap, use_argmax)
                    gt_landmark = gt_landmarks_np[i, j]
                    scale = scaling_factors_np[i]
                    
                    pred_scaled = pred_landmark * scale
                    gt_scaled = gt_landmark * scale
                    
                    # Physical distance
                    distance = compute_euclidean_distance(pred_scaled, gt_scaled)
                    metrics['radial_errors'].append(distance)
                    
                    # Expected radial error
                    expected_distance = calculate_ere(pred_heatmap, pred_scaled, scale)
                    metrics['expected_radial_errors'].append(expected_distance)
                    
                    # Additional metrics
                    mode_prob = get_mode_probability(pred_heatmap)
                    metrics['mode_probabilities'].append(mode_prob)
                    metrics['pixel_size'].append(scale)
                    metrics['per_landmark_errors'][j].append(distance)
                    
                    img_pred_landmarks[j] = pred_landmark
                
                batch_pred_landmarks.append(img_pred_landmarks)
            
            collected_pred_landmarks.extend(batch_pred_landmarks)
    
    radial_errors_array = np.array(metrics['radial_errors'])
    expected_radial_errors_array = np.array(metrics['expected_radial_errors'])

    results = _compute_landmark_metrics(metrics, num_landmarks, radial_errors_array, expected_radial_errors_array)

    all_gt_landmarks = np.concatenate(collected_gt_landmarks, axis=0)
    all_scaling_factors = np.concatenate(collected_scaling_factors, axis=0)
    all_pred_landmarks = np.array(collected_pred_landmarks)
    
    # Always generate visualizations
    all_images = np.concatenate(collected_images, axis=0)
    all_heatmaps = np.concatenate(collected_heatmaps, axis=0)
    
    # Extract landmark labels
    sample_labels = None
    if collected_labels:
        for label_entry in collected_labels:
            if label_entry and len(label_entry) > 0:
                sample_labels = label_entry[:num_landmarks]
                break
    
    # Limit visualization to first 50 samples to avoid matplotlib memory issues
    max_vis_samples = min(50, len(all_images))
    plot_landmarks_test_results(
        all_images[:max_vis_samples], 
        all_gt_landmarks[:max_vis_samples], 
        all_pred_landmarks[:max_vis_samples], 
        all_heatmaps[:max_vis_samples],
        collected_names[:max_vis_samples], 
        all_scaling_factors[:max_vis_samples], 
        save_dir, num_landmarks,
        radial_errors_array, metrics['per_landmark_errors'],
        labels=sample_labels, threshold=2.0
    )

    return results
