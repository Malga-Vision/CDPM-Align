"""
DDPM training and testing implementation.
"""

import os
import time
from typing import Dict, Any, Tuple, List, Optional

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torchmetrics import MeanSquaredError
from torchmetrics.image.psnr import PeakSignalNoiseRatio
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.fid import FrechetInceptionDistance
import matplotlib.pyplot as plt

from .ddpm_model import DDPM


def save_generated_images(image_lists, save_path, title, image_titles=None):
    """Save generated images in a grid format (rows: real/generated/diff, cols: batch samples)."""
    # Convert tensors to numpy
    processed_lists = []
    for img_list in image_lists:
        if isinstance(img_list, torch.Tensor):
            img_list = img_list.detach().cpu().numpy()
        processed_lists.append(img_list)

    num_rows = len(processed_lists)
    num_cols = processed_lists[0].shape[0]

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 3, num_rows * 3))

    if num_rows == 1 and num_cols == 1:
        axes = np.array([[axes]])
    elif num_rows == 1:
        axes = axes.reshape(1, -1)
    elif num_cols == 1:
        axes = axes.reshape(-1, 1)

    row_labels = ['Real', 'Generated', 'Difference']

    for row_idx, img_batch in enumerate(processed_lists):
        for col_idx in range(num_cols):
            ax = axes[row_idx, col_idx]

            img = img_batch[col_idx]

            if img.shape[0] in [1, 3]:
                img = np.transpose(img, (1, 2, 0))

            if img.shape[-1] == 1:
                img = img.squeeze(-1)

            img_min, img_max = img.min(), img.max()
            if img_max > img_min:
                img = (img - img_min) / (img_max - img_min)

            if img.ndim == 2:
                ax.imshow(img, cmap='gray', vmin=0, vmax=1)
            else:
                ax.imshow(np.clip(img, 0, 1))

            if col_idx == 0:
                ax.set_ylabel(row_labels[row_idx] if row_idx < len(row_labels) else f'Row {row_idx}',
                            fontsize=10, fontweight='bold')

            if row_idx == 0 and image_titles is not None and col_idx < len(image_titles):
                ax.set_title(image_titles[col_idx], fontsize=8)

            ax.axis('off')

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved generated images to: {save_path}")


def initialize_ddpm(config: Dict[str, Any], phase: str, device: torch.device) -> DDPM:
    """Initialize a DDPM model for the given phase ('train' or 'test')."""
    if phase == "train":
        ddpm = DDPM(
            image_size=max(config['dataset']['image_size']),
            channels=config['dataset']['image_channels'],
            device=device,
            lr=config['training']['learning_rate'],
            optimizer=config['training']['optimizer'],
            timesteps=config['model']['beta_schedule']['train']['n_timestep'],
            beta_schedule=config['model']['beta_schedule']['train']['schedule'],
            beta_start=config['model']['beta_schedule']['train']['linear_start'],
            beta_end=config['model']['beta_schedule']['train']['linear_end'],
            unet_self_condition=config['model']['unet']['self_condition'],
            unet_channels=config['model']['unet']['channel_mults'],
            unet_res_blocks=config['model']['unet']['res_blocks'],
            unet_att_heads=config['model']['unet']['num_head_channels'],
            unet_att_res=config['model']['unet']['attn_res'],
            use_ema=config['model']['use_ema'],
        )
        
    elif phase == "test":
        ddpm = DDPM(
            image_size=max(config['dataset']['image_size']),
            channels=config['dataset']['image_channels'],
            device=device,
            lr=config['training']['learning_rate'],
            optimizer=config['training']['optimizer'],
            timesteps=config['model']['beta_schedule']['test']['n_timestep'],
            beta_schedule=config['model']['beta_schedule']['test']['schedule'],
            beta_start=config['model']['beta_schedule']['test']['linear_start'],
            beta_end=config['model']['beta_schedule']['test']['linear_end'],
            unet_self_condition=config['model']['unet']['self_condition'],
            unet_channels=config['model']['unet']['channel_mults'],
            unet_res_blocks=config['model']['unet']['res_blocks'],
            unet_att_heads=config['model']['unet']['num_head_channels'],
            unet_att_res=config['model']['unet']['attn_res'],
        )
    else:
        raise ValueError(f"Phase {phase} is not valid. Must be either 'train' or 'test'")

    return ddpm


def save_model(model, optimizer, epoch, n_iter, loss, save_path, name="last_model"):
    """Save model checkpoint."""
    torch.save({
        "epoch": epoch,
        "n_iter": n_iter,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }, f"{save_path}/{name}.pt")


def save_ema_model(ema_model, epoch, n_iter, save_path, name="last_ema_model"):
    """Save EMA model checkpoint."""
    torch.save({
        "epoch": epoch,
        "n_iter": n_iter,
        "model_state_dict": ema_model.state_dict(),
    }, f"{save_path}/{name}.pt")


def load_model(save_model_path, device, ddpm):
    """Load saved model checkpoint into DDPM."""
    assert os.path.exists(save_model_path), f"Model {save_model_path} does not exist"
    checkpoint = torch.load(save_model_path, map_location=device)
    ddpm.model.load_state_dict(checkpoint['model_state_dict'])
    ddpm.optimizer.load_state_dict(checkpoint['optimizer_state_dict']) if "optimizer_state_dict" in checkpoint else None
    epoch = checkpoint.get('epoch', 'undefined')
    n_iter = checkpoint.get('n_iter', 'undefined')
    loss = checkpoint.get('loss', np.inf)
    del checkpoint
    print(f"Model loaded for epoch: {epoch} and iteration: {n_iter} with loss: {loss}")
    return epoch, n_iter, loss


def load_ema_model(save_model_path, device, ddpm):
    """Load saved EMA model checkpoint."""
    assert os.path.exists(save_model_path), f"Model {save_model_path} does not exist"
    ema_checkpoint = torch.load(save_model_path, map_location=device)
    ddpm.ema_model.load_state_dict(ema_checkpoint['model_state_dict'])
    ema_epoch = ema_checkpoint.get('epoch', 0)
    ema_n_iter = ema_checkpoint.get('n_iter', 0)
    del ema_checkpoint
    print(f"EMA Model loaded for epoch: {ema_epoch} and iteration: {ema_n_iter}")
    return ema_epoch, ema_n_iter


def compute_diff(x, x_hat):
    """Compute normalized, thresholded absolute difference between two tensors."""
    assert x.device == x_hat.device, "Tensors must be on the same device"
    assert x.shape == x_hat.shape, "Tensors must have the same shape"

    diff = torch.abs(x - x_hat)
    diff = (diff - diff.min()) / (diff.max() - diff.min() + 1e-8)
    diff[diff < 0.1] = 0
    return diff


def compute_diffusion_metrics(ddpm, x, x_names, batch_size, timesteps, channels,
                             use_ema, epoch, n_iter, batch_idx, root_path, device,
                             metrics_dict=None):
    """Compute SSIM, MSE, PSNR, and FID metrics for the diffusion model."""
    if metrics_dict is None:
        metrics_dict = {
            'psnr': PeakSignalNoiseRatio(data_range=None, reduction="elementwise_mean"),
            'ssim': StructuralSimilarityIndexMeasure(data_range=None, reduction="elementwise_mean"),
            'mse': MeanSquaredError(),
            'fid': FrechetInceptionDistance(normalize=True)
        }

    ddpm.model.eval()
    x_hat = ddpm.sample(model=ddpm.model, x_cond=x, batch_size=batch_size, timesteps=timesteps)

    results = {
        'model': {},
        'ema': {} if use_ema else None
    }

    x_cpu = x.detach().cpu()
    x_hat_cpu = x_hat.detach().cpu()

    ssim_metric = metrics_dict['ssim'](x_hat_cpu, x_cpu)
    mse_metric = metrics_dict['mse'](x_hat_cpu, x_cpu)
    psnr_metric = metrics_dict['psnr'](x_hat_cpu, x_cpu)

    real_images = x_cpu if channels == 3 else x_cpu.repeat(1, 3, 1, 1)
    fake_images = x_hat_cpu if channels == 3 else x_hat_cpu.repeat(1, 3, 1, 1)

    if batch_size > 1:
        metrics_dict['fid'].update(real_images, real=True)
        metrics_dict['fid'].update(fake_images.clamp(0, 1), real=False)
        fid_score = metrics_dict['fid'].compute()
    else:
        fid_score = 0.0

    diff = compute_diff(real_images, fake_images)

    train_imgs_path = f"{root_path}/images/train"
    os.makedirs(train_imgs_path, exist_ok=True)
    image_titles = list(x_names[:batch_size])
    save_generated_images([real_images, fake_images, diff], 
               f"{train_imgs_path}/train_epoch{epoch}_iteration{n_iter}_batch{batch_idx}.jpg", 
               f"Epoch {epoch} - Iteration {n_iter} - Batch {batch_idx} - Timesteps {timesteps}", 
               image_titles)

    results['model'] = {
        'ssim': ssim_metric.item(),
        'mse': mse_metric.item(),
        'psnr': psnr_metric.item(),
        'fid': fid_score,
        'images': {
            'real': real_images,
            'fake': fake_images,
            'diff': diff
        }
    }

    if use_ema:
        ema_x_hat = ddpm.sample(model=ddpm.ema_model, x_cond=x, batch_size=batch_size, timesteps=timesteps)
        ema_x_hat_cpu = ema_x_hat.detach().cpu()

        ema_ssim_metric = metrics_dict['ssim'](ema_x_hat_cpu, x_cpu)
        ema_mse_metric = metrics_dict['mse'](ema_x_hat_cpu, x_cpu)
        ema_psnr_metric = metrics_dict['psnr'](ema_x_hat_cpu, x_cpu)

        ema_fake_images = ema_x_hat_cpu if channels == 3 else ema_x_hat_cpu.repeat(1, 3, 1, 1)

        if batch_size > 1:
            metrics_dict['fid'].update(ema_fake_images.clamp(0, 1), real=False)
            ema_fid_score = metrics_dict['fid'].compute()
        else:
            ema_fid_score = 0.0

        ema_diff = compute_diff(real_images, ema_fake_images)

        save_generated_images([real_images, ema_fake_images, ema_diff],
                   f"{train_imgs_path}/train_epoch{epoch}_iteration{n_iter}_batch{batch_idx}_ema.jpg", 
                   f"Epoch {epoch} - Iteration {n_iter} - Batch {batch_idx} - Timesteps {timesteps}", 
                   image_titles)

        results['ema'] = {
            'ssim': ema_ssim_metric.item(),
            'mse': ema_mse_metric.item(),
            'psnr': ema_psnr_metric.item(),
            'fid': ema_fid_score,
            'images': {
                'fake': ema_fake_images,
                'diff': ema_diff
            }
        }

    ddpm.model.train()
    
    return results


def generate_metrics_message(results, curr_lr, epoch, n_iter):
    """Generate a formatted message for logging metrics."""
    model_metrics = results['model']
    message = (
        f"<b>Epoch/Iteration {epoch}/{n_iter}</b>\n"
        f"  • <b>SSIM:</b> {model_metrics['ssim']:.4f} \n"
        f"  • <b>MSE:</b> {model_metrics['mse']:.4f} \n"
        f"  • <b>PSNR:</b> {model_metrics['psnr']:.4f} \n"
        f"  • <b>FID:</b> {model_metrics['fid']:.4f} \n"
        f"  • <b>LR:</b> {curr_lr:.6f}"
    )

    if results['ema'] is not None:
        ema_metrics = results['ema']
        message += (
            f"\n\n<b>EMA Epoch/Iteration {epoch}/{n_iter}</b>\n"
            f"  • <b>SSIM:</b> {ema_metrics['ssim']:.4f} \n"
            f"  • <b>MSE:</b> {ema_metrics['mse']:.4f} \n"
            f"  • <b>PSNR:</b> {ema_metrics['psnr']:.4f} \n"
            f"  • <b>FID:</b> {ema_metrics['fid']:.4f} \n"
            f"  • <b>LR:</b> {curr_lr:.6f}"
        )
    
    return message


def train_diffusion_model(config, train_dataloader, output_dir, device, continue_training=False):
    """Train a DDPM diffusion model."""
    image_size = config['dataset']['image_size']
    channels = config['dataset']['image_channels']
    batch_size = config['dataloader']['batch_size']
    grad_accumulation = config['dataloader']['grad_accumulation']
    iterations = config['training']['iterations']
    epochs = int(iterations / (len(train_dataloader) / grad_accumulation)) + 1
    loss_type = config['training']['loss_function']
    timesteps = config['model']['beta_schedule']['train']['n_timestep']
    freq_metrics = config['training']['freq_metrics']
    freq_checkpoint = config['training']['freq_checkpoint']
    use_ema = config['model']['use_ema']

    ddpm = initialize_ddpm(config, phase="train", device=device)
    os.makedirs(f"{output_dir}/models", exist_ok=True)
    os.makedirs(f"{output_dir}/images", exist_ok=True)

    if continue_training:
        model_path = os.path.join(output_dir, "models", "last_model.pt")
        start_epoch, n_iter, best_loss = load_model(model_path, device, ddpm)
        
        if use_ema:
            ema_model_path = os.path.join(output_dir, "models", "last_ema_model.pt")
            ema_epoch, ema_n_iter = load_ema_model(ema_model_path, device, ddpm)
        
    else:
        start_epoch, n_iter = 0, 0
        best_loss = np.inf

    metrics_dict = {
        'psnr': PeakSignalNoiseRatio(data_range=None, reduction="elementwise_mean"),
        'ssim': StructuralSimilarityIndexMeasure(data_range=None, reduction="elementwise_mean"),
        'mse': MeanSquaredError(),
        'fid': FrechetInceptionDistance(normalize=True)
    }

    start_time = time.time()

    try:
        epoch_pbar = tqdm(range(start_epoch, epochs), initial=start_epoch, total=epochs, desc="Epoch")
        for epoch in epoch_pbar:
            epoch_loss = 0.0
            epoch_pbar.set_postfix({"iterations": n_iter, "loss": f"{epoch_loss:.6f}"})
            
            batch_pbar = tqdm(train_dataloader, desc="Training Batches", leave=False)
            for batch_idx, data in enumerate(batch_pbar):
                curr_lr = ddpm.optimizer.param_groups[0]['lr']
                batch_pbar.set_postfix({"lr": f"{curr_lr:.6f}", "iter": n_iter})

                x = data['image'].to(device)
                x_names = data['name']
                batch_size = x.shape[0]

                if batch_idx == 0 and epoch == 0:
                    ddpm.save_noising_process_image(x, f'{output_dir}/images/noising_process.png')

                t = ddpm.sample_timesteps(batch_size)
                loss = ddpm.p_losses(x=x, t=t, loss_type=loss_type)
                loss = loss / grad_accumulation

                loss.backward()
                nn.utils.clip_grad_norm_(ddpm.model.parameters(), max_norm=1.0)
                epoch_loss += loss.item()

                if ((batch_idx + 1) % grad_accumulation == 0) or (batch_idx + 1 == len(train_dataloader)):
                    ddpm.optimizer.step()
                    ddpm.optimizer.zero_grad()

                    if use_ema:
                        ddpm.update_ema()

                    n_iter += 1
                    batch_pbar.set_postfix({"lr": f"{curr_lr:.6f}", "iter": n_iter, "loss": f"{loss.item():.6f}"})

                    save_model(ddpm.model, ddpm.optimizer, epoch, n_iter, loss, f"{output_dir}/models", name="last_model")
                    if use_ema:
                        save_ema_model(ddpm.ema_model, epoch, n_iter, f"{output_dir}/models", name="last_ema_model")


                if n_iter % freq_metrics == 0 and batch_idx % grad_accumulation == 0 and n_iter != 0:
                    metrics_results = compute_diffusion_metrics(
                        ddpm=ddpm,
                        x=x,
                        x_names=x_names,
                        batch_size=batch_size,
                        timesteps=timesteps,
                        channels=channels,
                        use_ema=use_ema,
                        epoch=epoch,
                        n_iter=n_iter,
                        batch_idx=batch_idx,
                        root_path=f"{output_dir}/images",
                        device=device,
                        metrics_dict=metrics_dict
                    )
                    message = generate_metrics_message(metrics_results, curr_lr, epoch, n_iter)
                    print(message)


                if n_iter % freq_checkpoint == 0 and n_iter != 0 and batch_idx % grad_accumulation == 0:
                    print(f"Saving checkpoint at epoch {epoch}, iter {n_iter}, loss: {loss.item():.4f}")
                    save_model(ddpm.model, ddpm.optimizer, epoch, n_iter, loss, f"{output_dir}/models", name=f"model_epoch{epoch}_step{n_iter}")
                    if use_ema:
                        save_ema_model(ddpm.ema_model, epoch, n_iter, f"{output_dir}/models", name=f"ema_model_epoch{epoch}_step{n_iter}")

                    # Exit if reached specified iterations
                    if n_iter % iterations == 0:
                        print(f"Reached {iterations} iterations. Exiting training.")
                        return ddpm

            epoch_loss /= len(train_dataloader)
            epoch_pbar.set_postfix({"iterations": n_iter, "loss": f"{epoch_loss:.4f}"})
                            
    except (KeyboardInterrupt, SystemExit, Exception) as e:
        if isinstance(e, Exception):
            print(f"Exception: {e}")
        print("\nTraining interrupted. Saving final state...")
        save_model(ddpm.model, ddpm.optimizer, epoch, n_iter, loss, f"{output_dir}/models", name="last_model")
        if use_ema:
            save_ema_model(ddpm.ema_model, epoch, n_iter, f"{output_dir}/models", name="last_ema_model")
        
    finally:
        print(f"Training completed in {time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))}")
        
    return ddpm

def compute_fid(real_images: torch.Tensor, generated_images: torch.Tensor, device: str = "cuda") -> float:
    """Compute FID score by processing images in batches."""
    fid_metric = FrechetInceptionDistance(feature=2048, normalize=True).to(device)

    for batch in torch.split(real_images, 16):
        fid_metric.update(torch.clamp(batch, 0.0, 1.0), real=True)

    for batch in torch.split(generated_images, 16):
        fid_metric.update(torch.clamp(batch, 0.0, 1.0), real=False)
    
    return fid_metric.compute().item()


def evaluate_diffusion_model(config, test_dataloader, output_dir, device, num_generated_images: int = 256):
    """Evaluate a trained diffusion model by generating images and computing metrics."""
    ddpm = initialize_ddpm(config, phase="test", device=device)
    model_path = f"{output_dir}/models/{config['model']['model_for_inference']}"
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model {model_path} does not exist")
    
    checkpoint = torch.load(model_path, map_location=device)
    ddpm.model.load_state_dict(checkpoint['model_state_dict'])
    model_iteration = checkpoint.get('n_iter', 0)
    model_epoch = checkpoint.get('epoch', 'undefined')
    
    print(f"Model loaded for epoch: {model_epoch} and iteration: {model_iteration}\n{model_path}")
    del checkpoint

    metrics_dict = {
        'psnr': PeakSignalNoiseRatio(data_range=None, reduction="elementwise_mean"),
        'ssim': StructuralSimilarityIndexMeasure(data_range=None, reduction="elementwise_mean"),
        'mse': MeanSquaredError(),
    }

    all_real_images = []
    all_generated_images = []
    generated_count = 0
    ssim_scores = []
    mse_scores = []
    psnr_scores = []

    ddpm.model.eval()
    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm(test_dataloader, desc="Processing test batches")):
            batch_size = data['image'].shape[0]
            x_names = data['name']
            x = data['image'].to(device)
            
            if not (0.0 <= x.min() and x.max() <= 1.0):
                raise ValueError(f"Image pixel range is not [0, 1]: [{x.min()}, {x.max()}]")

            x_cpu = x.detach().cpu()
            all_real_images.extend(x_cpu)

            if _should_generate_images(num_generated_images, generated_count):
                images_to_generate, x_for_generation = _prepare_generation_batch(
                    x, batch_size, num_generated_images, generated_count
                )

                print(f"Generating {images_to_generate} images for batch {batch_idx}...")
                x_hat = ddpm.sample(
                    model=ddpm.model, 
                    x_cond=x_for_generation, 
                    batch_size=images_to_generate, 
                    timesteps=ddpm.timesteps
                )
                x_hat = x_hat.clamp(0, 1).detach().cpu()

                all_generated_images.extend(x_hat)
                generated_count += images_to_generate

                real_batch_for_metrics = x_cpu[:images_to_generate]
                batch_ssim = metrics_dict['ssim'](x_hat, real_batch_for_metrics).item()
                batch_mse = metrics_dict['mse'](x_hat, real_batch_for_metrics).item()
                batch_psnr = metrics_dict['psnr'](x_hat, real_batch_for_metrics).item()
                
                ssim_scores.append(batch_ssim)
                mse_scores.append(batch_mse)
                psnr_scores.append(batch_psnr)

                _save_batch_visualization(
                    real_batch_for_metrics, x_hat, x_names[:images_to_generate],
                    output_dir, config, model_iteration, batch_idx
                )
            else:
                print(f"Skipping generation for batch {batch_idx} (target: {generated_count}/{num_generated_images})")

    all_real_tensor = torch.stack(all_real_images)
    all_generated_tensor = torch.stack(all_generated_images)

    avg_ssim = np.mean(ssim_scores) if ssim_scores else 0.0
    avg_mse = np.mean(mse_scores) if mse_scores else 0.0
    avg_psnr = np.mean(psnr_scores) if psnr_scores else 0.0

    if len(all_generated_tensor) > 1:
        channels = config['dataset']['image_channels']
        real_fid = all_real_tensor if channels == 3 else all_real_tensor.repeat(1, 3, 1, 1)
        fake_fid = all_generated_tensor if channels == 3 else all_generated_tensor.repeat(1, 3, 1, 1)

        real_fid = real_fid.to(device)
        fake_fid = fake_fid.to(device)
        
        fid_score = compute_fid(real_fid, fake_fid, device=str(device))
    else:
        fid_score = 0.0

    metrics_results = {
        'ssim': avg_ssim,
        'mse': avg_mse,
        'psnr': avg_psnr,
        'fid': fid_score
    }

    _log_evaluation_results(
        metrics_results, len(all_generated_tensor), len(all_generated_tensor), len(all_real_tensor),
        output_dir, config, model_iteration
    )
    
    return all_real_tensor, all_generated_tensor, model_iteration


def _should_generate_images(num_generated_images: Optional[int], generated_count: int) -> bool:
    """Check if we should continue generating images."""
    return num_generated_images is None or generated_count < num_generated_images


def _prepare_generation_batch(x: torch.Tensor, batch_size: int, 
                            num_generated_images: Optional[int], generated_count: int) -> Tuple[int, torch.Tensor]:
    """Prepare the batch for image generation."""
    if num_generated_images is None:
        return batch_size, x
    else:
        images_to_generate = min(batch_size, num_generated_images - generated_count)
        return images_to_generate, x[:images_to_generate]


def _save_batch_visualization(x_real: torch.Tensor, x_generated: torch.Tensor,
                            x_names: List[str], output_dir: str, config: Dict,
                            model_iteration: int, batch_idx: int) -> None:
    """Save visualization results for a batch."""
    diff = compute_diff(x_real, x_generated)

    test_imgs_path = (f"{output_dir}/images/test_steps"
                     f"{config['model']['beta_schedule']['test']['n_timestep']}_iter{model_iteration}")
    os.makedirs(test_imgs_path, exist_ok=True)

    image_titles = list(x_names)
    save_generated_images(
        [x_real, x_generated, diff],
        f"{test_imgs_path}/test_batch{batch_idx}.jpg",
        f"Batch {batch_idx}",
        image_titles
    )


def _log_evaluation_results(metrics: Dict[str, float], num_generated: int,
                          num_real_for_metrics: int, total_real: int,
                          output_dir: str, config: Dict, model_iteration: int) -> None:
    """Log evaluation results and save to txt file."""
    results_message = (
        f"{'='*60}\n"
        f"EVALUATION RESULTS\n"
        f"{'='*60}\n"
        f"Dataset Information:\n"
        f"  • Total real images in test set: {total_real}\n"
        f"  • Generated images for evaluation: {num_generated}\n"
        f"  • Real images used for metrics: {num_real_for_metrics}\n"
        f"\n"
        f"Metrics (computed across {num_generated} image pairs):\n"
        f"  • SSIM:  {metrics['ssim']:.4f} (averaged from batch scores)\n"
        f"  • MSE:   {metrics['mse']:.6f} (averaged from batch scores)\n"
        f"  • PSNR:  {metrics['psnr']:.4f} (averaged from batch scores)\n"
        f"  • FID:   {metrics['fid']:.4f} (computed across all images)\n"
        f"{'='*60}"
    )
    
    print(results_message)

    test_imgs_path = (f"{output_dir}/images/test_steps"
                     f"{config['model']['beta_schedule']['test']['n_timestep']}_iter{model_iteration}")
    os.makedirs(test_imgs_path, exist_ok=True)

    import datetime
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    file_results_message = (
        f"EVALUATION RESULTS\n"
        f"Generated on: {timestamp}\n"
        f"Model iteration: {model_iteration}\n"
        f"Test timesteps: {config['model']['beta_schedule']['test']['n_timestep']}\n"
        f"{'='*60}\n\n"
        f"Dataset Information:\n"
        f"  • Total real images in test set: {total_real}\n"
        f"  • Generated images for evaluation: {num_generated}\n"
        f"  • Real images used for metrics: {num_real_for_metrics}\n"
        f"\n"
        f"Metrics (computed across {num_generated} image pairs):\n"
        f"  • SSIM:  {metrics['ssim']:.4f} (averaged from batch scores)\n"
        f"  • MSE:   {metrics['mse']:.6f} (averaged from batch scores)\n"
        f"  • PSNR:  {metrics['psnr']:.4f} (averaged from batch scores)\n"
        f"  • FID:   {metrics['fid']:.4f} (computed across all images)\n"
        f"\n"
        f"Raw Metrics Values:\n"
        f"  • SSIM: {metrics['ssim']}\n"
        f"  • MSE: {metrics['mse']}\n"
        f"  • PSNR: {metrics['psnr']}\n"
        f"  • FID: {metrics['fid']}\n"
        f"{'='*60}\n"
    )

    results_file_path = f"{test_imgs_path}/evaluation_results.txt"
    with open(results_file_path, 'w') as f:
        f.write(file_results_message)
    
    print(f"Evaluation results saved to: {results_file_path}")