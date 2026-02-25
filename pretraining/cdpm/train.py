

import os
import math
from typing import Dict
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from .diffusion import GaussianDiffusionSampler, GaussianDiffusionTrainer
from .model import UNet
from .scheduler import GradualWarmupScheduler
from pretraining.eval_utils import compute_per_class_fid as _compute_per_class_fid
from datasets.combined_dataset import CombinedDataset
from datasets.nih_chest_dataset import NIHCDPMDataset

import sys
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from datasets.chest_dataset import ChestDataset
from datasets.hand_dataset import HandDataset
from datasets.cephalo_dataset import CephaloDataset
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.psnr import PeakSignalNoiseRatio
from torchmetrics import MeanSquaredError


def _build_checkpoint_dict(net_model, optimizer, warmUpScheduler, current_iteration, epoch,
                           gradient_norms, **extra):
    d = {
        'model_state_dict': net_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': warmUpScheduler.state_dict(),
        'learning_rate': warmUpScheduler.get_last_lr(),
        'iteration': current_iteration,
        'epoch': epoch,
        'avg_gradient_norm': sum(gradient_norms) / len(gradient_norms) if gradient_norms else 0.0,
    }
    d.update(extra)
    return d


@torch.no_grad()
def validate_at_checkpoint(model_config: Dict, net_model: torch.nn.Module, device: torch.device,
                          iteration: int, epoch: int, num_classes: int, dataset: CombinedDataset):
    """
    Fast validation: generate 4 samples per class using accelerated sampling (50 steps).
    Skips FID (unreliable with few samples) for speed.
    """
    print(f"\n{'='*60}")
    print(f"VALIDATION @ iter {iteration} (epoch {epoch})")
    print(f"{'='*60}")
    
    sampler = GaussianDiffusionSampler(
        net_model, model_config["beta_1"], model_config["beta_T"], model_config["T"], 
        w=model_config["w"]).to(device)
    
    val_dir = os.path.join(model_config["sampled_dir"], "validation", f"iter_{iteration}")
    os.makedirs(val_dir, exist_ok=True)
    
    num_samples = 4  # Fixed: generate only 4 images per class
    ddim_steps = model_config.get("ddim_steps", None) 
    all_metrics = {'ssim': [], 'psnr': [], 'mse': []}
    
    classes_to_validate = model_config.get('classes_to_validate', [-1])
    
    # Get a small batch of real images for metrics (one pass through data)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=2)
    real_by_class = {c: [] for c in range(num_classes)}
    
    # Determine which real classes we need to collect (for unconditional, we need all classes)
    need_all_classes = -1 in classes_to_validate
    classes_needing_real_data = set(range(num_classes)) if need_all_classes else set(c for c in classes_to_validate if c >= 0)
    
    for batch in dataloader:
        for c in range(num_classes):
            if c in classes_needing_real_data and len(real_by_class[c]) < num_samples:
                mask = batch['class_label'] == c
                if mask.any():
                    real_by_class[c].append(batch['image'][mask][:num_samples - len(real_by_class[c])])
        if all(len(v) >= 1 and sum(x.shape[0] for x in v) >= num_samples for c, v in real_by_class.items() if c in classes_needing_real_data):
            break
    
    # Process each selected class
    for class_label in classes_to_validate:
        # For unconditional generation (class -1), use random real images
        if class_label == -1:
            print(f"\nClass -1 (Unconditional): Generating {num_samples} samples...")
            # Get random real images for metrics
            all_real = []
            for c in range(num_classes):
                if real_by_class[c]:
                    all_real.extend(real_by_class[c])
            if not all_real:
                print("  No data available")
                continue
            real_images = torch.cat(all_real, dim=0)[:num_samples].to(device)
            actual_samples = min(num_samples, real_images.shape[0])
            real_images = real_images[:actual_samples]
        else:
            # Get real images for this class
            if class_label not in real_by_class or not real_by_class[class_label]:
                print(f"Class {class_label}: No data")
                continue
            
            real_images = torch.cat(real_by_class[class_label], dim=0)[:num_samples].to(device)
            actual_samples = real_images.shape[0]
        
        # Generate samples with fast DDIM-style sampling
        noisy_images = torch.randn(
            size=[actual_samples, model_config["img_channels"], model_config["img_size"], model_config["img_size"]], 
            device=device)
        class_labels = torch.full((actual_samples,), class_label + 1, dtype=torch.long, device=device)
        generated = sampler(noisy_images, class_labels, start_t=model_config["T"], ddim_steps=ddim_steps)
        
        # Save visualization (4 images in a row)
        save_image(generated, os.path.join(val_dir, f"class_{class_label}.png"), nrow=4)
        
        # Compute metrics
        real_clamped = torch.clamp(real_images, 0.0, 1.0)
        gen_clamped = torch.clamp(generated, 0.0, 1.0)
        
        ssim_val = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)(gen_clamped, real_clamped).item()
        psnr_val = PeakSignalNoiseRatio(data_range=1.0).to(device)(gen_clamped, real_clamped).item()
        mse_val = MeanSquaredError().to(device)(gen_clamped, real_clamped).item()
        
        all_metrics['ssim'].append(ssim_val)
        all_metrics['psnr'].append(psnr_val)
        all_metrics['mse'].append(mse_val)
        
        print(f"Class {class_label}: SSIM={ssim_val:.3f} | PSNR={psnr_val:.2f} | MSE={mse_val:.4f}")
        
        del real_images, generated, noisy_images
    
    # Print summary
    if all_metrics['ssim']:
        print(f"{'─'*60}")
        print(f"Mean: SSIM={sum(all_metrics['ssim'])/len(all_metrics['ssim']):.3f} | "
              f"PSNR={sum(all_metrics['psnr'])/len(all_metrics['psnr']):.2f} | "
              f"MSE={sum(all_metrics['mse'])/len(all_metrics['mse']):.4f}")
    print(f"{'='*60}\n")
    
    # Save metrics
    with open(os.path.join(val_dir, "metrics.txt"), 'w') as f:
        f.write(f"Validation at Iteration {iteration} (Epoch {epoch})\n")
        if all_metrics['ssim']:
            f.write(f"SSIM - Mean: {sum(all_metrics['ssim'])/len(all_metrics['ssim']):.4f} | Per-class: {all_metrics['ssim']}\n")
            f.write(f"PSNR - Mean: {sum(all_metrics['psnr'])/len(all_metrics['psnr']):.4f} | Per-class: {all_metrics['psnr']}\n")
            f.write(f"MSE  - Mean: {sum(all_metrics['mse'])/len(all_metrics['mse']):.6f} | Per-class: {all_metrics['mse']}\n")
    
    del sampler, real_by_class
    torch.cuda.empty_cache()


def compute_per_class_fid(model_config: Dict, net_model: torch.nn.Module, device: torch.device,
                          num_classes: int, dataset):
    """Compute per-class FID using the shared utility in pretraining.eval_utils."""
    ddim_steps = model_config.get("ddim_steps", 50)
    sampler = GaussianDiffusionSampler(
        net_model, model_config["beta_1"], model_config["beta_T"], model_config["T"],
        w=model_config["w"]).to(device)

    def generate_fn(class_idx, current_batch_size, dev):
        noisy = torch.randn(
            current_batch_size, model_config["img_channels"],
            model_config["img_size"], model_config["img_size"], device=dev)
        labels = torch.full((current_batch_size,), class_idx + 1, dtype=torch.long, device=dev)
        return sampler(noisy, labels, start_t=model_config["T"], ddim_steps=ddim_steps)

    run_name = get_run_name(model_config)
    fid_path = os.path.join(model_config["sampled_dir"], run_name, "final_fid.txt")

    _compute_per_class_fid(
        num_classes=num_classes,
        dataset=dataset,
        generate_fn=generate_fn,
        fid_path=fid_path,
        device=device,
        ddim_steps=ddim_steps,
    )


class SingleClassDataset:
    """Wraps any dataset to treat all samples as a single class (class_label=0)."""
    def __init__(self, dataset):
        self._dataset = dataset
        self.num_classes = 1

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx):
        sample = self._dataset[idx]
        sample['class_label'] = 0
        return sample


def get_run_name(model_config: Dict):
    """Get a descriptive subdirectory name based on dataset_type and conditioned."""
    dataset_type = model_config.get("dataset_type", "combined")
    conditioned = model_config.get("conditioned", True)
    if conditioned:
        return dataset_type
    else:
        return f"{dataset_type}_unconditioned"


def get_dataset(model_config: Dict):
    """
    Prepares the dataset based on configuration.
    Supports:
    - 'combined': Chest, Hand, and Cephalo X-rays (multi-class if conditioned, single-class otherwise)
    - 'chest', 'hand', 'cephalo': Individual datasets (single-class)
    - 'nih': NIH Chest X-rays only (single-class)

    Args:
        model_config (Dict): Configuration dictionary.

    Returns:
        tuple: (dataset, num_classes)
    """
    dataset_type = model_config.get("dataset_type", "combined")
    conditioned = model_config.get("conditioned", True)
    
    if dataset_type == "nih":
        # NIH Chest X-ray dataset (single class)
        dataset = NIHCDPMDataset(
            root_dir=model_config["data_dir"],
            phase="train_val",
            image_size=(model_config["img_size"], model_config["img_size"]),
            max_samples=model_config.get("max_samples", None)
        )
    elif dataset_type in ("chest", "hand", "cephalo"):
        # Individual dataset (single class)
        dataset_cls = {"chest": ChestDataset, "hand": HandDataset, "cephalo": CephaloDataset}[dataset_type]
        raw_dataset = dataset_cls(
            root_dir=model_config["data_dir"],
            phase="train_val",
            image_size=(model_config["img_size"], model_config["img_size"]),
            channels=1,
            task_type='pretraining',
        )
        dataset = SingleClassDataset(raw_dataset)
    else:
        # Combined dataset (chest, hand, cephalo)
        dataset = CombinedDataset(
            root_dir=model_config["data_dir"],
            phase="train_val",
            image_size=(model_config["img_size"], model_config["img_size"]),
        )
        if not conditioned:
            dataset = SingleClassDataset(dataset)
    
    num_classes = dataset.num_classes
    return dataset, num_classes


def _setup_model_and_optimizer(model_config: Dict, device: torch.device,
                               num_classes: int, epochs: int):
    """Create UNet, optimizer, LR schedulers, and diffusion trainer."""
    net_model = UNet(
        T=model_config["T"], num_labels=num_classes, ch=model_config["channel"],
        ch_mult=model_config["channel_mult"],
        attn_heads=model_config["attn_heads"],
        dim_head=model_config["dim_head"],
        use_linear_attn=model_config["use_linear_attn"],
        num_res_blocks=model_config["num_res_blocks"],
        dropout=model_config["dropout"],
        img_channels=model_config["img_channels"],
    ).to(device)

    optimizer = torch.optim.AdamW(net_model.parameters(), lr=model_config["lr"], weight_decay=1e-4)

    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=epochs, eta_min=0, last_epoch=-1
    )
    warmup_epochs = max(1, epochs // 10)
    warmup_scheduler = GradualWarmupScheduler(
        optimizer=optimizer,
        multiplier=model_config["multiplier"],
        warm_epoch=warmup_epochs,
        after_scheduler=cosine_scheduler,
    )

    trainer = GaussianDiffusionTrainer(
        net_model, model_config["beta_1"], model_config["beta_T"], model_config["T"]
    ).to(device)

    return net_model, optimizer, warmup_scheduler, trainer


def _resume_from_checkpoint(model_config: Dict, device: torch.device,
                             net_model, optimizer, warmup_scheduler):
    """Load a saved checkpoint when resume=True. Returns (current_iteration, start_epoch)."""
    if not model_config.get("resume", False):
        return 0, 0

    if model_config.get("resume_checkpoint"):
        resume_path = model_config["resume_checkpoint"]
    else:
        run_name = get_run_name(model_config)
        resume_path = os.path.join(model_config["save_dir"], run_name, "last_model.pt")

    if not os.path.exists(resume_path):
        print(f"⚠️  Resume requested but checkpoint not found: {resume_path}")
        print("   Starting training from scratch...")
        return 0, 0

    print(f"\n🔄 Resuming from checkpoint: {resume_path}")
    checkpoint = torch.load(resume_path, map_location=device)

    net_model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    warmup_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    current_iteration = checkpoint.get('iteration', 0)
    start_epoch = checkpoint.get('epoch', 0)

    print(f"✅ Resumed: iteration={current_iteration}, epoch={start_epoch}")
    print(f"   Learning rate: {optimizer.param_groups[0]['lr']:.2e}")
    return current_iteration, start_epoch


def train(model_config: Dict):
    device = torch.device(model_config["device"])
    
    # Enable cuDNN benchmark for speed optimization
    torch.backends.cudnn.benchmark = True

    # Data transform - keep as 1-channel grayscale for medical imaging
    # Transforms are handled internally by dataset classes via get_transforms()

    # Mixed precision flag
    use_amp = model_config["mixed_precision"]
    if use_amp:
        print("Using mixed precision training (torch.cuda.amp)")
    
    # Enable anomaly detection for debugging (disable in production for speed)
    # torch.autograd.set_detect_anomaly(True)

    # Prepare dataset
    dataset, num_classes = get_dataset(model_config)

    # Optimized DataLoader for high-resolution training
    dataloader = DataLoader(
        dataset, batch_size=model_config["batch_size"], shuffle=True,
        num_workers=4, drop_last=True, pin_memory=True, persistent_workers=False
    )

    # For mixed precision (PyTorch 2.x compatible)
    from torch.amp import GradScaler, autocast
    scaler = GradScaler(device) if use_amp else None

    # Training setup
    iterations = model_config["iterations"]
    iters_to_accumulate = model_config["iters_to_accumulate"]
    
    # Compute epochs needed
    batches_per_epoch = len(dataloader)
    iterations_per_epoch = math.ceil(batches_per_epoch / iters_to_accumulate)
    epochs = math.ceil(iterations / iterations_per_epoch)
    total_batches = iterations * iters_to_accumulate

    print(f"\n{'='*80}")
    print(f"TRAINING CONFIGURATION")
    print(f"{'='*80}")
    print(f"Dataset: {len(dataset)} samples")
    print(f"Batch size: {model_config['batch_size']} | Accumulation steps: {iters_to_accumulate}")
    print(f"Batches per epoch: {batches_per_epoch} | Iterations per epoch: {iterations_per_epoch}")
    print(f"Total epochs: {epochs} | Total iterations: {iterations}")
    print(f"Total batches to process: {total_batches}")
    print(f"{'='*80}\n")


    net_model, optimizer, warmUpScheduler, trainer = _setup_model_and_optimizer(
        model_config, device, num_classes, epochs)

    current_iteration, start_epoch = _resume_from_checkpoint(
        model_config, device, net_model, optimizer, warmUpScheduler)

    # Track resumed iteration to avoid duplicate validation
    resumed_iteration = current_iteration if model_config.get("resume", False) else -1
    
    # Gradient monitoring
    gradient_norms = []

    # start training
    try:
        for epoch in range(start_epoch, epochs):
            with tqdm(dataloader, dynamic_ncols=True) as tqdmDataLoader:
                for i, data in enumerate(tqdmDataLoader):
                    images = data['image']
                    labels = data['class_label']
                    
                    if i % iters_to_accumulate == 0:
                        optimizer.zero_grad()
                    
                    x_0 = images.to(device)
                    labels = labels.to(device) + 1
                    
                    # Unconditional training with 10% probability
                    if np.random.rand() < 0.1:
                        labels = torch.zeros_like(labels).to(device)

                    if use_amp:
                        with autocast(device.type):
                            loss = trainer(x_0, labels, use_checkpoint=model_config["gradient_checkpointing"]).mean()
                            loss = loss / iters_to_accumulate
                    else:
                        loss = trainer(x_0, labels, use_checkpoint=model_config["gradient_checkpointing"]).mean()
                        loss = loss / iters_to_accumulate
                    
                    # Check for NaN loss before backward pass
                    if torch.isnan(loss) or torch.isinf(loss):
                        print(f"\n⚠️  NaN/Inf loss detected at epoch {epoch}, iter {current_iteration}. Skipping batch.")
                        optimizer.zero_grad()
                        if use_amp:
                            scaler.update()  # Still need to update scaler state
                        continue
                    
                    # Backward pass
                    if use_amp:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()

                    # Perform optimizer step every iters_to_accumulate
                    if (i + 1) % iters_to_accumulate == 0 or (i + 1) == len(tqdmDataLoader):
                        if use_amp:
                            scaler.unscale_(optimizer)
                            grad_norm = torch.nn.utils.clip_grad_norm_(net_model.parameters(), model_config["grad_clip"])
                            
                            # Skip step if gradients are NaN/Inf (scaler handles this automatically but we log it)
                            if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                                print(f"\n⚠️  NaN/Inf gradients at epoch {epoch}, iter {current_iteration}. Skipping optimizer step.")
                                optimizer.zero_grad()
                                scaler.update()
                                # Don't increment iteration on NaN
                                continue
                            
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            grad_norm = torch.nn.utils.clip_grad_norm_(net_model.parameters(), model_config["grad_clip"])
                            
                            # Skip step if gradients are NaN/Inf
                            if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                                print(f"\n⚠️  NaN/Inf gradients at epoch {epoch}, iter {current_iteration}. Skipping optimizer step.")
                                optimizer.zero_grad()
                                continue
                            
                            optimizer.step()
                        
                        # Store gradient norm for monitoring
                        gradient_norms.append(grad_norm.item())
                        if len(gradient_norms) > 100:
                            gradient_norms.pop(0)
                        
                        current_iteration += 1

                        # Save last_model checkpoint periodically
                        if current_iteration % 10 == 0:
                            run_name = get_run_name(model_config)
                            last_model_path = os.path.join(model_config["save_dir"], run_name, "last_model.pt")
                            os.makedirs(os.path.dirname(last_model_path), exist_ok=True)
                            checkpoint_dict = _build_checkpoint_dict(
                                net_model, optimizer, warmUpScheduler,
                                current_iteration, epoch, gradient_norms)
                            torch.save(checkpoint_dict, last_model_path)
                            del checkpoint_dict
                        
                        # Save checkpoint and run validation at freq_save intervals
                        if model_config["freq_save"] > 0 and current_iteration % model_config["freq_save"] == 0 and current_iteration > 0 and current_iteration != resumed_iteration:
                            run_name = get_run_name(model_config)
                            model_path = os.path.join(model_config["save_dir"], run_name, 
                                                     f"ckpt_{current_iteration}_iterations.pt")
                            os.makedirs(os.path.dirname(model_path), exist_ok=True)
                            checkpoint_dict = _build_checkpoint_dict(
                                net_model, optimizer, warmUpScheduler,
                                current_iteration, epoch, gradient_norms)
                            torch.save(checkpoint_dict, model_path)
                            del checkpoint_dict
                            print(f"\n💾 Checkpoint saved: iteration {current_iteration}")
                            
                            # Log gradient statistics
                            if gradient_norms:
                                print(f"📊 Gradient stats - Min: {min(gradient_norms):.2f}, "
                                      f"Max: {max(gradient_norms):.2f}, "
                                      f"Avg: {sum(gradient_norms)/len(gradient_norms):.2f}")
                            
                            # Run validation at checkpoint
                            if model_config.get("validate_at_save", True):
                                net_model.eval()
                                validate_at_checkpoint(model_config, net_model, device, current_iteration, epoch, num_classes, dataset)
                                net_model.train()
                    
                    postfix_dict = {
                        "epoch": epoch,
                        "iter": current_iteration,
                        "loss": f"{loss.item():.4f}",
                        "LR": f"{optimizer.state_dict()['param_groups'][0]['lr']:.2e}"
                    }
                    
                    # Add gradient norm to display if available
                    if gradient_norms:
                        avg_grad_norm = sum(gradient_norms[-10:]) / min(len(gradient_norms), 10)
                        postfix_dict["grad"] = f"{avg_grad_norm:.2f}"

                    tqdmDataLoader.set_postfix(ordered_dict=postfix_dict)

            warmUpScheduler.step()
        
    except KeyboardInterrupt:
        print("\n\n" + "="*80)
        print("⚠️  TRAINING INTERRUPTED BY USER")
        print("="*80)
        print(f"Saving final checkpoint at iteration {current_iteration}...")
        
        # Safe checkpoint saving
        run_name = get_run_name(model_config)
        emergency_path = os.path.join(model_config["save_dir"], run_name, 
                                      f"ckpt_emergency_iter{current_iteration}.pt")
        os.makedirs(os.path.dirname(emergency_path), exist_ok=True)
        torch.save(_build_checkpoint_dict(
            net_model, optimizer, warmUpScheduler,
            current_iteration, epoch, gradient_norms), emergency_path)
        print(f"✅ Emergency checkpoint saved to: {emergency_path}")
        print("="*80 + "\n")
        return
    
    except Exception as e:
        print("\n\n" + "="*80)
        print(f"❌ TRAINING FAILED WITH ERROR: {e}")
        print("="*80)
        print(f"Saving emergency checkpoint at iteration {current_iteration}...")
        
        # Safe checkpoint saving on error
        run_name = get_run_name(model_config)
        emergency_path = os.path.join(model_config["save_dir"], run_name, 
                                      f"ckpt_error_iter{current_iteration}.pt")
        os.makedirs(os.path.dirname(emergency_path), exist_ok=True)
        torch.save(_build_checkpoint_dict(
            net_model, optimizer, warmUpScheduler,
            current_iteration, epoch, gradient_norms, error=str(e)), emergency_path)
        print(f"✅ Error checkpoint saved to: {emergency_path}")
        print(f"Error details: {e}")
        print("="*80 + "\n")
        raise
    
    print(f"\n{'='*80}")
    print(f"TRAINING COMPLETE")
    print(f"{'='*80}")
    print(f"Total iterations: {current_iteration}")
    print(f"Total epochs: {epoch + 1}")
    print(f"{'='*80}\n")

    # Compute per-class FID at end of training
    torch.cuda.empty_cache()
    net_model.eval()
    compute_per_class_fid(model_config, net_model, device, num_classes, dataset)


def eval(model_config: Dict):
    """
    Generate samples for all classes without accumulating in memory.
    Saves images and optionally computes metrics batch-by-batch.
    """
    device = torch.device(model_config["device"])
    run_name = get_run_name(model_config)
    output_dir = os.path.join(model_config["sampled_dir"], run_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Data transform
    # Transforms are handled internally by dataset classes via get_transforms()

    # Prepare dataset
    dataset, num_classes = get_dataset(model_config)

    # Load model
    model = UNet(T=model_config["T"], num_labels=num_classes, ch=model_config["channel"],
                 ch_mult=model_config["channel_mult"], 
                 attn_heads=model_config["attn_heads"],
                 dim_head=model_config["dim_head"],
                 use_linear_attn=model_config["use_linear_attn"],
                 num_res_blocks=model_config["num_res_blocks"], 
                 dropout=model_config["dropout"], 
                 img_channels=model_config["img_channels"]).to(device)
    
    model_path = model_config["load_weights"]
    if model_path is None:
        raise ValueError("❌ Error: --load_weights must be specified when running in eval mode. "
                        "Please provide the path to a trained model checkpoint.")
    
    iterations = int(os.path.basename(model_path).split("_")[1])

    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        model.load_state_dict(ckpt)
    print(f"✅ Model loaded from {model_path}")
    del ckpt

    model.eval()
    sampler = GaussianDiffusionSampler(
        model, model_config["beta_1"], model_config["beta_T"], model_config["T"], w=model_config["w"]).to(device)
    
    batch_size = model_config["batch_size"]
    samples_per_class = model_config["images_to_sample"]
    num_batches = math.ceil(samples_per_class / batch_size)
    
    print(f"\n{'='*80}")
    print(f"GENERATING {samples_per_class} SAMPLES PER CLASS ({num_classes} classes)")
    print(f"Total: {samples_per_class * num_classes} images in {num_batches * num_classes} batches")
    print(f"{'='*80}\n")

    # Generate images for each class independently (memory-efficient)
    for class_label in range(num_classes):
        print(f"\nClass {class_label}: Generating {samples_per_class} images...")
        
        for batch_idx in range(num_batches):
            current_batch_size = min(batch_size, samples_per_class - batch_idx * batch_size)
            
            with torch.no_grad():
                noisy_images = torch.randn(
                    size=[current_batch_size, model_config["img_channels"], 
                          model_config["img_size"], model_config["img_size"]], 
                    device=device)
                class_labels = torch.full((current_batch_size,), class_label + 1, 
                                        dtype=torch.long, device=device)
                ddim_steps = model_config.get("ddim_steps", None)
                sampledImgs = sampler(noisy_images, class_labels, start_t=model_config["T"], ddim_steps=ddim_steps)

                # Save PNG visualization
                png_path = os.path.join(output_dir, 
                                       f"class_{class_label}_batch_{batch_idx}_iter{iterations}.png")
                save_image(sampledImgs, png_path, nrow=model_config["nrow"])
                
                # Save as numpy array
                imgs_numpy = sampledImgs.cpu().detach().permute(0, 2, 3, 1).numpy()
                npy_path = os.path.join(output_dir, 
                                       f"class_{class_label}_batch_{batch_idx}_iter{iterations}.npy")
                np.save(npy_path, imgs_numpy)
                
                print(f"  Batch {batch_idx + 1}/{num_batches}: Saved {current_batch_size} images")
                
                # Clean up immediately to avoid accumulation
                del sampledImgs, imgs_numpy, noisy_images, class_labels
        
        torch.cuda.empty_cache()
    
    print(f"\n{'='*80}")
    print(f"✅ GENERATION COMPLETE")
    print(f"Output directory: {output_dir}")
    print(f"{'='*80}\n")
    
    # Optionally run validation with metrics
    if model_config.get("validate_at_eval", False):
        print("\nRunning validation with metrics...")
        validate_at_checkpoint(model_config, model, device, iterations, 0, num_classes, dataset)
    
    del sampler, model
    torch.cuda.empty_cache()
