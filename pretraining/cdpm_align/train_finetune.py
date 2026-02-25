"""
CDPM Finetuning Training Loop with DINO-style Alignment.

Takes a pretrained CDPM model and finetunes it with the combined
diffusion + alignment loss to improve learned representations.
"""

import os
import math
from typing import Tuple, Dict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from tqdm import tqdm
from torchvision.utils import save_image


from .model_wrapper import create_model_from_pretrained
from .projector_heads import MultiLayerProjector
from .diffusion_utils import GaussianDiffusion
from .losses import CDPMAlignLoss
from datasets.combined_dataset import CombinedDataset
from pretraining.eval_utils import compute_per_class_fid as _compute_per_class_fid


def train_step(
    x_0: torch.Tensor,
    labels: torch.Tensor,
    model: nn.Module,
    loss_fn: CDPMAlignLoss,
    diffusion: GaussianDiffusion,
    config,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Single training step with 4 forward passes per batch.

    For finetuning, we use the single-class integer labels from the combined dataset
    but treat them as conditional labels for the diffusion model.
    """
    device = x_0.device
    batch_size = x_0.shape[0]

    # Convert class labels to conditional format:
    # labels are 0, 1, 2 (chest, hand, cephalo)
    # For conditional: add 1 to get 1, 2, 3
    # For unconditional: use 0
    labels_cond = labels + 1  # [B] -> 1, 2, 3
    labels_uncond = torch.zeros_like(labels)  # [B] -> 0

    # Sample two different timesteps
    t1 = diffusion.sample_timesteps(batch_size, device, config['bias_mid_timesteps'], config['mid_timestep_weight'])
    t2 = diffusion.sample_timesteps(batch_size, device, config['bias_mid_timesteps'], config['mid_timestep_weight'])

    # Forward diffusion
    x_t1, eps1 = diffusion.q_sample(x_0, t1)
    x_t2, eps2 = diffusion.q_sample(x_0, t2)

    # 4 forward passes
    use_ckpt = config['gradient_checkpointing']
    eps_u1, feats_u1 = model(x_t1, t1, labels_uncond, return_features=True, use_checkpoint=use_ckpt)
    eps_c1, feats_c1 = model(x_t1, t1, labels_cond, return_features=True, use_checkpoint=use_ckpt)
    eps_u2, feats_u2 = model(x_t2, t2, labels_uncond, return_features=True, use_checkpoint=use_ckpt)
    eps_c2, feats_c2 = model(x_t2, t2, labels_cond, return_features=True, use_checkpoint=use_ckpt)

    # Compute loss
    loss, loss_dict = loss_fn(
        eps_c1, eps_u1, eps1,
        eps_c2, eps_u2, eps2,
        feats_u1, feats_c1, feats_u2, feats_c2
    )

    return loss, loss_dict


def save_checkpoint(path: str, model, projector, optimizer, scheduler, iteration: int, config):
    """Save training checkpoint."""
    torch.save({
        'model_state_dict': model.base.state_dict(),
        'projector_state_dict': projector.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'iteration': iteration,
        'config': config,
    }, path)


def load_checkpoint(path: str, model, projector, optimizer, scheduler, device) -> int:
    """Load checkpoint and return iteration number."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.base.load_state_dict(ckpt['model_state_dict'])
    projector.load_state_dict(ckpt['projector_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    return ckpt['iteration']


def generate_samples(model, diffusion, config, iteration: int, num_samples: int = 12):
    """Generate samples per class for evaluation (4 per class)."""
    model.eval()
    device = torch.device(config['device'])

    samples_dir = os.path.join(config['save_dir'], 'samples')
    os.makedirs(samples_dir, exist_ok=True)

    with torch.no_grad():
        all_samples = []

        for class_idx in range(config['num_classes']):
            # Labels: 0=unconditional, 1=chest, 2=hand, 3=cephalo
            labels = torch.full((4,), class_idx + 1, dtype=torch.long, device=device)
            labels_uncond = torch.zeros_like(labels)

            shape = (4, config['img_channels'], config['img_size'], config['img_size'])

            samples = diffusion.sample(
                model, shape, labels, labels_uncond,
                w=config['w'],
                ddim_steps=config['ddim_steps'],
                device=device,
                verbose=False
            )

            samples = torch.clamp(samples, 0, 1)
            all_samples.append(samples)

        # Concatenate all samples
        all_samples = torch.cat(all_samples, dim=0)
        save_path = os.path.join(samples_dir, f'samples_iter_{iteration}.png')
        save_image(all_samples, save_path, nrow=4)
        print(f"Saved samples: {save_path}")

    model.train()


def compute_per_class_fid(model, diffusion, config, dataset):
    """Compute per-class FID using the shared utility in pretraining.eval_utils."""
    device = torch.device(config['device'])
    model.eval()
    ddim_steps = config['ddim_steps']

    def generate_fn(class_idx, current_batch_size, dev):
        labels = torch.full((current_batch_size,), class_idx + 1, dtype=torch.long, device=dev)
        labels_uncond = torch.zeros_like(labels)
        shape = (current_batch_size, config['img_channels'], config['img_size'], config['img_size'])
        return diffusion.sample(
            model, shape, labels, labels_uncond,
            w=config['w'], ddim_steps=ddim_steps, device=dev, verbose=False,
        )

    fid_path = os.path.join(config['save_dir'], "final_fid.txt")

    _compute_per_class_fid(
        num_classes=config['num_classes'],
        dataset=dataset,
        generate_fn=generate_fn,
        fid_path=fid_path,
        device=device,
        ddim_steps=ddim_steps,
    )


def train(config):
    """Main finetuning function."""
    device = torch.device(config['device'])
    torch.backends.cudnn.benchmark = True

    save_dir = config['save_dir']
    os.makedirs(save_dir, exist_ok=True)

    print(f"Device: {device}")
    print(f"Save dir: {save_dir}")
    print(f"Pretrained checkpoint: {config['pretrained_checkpoint']}")

    # Data - transforms handled internally by dataset classes via get_transforms()
    dataset = CombinedDataset(
        root_dir=config['data_dir'],
        phase='train_val',
        image_size=(config['img_size'], config['img_size']),
    )

    dataloader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4,
        drop_last=True,
        pin_memory=True,
        persistent_workers=True,
    )

    # Model
    model = create_model_from_pretrained(
        checkpoint_path=config['pretrained_checkpoint'],
        T=config['T'],
        num_classes=config['num_classes'],
        ch=config['channel'],
        ch_mult=config['channel_mult'],
        num_res_blocks=config['num_res_blocks'],
        dropout=config['dropout'],
        img_channels=config['img_channels'],
        attn_heads=config['attn_heads'],
        dim_head=config['dim_head'],
        use_linear_attn=config['use_linear_attn'],
        device=device
    )

    # Projector (randomly initialized)
    projector = MultiLayerProjector(
        feature_dims=model.get_feature_dims(),
        hidden_dim=config['proj_hidden_dim'],
        out_dim=config['proj_out_dim']
    ).to(device)

    # Diffusion
    diffusion = GaussianDiffusion(
        beta_1=config['beta_1'],
        beta_T=config['beta_T'],
        T=config['T']
    ).to(device)

    # Loss
    loss_fn = CDPMAlignLoss(
        projector=projector,
        lambda_diff=config['lambda_diff'],
        lambda_align=config['lambda_align']
    )

    # Optimizer - finetune with lower learning rate
    optimizer = optim.AdamW(
        list(model.parameters()) + list(projector.parameters()),
        lr=config['lr'],
        weight_decay=config['weight_decay']
    )

    # Scheduler
    batches_per_epoch = len(dataloader)
    epochs = math.ceil(config['iterations'] / batches_per_epoch)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Mixed precision
    scaler = GradScaler(device)

    # Resume
    current_iter = 0
    checkpoint_path = os.path.join(save_dir, 'last_model.pt')

    if config['resume'] and os.path.exists(checkpoint_path):
        print(f"Resuming from {checkpoint_path}")
        current_iter = load_checkpoint(checkpoint_path, model, projector, optimizer, scheduler, device)
        print(f"Resumed at iteration {current_iter}")

    print(f"\nDataset: {len(dataset)} samples")
    print(f"Epochs: {epochs}, Iterations: {config['iterations']}")
    print(f"Batch size: {config['batch_size']}")
    print()

    # Training loop
    model.train()
    projector.train()

    try:
        for epoch in range(epochs):
            pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

            for batch in pbar:
                if current_iter >= config['iterations']:
                    break

                x_0 = batch['image'].to(device, non_blocking=True)
                labels = batch['class_label'].to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                with autocast(device.type):
                    loss, loss_dict = train_step(x_0, labels, model, loss_fn, diffusion, config)

                if torch.isnan(loss):
                    print(f"NaN loss at iter {current_iter}, skipping")
                    continue

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    list(model.parameters()) + list(projector.parameters()),
                    config['grad_clip']
                )
                scaler.step(optimizer)
                scaler.update()

                current_iter += 1

                # Progress bar
                pbar.set_postfix({
                    'iter': current_iter,
                    'loss': f"{loss.item():.7f}",
                    'diff': f"{loss_dict['loss_diff'].item():.7f}",
                    'align': f"{loss_dict['loss_align'].item():.7f}",
                })

                # Save checkpoint
                if current_iter % config['freq_save'] == 0:
                    ckpt_path = os.path.join(save_dir, f'ckpt_{current_iter}.pt')
                    save_checkpoint(ckpt_path, model, projector, optimizer, scheduler, current_iter, config)
                    save_checkpoint(checkpoint_path, model, projector, optimizer, scheduler, current_iter, config)
                    print(f"\nSaved checkpoint: {ckpt_path}")

                    generate_samples(model, diffusion, config, current_iter)
                    model.train()

            scheduler.step()

            if current_iter >= config['iterations']:
                break

    except KeyboardInterrupt:
        print("\nInterrupted - saving emergency checkpoint")
        save_checkpoint(os.path.join(save_dir, 'emergency.pt'), model, projector, optimizer, scheduler, current_iter, config)

    # Final save
    final_path = os.path.join(save_dir, f'final_{current_iter}.pt')
    save_checkpoint(final_path, model, projector, optimizer, scheduler, current_iter, config)
    save_checkpoint(checkpoint_path, model, projector, optimizer, scheduler, current_iter, config)
    print(f"\nTraining complete. Final model: {final_path}")

    # Final samples
    generate_samples(model, diffusion, config, current_iter, num_samples=12)

    # Compute per-class FID at end of training
    compute_per_class_fid(model, diffusion, config, dataset)


def eval_generate(config):
    """Generate samples using finetuned model."""
    device = torch.device(config['device'])

    model = create_model_from_pretrained(
        checkpoint_path=config['pretrained_checkpoint'],
        T=config['T'],
        num_classes=config['num_classes'],
        ch=config['channel'],
        ch_mult=config['channel_mult'],
        num_res_blocks=config['num_res_blocks'],
        dropout=config['dropout'],
        img_channels=config['img_channels'],
        attn_heads=config['attn_heads'],
        dim_head=config['dim_head'],
        use_linear_attn=config['use_linear_attn'],
        device=device
    )

    # Load finetuned weights
    checkpoint_path = os.path.join(config['save_dir'], 'last_model.pt')
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.base.load_state_dict(ckpt['model_state_dict'])
    print(f"Loaded finetuned model from {checkpoint_path}")

    diffusion = GaussianDiffusion(
        beta_1=config['beta_1'],
        beta_T=config['beta_T'],
        T=config['T']
    ).to(device)

    model.eval()

    output_dir = os.path.join(config['save_dir'], 'eval_samples')
    os.makedirs(output_dir, exist_ok=True)

    num_batches = math.ceil(config['images_to_sample'] / config['batch_size'])

    for class_idx in range(config['num_classes']):
        print(f"\nGenerating samples for class {class_idx}...")

        for batch_idx in range(num_batches):
            current_batch = min(config['batch_size'], config['images_to_sample'] - batch_idx * config['batch_size'])

            with torch.no_grad():
                labels = torch.full((current_batch,), class_idx + 1, dtype=torch.long, device=device)
                labels_uncond = torch.zeros_like(labels)

                shape = (current_batch, config['img_channels'], config['img_size'], config['img_size'])

                samples = diffusion.sample(
                    model, shape, labels, labels_uncond,
                    w=config['w'],
                    ddim_steps=config['ddim_steps'],
                    device=device,
                    verbose=True
                )

                samples = torch.clamp(samples, 0, 1)
                save_path = os.path.join(output_dir, f'class_{class_idx}_batch_{batch_idx}.png')
                save_image(samples, save_path, nrow=4)
                print(f"Saved: {save_path}")

    print(f"\nGeneration complete. Samples in: {output_dir}")
