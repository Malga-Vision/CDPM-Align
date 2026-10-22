import argparse
import os
from .train import train, eval
from utils.utils import load_config_with_overrides

def parse_args():
    """Parse command-line arguments to override default configurations."""
    parser = argparse.ArgumentParser(
        description="Conditional Diffusion Training for Combined Chest & Hand Datasets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Mode
    parser.add_argument("--mode", type=str, choices=["train", "eval"], 
                        help="Mode: 'train' to train the model, 'eval' to evaluate/generate samples")
    
    # Data settings
    data_group = parser.add_argument_group('Data Settings')
    data_group.add_argument("--dataset_type", type=str, choices=["combined", "chest", "hand", "cephalo", "nih"],
                           help="Dataset type: 'combined', 'chest', 'hand', 'cephalo', or 'nih'")
    data_group.add_argument("--conditioned", dest='conditioned', action='store_true', default=None,
                           help="Enable class-conditional training")
    data_group.add_argument("--unconditioned", dest='conditioned', action='store_false', default=None,
                           help="Disable class conditioning (treat all data as single class)")
    data_group.add_argument("--data_dir", type=str, 
                           help="Directory containing datasets (chest and hand)")
    data_group.add_argument("--img_size", type=int, 
                           help="Image size (square)")
    data_group.add_argument("--img_channels", type=int, 
                           help="Number of image channels (1 for grayscale, 3 for RGB)")
    data_group.add_argument("--max_samples", type=int,
                           help="Maximum number of samples (for debugging, None = use all)")
    
    # Model architecture
    model_group = parser.add_argument_group('Model Architecture')
    model_group.add_argument("--channel", type=int, 
                            help="Base number of channels")
    model_group.add_argument("--attn_heads", type=int,
                            help="Number of attention heads (4-8 recommended for better representation)")
    model_group.add_argument("--dim_head", type=int,
                            help="Dimension per attention head (32=efficient, 64=more capacity)")
    model_group.add_argument("--use_linear_attn", dest='use_linear_attn', action='store_true', default=None,
                            help="Use memory-efficient LinearAttention (recommended for high-res)")
    model_group.add_argument("--no_linear_attn", dest='use_linear_attn', action='store_false', default=None,
                            help="Disable LinearAttention (saves a bit of memory but worse representation)")
    model_group.add_argument("--dropout", type=float, 
                            help="Dropout rate")
    model_group.add_argument("--T", type=int, 
                            help="Number of diffusion time steps")
    
    # Training hyperparameters
    train_group = parser.add_argument_group('Training Hyperparameters')
    train_group.add_argument("--batch_size", type=int, 
                            help="Batch size")
    train_group.add_argument("--iterations", type=int, 
                            help="Number of training iterations")
    train_group.add_argument("--iters_to_accumulate", type=int, 
                            help="Gradient accumulation steps")
    train_group.add_argument("--lr", type=float, 
                            help="Learning rate")
    train_group.add_argument("--mixed_precision", dest='mixed_precision', action='store_true', default=None,
                            help="Use mixed precision training (FP16) - saves memory")
    train_group.add_argument("--no_mixed_precision", dest='mixed_precision', action='store_false', default=None,
                            help="Disable mixed precision training")
    train_group.add_argument("--gradient_checkpointing", dest='gradient_checkpointing', action='store_true', default=None,
                            help="Enable gradient checkpointing (reduces memory, increases compute time)")
    train_group.add_argument("--no_gradient_checkpointing", dest='gradient_checkpointing', action='store_false', default=None,
                            help="Disable gradient checkpointing")
    
    # Sampling settings
    sample_group = parser.add_argument_group('Sampling Settings')
    sample_group.add_argument("--w", type=float, 
                             help="Classifier-free guidance weight for sampling")
    sample_group.add_argument("--images_to_sample", type=int, 
                             help="Number of images to sample per class")
    sample_group.add_argument("--ddim_steps", type=int,
                             help="Number of DDIM steps for faster sampling (None=full DDPM sampling, e.g., 50 for faster)")
    
    # Device
    parser.add_argument("--device", type=str, 
                       help="Device to use, e.g., 'cuda:0' or 'cpu'")
    
    # Paths
    path_group = parser.add_argument_group('Paths')
    path_group.add_argument("--save_dir", type=str, 
                           help="Directory to save model checkpoints")
    path_group.add_argument("--freq_save", type=int, 
                           help="Iteration frequency to save checkpoints")
    path_group.add_argument("--load_weights", type=str, 
                           help="Path to model checkpoint for evaluation")
    path_group.add_argument("--sampled_dir", type=str, 
                           help="Directory to save sampled images")
    
    # Resume and validation
    resume_group = parser.add_argument_group('Resume & Validation')
    resume_group.add_argument("--resume", dest='resume', action='store_true', default=None,
                             help="Resume training from a checkpoint")
    resume_group.add_argument("--resume_checkpoint", type=str,
                             help="Path to checkpoint file to resume from")
    resume_group.add_argument("--no_validate_at_save", dest='validate_at_save', action='store_false', default=None,
                             help="Disable validation at each checkpoint save")

    # Config file (lowest priority — CLI args override JSON values)
    parser.add_argument('-c', '--config', type=str, default=None,
                        help='Path to JSON config file (CLI args take precedence)')

    return parser.parse_args()

def main():
    args = parse_args()
    json_path = args.config or "configs/pretraining/cdpm.json"
    cli_overrides = {k: v for k, v in vars(args).items() if k != "config"}
    modelConfig = load_config_with_overrides(json_path, cli_overrides)
    
    print("=" * 80)
    print("CONDITIONAL DIFFUSION MODEL - COMBINED CHEST & HAND DATASETS")
    print("=" * 80)
    print(f"\nConfiguration:")
    for key, value in modelConfig.items():
        print(f"  {key}: {value}")
    print()

    # Run train or eval based on the mode
    if modelConfig["mode"] == "train":
        train(modelConfig)
    else:
        eval(modelConfig)

if __name__ == '__main__':
    main()


# Usage examples:
# Training:
#   python -m pretraining.cdpm.run -c configs/pretraining/cdpm_combined.json
# Evaluation:
#   python -m pretraining.cdpm.run -c configs/pretraining/cdpm_combined.json --mode eval --load_weights <save_dir>/combined/ckpt_45000_iterations.pt --images_to_sample 64