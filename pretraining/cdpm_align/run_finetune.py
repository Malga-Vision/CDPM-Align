"""
CDPM Finetuning Entry Point.

Usage:
    python -m pretraining.cdpm_align.run_finetune --config configs/pretraining/cdpm_align.json
    python -m pretraining.cdpm_align.run_finetune --config configs/pretraining/cdpm_align.json --resume
    python -m pretraining.cdpm_align.run_finetune --config configs/pretraining/cdpm_align.json --mode eval
"""

import argparse
from utils.utils import load_config_with_overrides


def parse_args() -> dict:
    """Parse CLI args, load JSON config, return dict."""
    parser = argparse.ArgumentParser(description="CDPM Finetuning with Alignment")

    parser.add_argument("-c", "--config", type=str,
                        default="configs/pretraining/cdpm_align.json",
                        help="Path to JSON config file")
    parser.add_argument("--mode", type=str, choices=["train", "eval"])
    parser.add_argument("--pretrained_checkpoint", type=str)
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--img_size", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--device", type=str)
    parser.add_argument("--save_dir", type=str)
    parser.add_argument("--freq_save", type=int)
    parser.add_argument("--resume", action="store_true", default=None)
    parser.add_argument("--ddim_steps", type=int)
    parser.add_argument("--w", type=float)
    parser.add_argument("--images_to_sample", type=int)
    parser.add_argument("--channel", type=int)
    parser.add_argument("--T", type=int)
    parser.add_argument("--lambda_diff", type=float)
    parser.add_argument("--lambda_align", type=float)
    parser.add_argument("--no_gradient_checkpointing", dest="gradient_checkpointing", action="store_false", default=None)

    args = parser.parse_args()

    cli_overrides = {k: v for k, v in vars(args).items() if k != "config"}
    return load_config_with_overrides(args.config, cli_overrides)


def main():
    config = parse_args()

    print("=" * 60)
    print("CDPM Finetuning with DINO-style Alignment")
    print("=" * 60)
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()

    from .train_finetune import train, eval_generate

    if config['mode'] == "train":
        train(config)
    else:
        eval_generate(config)


if __name__ == "__main__":
    main()
