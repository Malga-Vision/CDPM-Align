"""Landmark Detection Training - Main Script

Usage:
    python -m landmark_detection.landmarks_main -c configs/landmarks/chest/cdpm_align.json
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

import torch

from utils.utils import get_model_parameters_info, get_pretrain_type_from_path
from utils.setup import setup_device, setup_optimizer, setup_scheduler, setup_loss_function
from utils.main_script_utils import create_standard_parser, initialize_experiment, add_downstream_args
from datasets.data_functions import setup_data_from_config
from landmark_detection.landmarks_train_test import train_and_validate, evaluate
from landmark_detection.landmarks_visualizations import plot_training_curves
from .model_factory import create_landmarks_model


def print_training_summary(config, model, num_landmarks, train_size, val_size, output_dir):
    """Print compact training summary."""
    print("\n" + "="*80)
    print("LANDMARKS DETECTION TRAINING")
    print("="*80)

    model_info = get_model_parameters_info(model, "Model")
    arch = config['model']['architecture'].lower()
    pretrained_path = config['model'].get('pretrained_path')
    encoder_weights = config['model'].get('encoder_weights')

    weight_type = get_pretrain_type_from_path(arch, pretrained_path, encoder_weights)

    encoder = config['model'].get('encoder')
    arch_str = config['model']['architecture'] + (f"+{encoder}" if encoder else "")

    print(f"Dataset: {config['dataset']['name']} | Landmarks: {num_landmarks}")
    print(f"Model: {arch_str} | Weights: {weight_type}")
    print(f"Params: {model_info['total_M']:.1f}M | Train: {train_size} | Val: {val_size}")
    print(f"Epochs: {config['training']['epochs']} | LR: {config['training']['learning_rate']} | BS: {config['dataloader']['batch_size']}")
    print(f"Output: {output_dir}")
    print("="*80 + "\n")


def main():
    # Parse arguments
    parser = create_standard_parser("Landmark Detection", "configs/landmarks/chest/cdpm_align.json")
    add_downstream_args(parser)
    args = parser.parse_args()
    
    # Initialize
    config, device, output_dir = initialize_experiment(args, task_type='landmarks')
    
    # Load data
    loaders = setup_data_from_config(config)
    train_loader = loaders['train']
    val_loader = loaders.get('val')
    test_loader = loaders['test']
    
    num_landmarks = train_loader.dataset.num_landmarks
    
    # Create model
    model = create_landmarks_model(config, device, num_landmarks)
    
    model_dir = os.path.join(output_dir, 'models')
    plot_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plot_dir, exist_ok=True)
    model_path = args.model_path or os.path.join(model_dir, 'best_checkpoint.pt')

    # Training mode
    if args.mode == 'train' and not os.path.exists(model_path):
        optimizer = setup_optimizer(config['training']['optimizer'], model.parameters(), 
                                   config['training']['learning_rate'], weight_decay=config['training']['weight_decay'])
        scheduler = setup_scheduler(config['training']['scheduler'], optimizer, 
                                   training_config=config['training'])
        loss_fn = setup_loss_function(config['training']['loss_function'])
        
        print_training_summary(config, model, num_landmarks, len(train_loader.dataset), 
                             len(val_loader.dataset), output_dir)
        
        start_time = time.time()
        model, history = train_and_validate(
            model=model,
            device=device,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_fn=loss_fn,
            epochs=config['training']['epochs'],
            save_path=model_dir,
            patience=config['training'].get('early_stopping_patience', 10),
            gradient_accumulation_steps=config['dataloader'].get('grad_accumulation', 1),
            continue_training=config['training'].get('resume', False)
        )
        
        plot_training_curves(history, save_dir=plot_dir)
        
        elapsed = time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))
        print(f"\n✅ Training completed in {elapsed}\n")
    
    # Evaluation mode
    if os.path.exists(model_path): #and not os.path.exists(os.path.join(output_dir, 'landmarks_results.txt')):
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
        print(f"✓ Loaded model from {model_path}")
        del checkpoint
        
        loss_fn = setup_loss_function(config['training']['loss_function'])
        
        print("\n" + "="*80)
        print(f"EVALUATING ON TEST SET ({len(test_loader.dataset)} samples)")
        print("="*80)
        
        start_time = time.time()
        results = evaluate(model, device, test_loader, loss_fn, num_landmarks, save_dir=output_dir)
        
        # Save results
        with open(os.path.join(output_dir, 'landmarks_results.txt'), 'w') as f:
            f.write(f"===== LANDMARKS DETECTION RESULTS ({num_landmarks} landmarks) =====\n\n")

            # --- Global metrics ---
            f.write("--- Global Error Statistics ---\n")
            f.write(f"MRE:  {results['mre']:.2f} ± {results['mre_std']:.2f}\n")
            f.write(f"P95:  {results['p95']:.2f}\n")
            f.write(f"ERE:  {results['ere']:.2f} ± {results['ere_std']:.2f}\n")
            f.write(f"Loss: {results['loss']:.4f}\n")
            f.write(f"Mode Prob: {np.mean(results['mode_probs']):.4f}\n")

            f.write(f"\n--- Per-Landmark Statistics ---\n")
            f.write(f"{'LM':>4s} {'MRE':>7s} {'±Std':>7s} {'P95':>7s}\n")
            for i in range(num_landmarks):
                f.write(f"  L{i+1:>2d} {results['per_landmark_mre'][i]:7.2f} {results['per_landmark_std'][i]:7.2f} "
                        f"{results['per_landmark_p95'][i]:7.2f}\n")

            f.write(f"\n--- SDR ---\n")
            for thresh, sdr in results['sdr'].items():
                f.write(f"  {thresh:.1f} mm: {sdr:.2f}%\n")
        
        # Create plots
        plt.figure(figsize=(12, 4))
        
        plt.subplot(131)
        plt.bar(range(1, len(results['per_landmark_mre']) + 1), results['per_landmark_mre'], 
                yerr=results['per_landmark_std'], capsize=5, alpha=0.7)
        plt.xlabel('Landmark')
        plt.ylabel('MRE (px)')
        plt.title('Per-Landmark Error')
        plt.grid(alpha=0.3)
        
        plt.subplot(132)
        thresholds = list(results['sdr'].keys())
        sdr_values = list(results['sdr'].values())
        plt.plot(thresholds, sdr_values, 'o-', linewidth=2, markersize=8)
        plt.xlabel('Threshold (px)')
        plt.ylabel('SDR (%)')
        plt.title('Detection Rate')
        plt.grid(alpha=0.3)
        
        plt.subplot(133)
        metrics = ['MRE', 'ERE', 'Loss']
        values = [results['mre'], results['ere'], results['loss']]
        bars = plt.bar(metrics, values, alpha=0.7, color=['skyblue', 'lightcoral', 'lightgreen'])
        for bar, val in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                    f'{val:.2f}', ha='center', va='bottom')
        plt.ylabel('Value')
        plt.title('Summary')
        plt.grid(alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, 'evaluation_summary.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Print results
        print(f"\n--- Global ---")
        print(f"MRE:  {results['mre']:.2f} ± {results['mre_std']:.2f} | P95: {results['p95']:.2f}")
        print(f"ERE:  {results['ere']:.2f} ± {results['ere_std']:.2f} | Loss: {results['loss']:.4f} | Mode Prob: {np.mean(results['mode_probs']):.4f}")
        print(f"\nSDR: " + " | ".join([f"{t:.1f}mm={v:.1f}%" for t, v in results['sdr'].items()]))
        
        elapsed = time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))
        print(f"\n✅ Evaluation completed in {elapsed}")
        print(f"💾 Results saved to {output_dir}\n")


if __name__ == "__main__":
    main()