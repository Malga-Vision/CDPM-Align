import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

from utils.main_script_utils import create_standard_parser, initialize_experiment, add_pretraining_args
from datasets.data_functions import setup_data_from_config
from pretraining.ddpm_divia.ddpm_train_test import train_diffusion_model, evaluate_diffusion_model


if __name__ == "__main__":
    # Parse arguments and initialize experiment
    parser = create_standard_parser("DDPM Pretraining", "./configs/pretraining/ddpm_config.json")
    add_pretraining_args(parser, task='ddpm')
    args = parser.parse_args()
    
    # Initialize experiment (handles config, device, seed, output dir, etc.)
    config, device, output_dir = initialize_experiment(args, task_type='pretraining')

    # Load dataset
    loaders = setup_data_from_config(config, use_train_val=True)
    train_loader = loaders['train_val']
    test_loader = loaders['test']

    if args.mode == 'train':
        # Train diffusion model
        print("----------------------------------------- START TRAINING -----------------------------------------")
        epochs = int(config['training']['iterations'] / (len(train_loader) / config['dataloader']['grad_accumulation']))
        print(f"Total epochs: {epochs} | Total iterations: {config['training']['iterations']} | Iterations per epoch: {len(train_loader)/config['dataloader']['grad_accumulation']}")
        train_diffusion_model(config, train_loader, config['environment']['output_dir'], device, continue_training=config['training']['resume'])
        print("----------------------------------------- END TRAINING -----------------------------------------")

    # Test diffusion model
    print("----------------------------------------- START TESTING -----------------------------------------")
    all_x, all_x_hat, model_iteration = evaluate_diffusion_model(config, test_loader, config['environment']['output_dir'], device)
    print("----------------------------------------- END TESTING -----------------------------------------")
