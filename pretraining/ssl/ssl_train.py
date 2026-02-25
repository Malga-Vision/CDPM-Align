import torch
import numpy as np
import logging
import time
from tqdm import tqdm
import os
from utils.utils import safe_save


def train_ssl_model(model, optimizer, train_loader, device, config=None):
    """
    Train a self-supervised learning model.
    
    Args:
        model (nn.Module): The SSL model to train
        optimizer (torch.optim.Optimizer): Optimizer for training
        train_dataloader (DataLoader): DataLoader for training data
        device (torch.device): Device to train on
        config (dict, optional): Configuration dictionary with training parameters
            - epochs: Maximum number of epochs to train
            - grad_accumulation: Number of steps for gradient accumulation
            - save_model_path: Path to save model checkpoints
            - max_iterations: Maximum number of iterations to train
            
    Returns:
        model (nn.Module): The trained SSL model
    """

    # Extract configuration parameters
    grad_accumulation = config['dataloader']['grad_accumulation']
    max_iterations = config['training']['iterations']
    epochs = int(config['training']['iterations'] / (len(train_loader) / config['dataloader']['grad_accumulation']))
    freq_checkpoint = config['training']['freq_checkpoint']
    
    # Create directory for saving models
    save_model_path = config['environment']['output_dir']
    os.makedirs(save_model_path, exist_ok=True)
    
    # Check if there are any saved models to resume training
    if config['training']['resume']:    
        # Get the latest model weights
        model_path = f'{save_model_path}/last_model.pth'
        checkpoint = torch.load(model_path, map_location=device)    

        # Load the model weights and optimizer state
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        starting_epoch = checkpoint['epoch']
        n_iter = checkpoint['iteration']
        print(f"Loading model weights from epoch {starting_epoch} and iteration {n_iter}")
        del checkpoint
    else:
        starting_epoch = 0
        n_iter = 0
    
    # Initialize training variables
    epochs_losses = []
    
    # Record training start time
    start_time = time.time()
    
    # Switch to train mode
    model.train()
    
    try:
        # Create a progress bar for epochs
        epoch_pbar = tqdm(range(starting_epoch, epochs), initial=starting_epoch, total=epochs, desc="Epoch")
        print("----------------------------------------- START TRAINING -----------------------------------------")

        for epoch in epoch_pbar:
            epoch_loss = 0
            
            # Create a progress bar for batches
            batch_pbar = tqdm(train_loader, desc=f"Training Epoch {epoch}", leave=False)
            
            for batch_idx, data in enumerate(batch_pbar):
                # Move data to device
                images = data['image'].to(device)
                        
                # Forward pass
                loss = model(images)

                # Normalize loss for gradient accumulation (must happen before backward)
                loss = loss / grad_accumulation
                loss_value = loss.item()
                epoch_loss += loss_value

                # Backward pass
                loss.backward()
                
                # Update progress bar with loss and iteration info
                batch_pbar.set_postfix({
                    "loss": f"{loss_value:.4f}", 
                    "iter": n_iter
                })
                
                # Perform optimizer step after gradient accumulation
                if ((batch_idx + 1) % grad_accumulation == 0) or (batch_idx + 1 == len(train_loader)):
                    optimizer.step()
                    optimizer.zero_grad()
                    n_iter += 1

                # Save backbone model at specified intervals
                if n_iter % freq_checkpoint == 0 and n_iter != 0 and batch_idx % grad_accumulation == 0:
                    print(f"Saving model weights at epoch {epoch} and iteration {n_iter}")
                    
                    # Save the backbone weights for downstream tasks
                    torch.save(model.backbone.state_dict(), 
                              f'{save_model_path}/model_epoch{epoch}_iter{n_iter}.pth')
                                
                    # Exit training if max iterations reached
                    if n_iter >= max_iterations:
                        print(f"Reached {max_iterations} iterations. Exiting training.")
                        
                        # Save final backbone model
                        torch.save(model.backbone.state_dict(), 
                                 f'{save_model_path}/model_final_epoch{epoch}_iter{n_iter}.pth')
                        
                        # Also save as encoder.pth for easy loading in downstream tasks
                        torch.save(model.backbone.state_dict(), 
                                 f'{save_model_path}/encoder.pth')
                        print(f"Encoder saved to {save_model_path}/encoder.pth")
                        
                        return model
            
            # Calculate average loss for the epoch
            avg_loss = epoch_loss / len(train_loader)
            epochs_losses.append(avg_loss)
            
            # Update epoch progress bar with loss info
            epoch_pbar.set_postfix({
                "loss": f"{avg_loss:.4f}",
                "iter": n_iter
            })
            
            # Log epoch summary
            if epoch % 5 == 0:
                message = f"Epoch {epoch}, Iteration: {n_iter}, Loss: {avg_loss:.4f}"
                print(message)

            # Save checkpoint once per epoch (not per optimizer step)
            state = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'iteration': n_iter
            }
            safe_save(state, f'{save_model_path}/last_model.pth')
            del state

    except (KeyboardInterrupt, StopIteration, SystemExit, Exception) as e:
        # Handle training interruptions
        print("\nTraining interrupted. Saving final state...")
        if isinstance(e, Exception) and not isinstance(e, (KeyboardInterrupt, StopIteration, SystemExit)):
            print(f"Error: {e}")
        
        # Save the final state
        state = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'iteration': n_iter
        }
        safe_save(state, f'{save_model_path}/last_model.pth')
        print(f"Final state saved in {save_model_path}/last_model.pth")
        
        # Also save encoder for downstream tasks
        torch.save(model.backbone.state_dict(), f'{save_model_path}/encoder.pth')
        print(f"Encoder saved to {save_model_path}/encoder.pth")
        
    finally:
        # Log training completion time
        training_time = time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))
        print(f"Training completed in {training_time}")
        
        # Ensure encoder is saved at the end of training (if not already saved)
        encoder_path = f'{save_model_path}/encoder.pth'
        if not os.path.exists(encoder_path):
            torch.save(model.backbone.state_dict(), encoder_path)
            print(f"Encoder saved to {encoder_path}")
    
    return model