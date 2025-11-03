from typing import Optional
import os
import numpy as np
import random
from tqdm import tqdm
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
import torch
import matplotlib.pyplot as plt


class BaseWorkspace:
    def __init__(self, cfg: OmegaConf, output_dir: Optional[str]=None):
        self.cfg = cfg
        self._output_dir = output_dir

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure training state
        self.global_step = 0
        self.epoch = 0
        
        # configure model
        self.vision = hydra.utils.instantiate(cfg.vision)
        self.predictor = hydra.utils.instantiate(cfg.predictor)
        self.controller = hydra.utils.instantiate(cfg.controller)
        
        # configure training setting
        self.model_to_train = None
        if cfg.training.stage == 1:
            self.model_to_train = self.vision
        elif cfg.training.stage == 2:
            self.model_to_train = self.predictor
        elif cfg.training.stage == 3:
            self.model_to_train = self.controller
        self.optimizer = hydra.utils.instantiate(
            cfg.optimizer,
            params=self.model_to_train.parameters()
        )
        
        if cfg.training.train_method == 'cma_es':
            self.train = self.train_cma_es
        
        self.device = cfg.device
        
        # configure eval
        self.save_video = cfg.eval.save_video
        self.video_filename = cfg.eval.video_filename

        # Initialize loss tracking for plotting
        self.train_losses = []
        self.val_losses = []

    @property
    def output_dir(self):
        output_dir = self._output_dir
        if output_dir is None:
            output_dir = HydraConfig.get().runtime.output_dir
        return output_dir

    def train(self):
        """
        Unified training loop for all deep learning models (stage 1, 2, 3).
        
        Training flow:
        1. Forward pass: model(batch) -> output
        2. Loss computation: model.loss(output, batch) -> loss (scalar)
        3. Backward pass and optimization
        
        Supports multiple stages:
        - Stage 1: VAE training (vision encoder)
        - Stage 2: MDNRNN training (memory model)
        - Stage 3: Controller training (evolutionary strategy)
        """
        if self.model_to_train is None:
            print("No model to train. Please check the training configuration.")
            return 
        
        # Get training config parameters
        num_epochs = self.cfg.training.num_epochs
        save_interval = self.cfg.training.get('save_interval', 10)
        plot_save_dir = self.cfg.training.get('plot_save_dir', 'outputs/plots')
        
        device = self.device
        model = self.model_to_train
        model.to(device)
        
        os.makedirs(self.cfg.training.checkpoint_dir, exist_ok=True)
        os.makedirs(plot_save_dir, exist_ok=True)

        # Clear loss tracking
        self.train_losses = []
        self.val_losses = []

        for epoch in range(num_epochs):
            train_epoch_loss = 0.0
            val_epoch_loss = 0.0
            
            # ==================== Training Phase ====================
            model.train()
            with tqdm(self.train_loader, desc=f'Train Epoch {epoch + 1}/{num_epochs}') as pbar:
                for batch_data in pbar:
                    # Extract batch based on stage
                    batch_data = self._prepare_batch(batch_data, device)
                    
                    # Forward pass: model processes batch directly
                    model_output = model(*batch_data['inputs'])
                    
                    # Loss computation: pass output to model.loss()
                    loss = model.loss(model_output, *batch_data['targets'])
                    
                    # Backward pass and optimization
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    train_epoch_loss += loss.item()
                    pbar.set_postfix({'loss': f"{loss.item():.6f}"})
                    self.global_step += 1
                    
            avg_train_loss = train_epoch_loss / len(self.train_loader)
            self.train_losses.append(avg_train_loss)
            
            # ==================== Validation Phase ====================
            model.eval()
            with torch.no_grad():
                with tqdm(self.val_loader, desc=f'Val Epoch {epoch + 1}/{num_epochs}') as pbar:
                    for batch_data in pbar:
                        batch_data = self._prepare_batch(batch_data, device)
                        model_output = model(*batch_data['inputs'])
                        loss = model.loss(model_output, *batch_data['targets'])
                        val_epoch_loss += loss.item()
                        pbar.set_postfix({'loss': f"{loss.item():.6f}"})
            
            avg_val_loss = val_epoch_loss / len(self.val_loader)
            self.val_losses.append(avg_val_loss)
            
            # Log epoch results
            print(f'Epoch {epoch + 1}/{num_epochs} | '
                  f'Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}')
            
            # Save checkpoint periodically
            if (epoch + 1) % save_interval == 0:
                self._save_checkpoint(epoch)
        
        # Save final model
        final_checkpoint_path = os.path.join(
            self.cfg.training.checkpoint_dir, 
            f'stage_{self.cfg.training.stage}_final.pth'
        )
        torch.save(model.state_dict(), final_checkpoint_path)
        print(f'Final model saved to {final_checkpoint_path}')
        
        # Plot training curves
        self._plot_losses(plot_save_dir)
        

    def _prepare_batch(self, batch_data, device):
        """
        Prepare batch data based on training stage.
        
        Stage 1 (VAE): Input is images, target is images
        Stage 2 (MDNRNN): Input is (z, a), target is z_next
        Stage 3 (Controller): Input is (z, h), target is actions
        
        Returns:
            dict with 'inputs' and 'targets' for model.loss()
        """
        stage = self.cfg.training.stage
        
        if stage == 1:
            # VAE: images -> encoded images
            images, _, _, _ = batch_data
            images = images.to(device)
            return {
                'inputs': [images],
                'targets': [images]
            }
        elif stage == 2:
            # MDNRNN: (z, a) -> z_next distribution
            # Note: This requires pre-encoded observations from VAE
            # Placeholder implementation
            images, actions, rewards, dones = batch_data
            images = images.to(device)
            actions = actions.to(device)
            return {
                'inputs': [images, actions],
                'targets': [images]  # Target should be z_next after VAE encoding
            }
        elif stage == 3:
            # Controller: (z, h) -> actions
            # Note: This requires pre-encoded observations and RNN hidden states
            # Placeholder implementation
            images, actions, rewards, dones = batch_data
            images = images.to(device)
            actions = actions.to(device)
            return {
                'inputs': [images],
                'targets': [actions]
            }
        else:
            raise ValueError(f"Unknown training stage: {stage}")
    
    
    def _save_checkpoint(self, epoch):
        """Save model checkpoint."""
        checkpoint_path = os.path.join(
            self.cfg.training.checkpoint_dir,
            f'stage_{self.cfg.training.stage}_epoch_{epoch + 1:04d}.pth'
        )
        torch.save(self.model_to_train.state_dict(), checkpoint_path)
        print(f'Checkpoint saved: {checkpoint_path}')
    
    
    def _plot_losses(self, save_dir='outputs/plots'):
        """
        Plot training and validation loss curves.
        
        Args:
            save_dir: Directory to save the plot
        """
        os.makedirs(save_dir, exist_ok=True)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        epochs = np.arange(1, len(self.train_losses) + 1)
        
        ax.plot(epochs, self.train_losses, 'b-', label='Train Loss', linewidth=2)
        ax.plot(epochs, self.val_losses, 'r-', label='Val Loss', linewidth=2)
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title(f'Training Curves - Stage {self.cfg.training.stage}', fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # Save figure
        stage = self.cfg.training.stage
        save_path = os.path.join(save_dir, f'stage_{stage}_training_curves.png')
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'Training curves saved to: {save_path}')
        plt.close(fig)
        
        # Also save loss values as CSV for future reference
        import csv
        csv_path = os.path.join(save_dir, f'stage_{stage}_losses.csv')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Epoch', 'Train Loss', 'Val Loss'])
            for i, (train_loss, val_loss) in enumerate(zip(self.train_losses, self.val_losses)):
                writer.writerow([i + 1, train_loss, val_loss])
        print(f'Loss values saved to: {csv_path}')