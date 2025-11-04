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
import cma
from datasets.dataloader import get_dataloaders
from common.model_loader import save_checkpoint, load_checkpoint


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
        
        # configure data loader 
        self.train_loader, self.val_loader, self.test_loader = get_dataloaders(
            h5_path=cfg.dataset.train,
            batch_size=cfg.training.batch_size,
            train_ratio=cfg.dataset.train_ratio,
            val_ratio=cfg.dataset.val_ratio,
            test_ratio=cfg.dataset.test_ratio,
            to_grayscale=cfg.dataset.to_grayscale,
            num_workers=cfg.dataset.num_workers
        )
        
        # configure env runner
        self.env_runner = hydra.utils.instantiate(cfg.env_runner)
        
        # configure model
        self.vision = hydra.utils.instantiate(cfg.vision)
        self.predictor = hydra.utils.instantiate(cfg.predictor)
        self.controller = hydra.utils.instantiate(cfg.controller)

        if cfg.training.vision_resume:
            load_checkpoint(self.vision, cfg.training.vision_path)
        if cfg.training.predictor_resume:
            load_checkpoint(self.predictor, cfg.training.predictor_path)
        if cfg.training.controller_resume:
            load_checkpoint(self.controller, cfg.training.controller_path)
            
        # configure training setting
        self.model_to_train = None
        if cfg.training.stage == 1:
            self.model_to_train = self.vision
        elif cfg.training.stage == 2:
            self.model_to_train = self.predictor
        elif cfg.training.stage == 3:
            self.model_to_train = self.controller
        self.optimizer = hydra.utils.instantiate(
            cfg.training.optimizer,
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
    
    
    def _prepare_batch(self, batch_data, device):
        """
        Prepare batch data based on training stage.
        
        Stage 1: Input is images, target is images
        Stage 2: Input is (z, a), target is z_next
        Stage 3: Input is (z, h), target is actions
        
        Returns:
            dict with 'inputs' and 'targets' for model.loss()
        """
        stage = self.cfg.training.stage
        
        if stage == 1:
            # VAE: images -> encoded images
            images, _, _, _, _ = batch_data
            images = images.to(device)
            return {
                'inputs': [images],
                'targets': [images]
            }
        elif stage == 2:
            # MDNRNN: (z, a) -> z_next distribution
            # Note: This requires pre-encoded observations from VAE
            # Placeholder implementation
            images, actions, rewards, dones, next_images = batch_data
            images = images.to(device)
            next_images = next_images.to(device)
            vision = self.vision.to(device)
            vision.eval()
            z = vision.encode(images).unsqueeze(1)  
            next_z = vision.encode(next_images).unsqueeze(1)
            actions = actions.to(device).unsqueeze(1)
            return {
                'inputs': [z, actions],
                'targets': [next_z]  # Target should be z_next after VAE encoding
            }
        else:
            raise ValueError(f"Unknown training stage: {stage}")
    
    ###############################################
    ###############################################
    ## Unified training pipeline using DL method ##
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
                    loss = model.loss(*model_output, *batch_data['targets'])
                    
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
                        loss = model.loss(*model_output, *batch_data['targets'])
                        val_epoch_loss += loss.item()
                        pbar.set_postfix({'loss': f"{loss.item():.6f}"})
            
            avg_val_loss = val_epoch_loss / len(self.val_loader)
            self.val_losses.append(avg_val_loss)
            
            # Log epoch results
            print(f'Epoch {epoch + 1}/{num_epochs} | '
                  f'Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}')
            
            # Save checkpoint periodically
            if (epoch + 1) % save_interval == 0:
                save_checkpoint(model, self.cfg.training.checkpoint_dir, self.cfg.training.stage, epoch)
        
        # Save final model
        save_checkpoint(model, self.cfg.training.checkpoint_dir, self.cfg.training.stage, epoch)
        
        # Plot training curves
        self._plot_losses(plot_save_dir)
        

    
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
    ###############################################
    ###############################################
    

    ###############################################
    ###############################################
    #### CMA-ES Training for Controller Model  ####
    def train_cma_es(self):
        """
        Train controller using CMA-ES evolutionary strategy.
        
        This method evolves controller parameters to maximize reward in the environment.
        Uses env_runner to evaluate each generation.
        """
        if self.cfg.training.stage != 3:
            print("CMA-ES training is only for Stage 3 (Controller). Please set training.stage=3")
            return
        
        # Load CMA-ES hyperparameters from config
        cma_cfg = self.cfg.training.cma_es
        initial_sigma = cma_cfg.initial_sigma
        max_generations = cma_cfg.max_generations
        popsize = cma_cfg.popsize
        checkpoint_interval = cma_cfg.checkpoint_interval
        num_rollouts = cma_cfg.num_rollouts
        max_steps = cma_cfg.max_steps
        
        # Setup metrics tracking
        metrics = {
            'generation': [],
            'best_reward': [],
            'mean_reward': [],
            'worst_reward': []
        }
        
        # Create directory for checkpoints
        checkpoint_dir = self.cfg.training.checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Get initial controller parameters
        controller = self.model_to_train
        initial_params = torch.nn.utils.parameters_to_vector(
            controller.parameters()).detach().cpu().numpy()

        # Initialize CMA-ES optimizer
        es = cma.CMAEvolutionStrategy(initial_params, initial_sigma, {'popsize': popsize})
        
        # Main training loop
        for generation in range(max_generations):
            # Generate population for this generation
            solutions = es.ask()  

            # Evaluate each solution multiple times and average
            generation_rewards = []
            for solution in solutions:
                rewards_for_solution = []
                for rollout in range(num_rollouts):
                    # Load solution parameters into controller
                    torch.nn.utils.vector_to_parameters(
                        torch.tensor(solution, dtype=torch.float32, device=self.device),
                        controller.parameters()
                    )
                    
                    # Run evaluation
                    total_reward = self._evaluate_controller(max_steps=max_steps)
                    rewards_for_solution.append(total_reward)
                
                # Average reward for this solution
                avg_reward = np.mean(rewards_for_solution)
                generation_rewards.append(avg_reward)
            
            generation_rewards = np.array(generation_rewards)

            # Update CMA-ES with rewards (negative because CMA-ES minimizes)
            es.tell(solutions, [-r for r in generation_rewards])
            
            # Calculate and log training statistics
            log = (f'Generation {generation + 1}/{max_generations} | '
                f'Best Reward: {np.max(generation_rewards):.2f} | '
                f'Avg Reward: {np.mean(generation_rewards):.2f} | '
                f'Worst: {np.min(generation_rewards):.2f} | '
                f'Sigma: {es.sigma:.4f}')
            print(log)

            # Update metrics
            metrics['generation'].append(generation + 1)
            metrics['best_reward'].append(float(np.max(generation_rewards)))
            metrics['worst_reward'].append(float(np.min(generation_rewards)))
            metrics['mean_reward'].append(float(np.mean(generation_rewards)))
                    
            # Save checkpoint at intervals
            if (generation + 1) % checkpoint_interval == 0:
                torch.nn.utils.vector_to_parameters(
                    torch.tensor(es.result.xbest, dtype=torch.float32, device=self.device),
                    controller.parameters()
                )
                checkpoint_path = os.path.join(checkpoint_dir, f'stage_3_cma_gen_{generation + 1:04d}.pth')
                torch.save(controller.state_dict(), checkpoint_path)
                print(f'Checkpoint saved: {checkpoint_path}')
        
        # Save final model
        torch.nn.utils.vector_to_parameters(
            torch.tensor(es.result.xbest, dtype=torch.float32, device=self.device),
            controller.parameters()
        )
        final_path = os.path.join(checkpoint_dir, 'stage_3_final.pth')
        torch.save(controller.state_dict(), final_path)
        print(f'Final model saved to {final_path}')
        
        return es, metrics
    
    
    def _evaluate_controller(self, max_steps: int = 1000, num_episodes: int = 1):
        """
        Evaluate controller using env_runner.
        
        Args:
            max_steps: maximum steps per episode
            num_episodes: number of episodes to average over
            
        Returns:
            average reward across episodes
        """
        results = self.env_runner.run(
            vision=self.vision,
            predictor=self.predictor,
            controller=self.model_to_train,
            num_episodes=num_episodes,
            max_steps=max_steps,
            render=False
        )
        
        return results['avg_reward']
    ###############################################
    ###############################################