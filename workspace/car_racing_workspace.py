import torch
import os
import hydra
import numpy as np
import random
import cma
import gymnasium as gym
from omegaconf import OmegaConf
from workspace.base_workspace import BaseWorkspace
from datasets.dataloader import get_car_racing_loaders

class CarRacingWorkspace(BaseWorkspace):
    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        # configure data loader 
        self.train_loader, self.val_loader, self.test_loader = get_car_racing_loaders(
            h5_path=cfg.dataset.train,
            batch_size=cfg.training.batch_size,
            train_ratio=cfg.dataset.train_ratio,
            val_ratio=cfg.dataset.val_ratio,
            test_ratio=cfg.dataset.test_ratio,
            to_grayscale=cfg.dataset.to_grayscale,
            num_workers=cfg.dataset.num_workers
        )

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
            cfg.training.optimizer,
            params=self.model_to_train.parameters()
        )
        
        if cfg.training.train_method == 'cma_es':
            self.train = self.train_cma_es
            
        self.device = cfg.device

        # configure env
        env = gym.make('CarRacing-v3', render_mode='rgb_array')
        self.env = env
        
        
    def train_cma_es(self, 
                     controller_class, 
                     memory, 
                     max_generations=100, 
                     max_steps=1000, 
                     popsize=16, 
                     checkpoint=10, 
                     rollouts=7, 
                     device='cpu'):
        INITIAL_SIGMA = 0.1
        SIGMA_DECAY = 0.992
        # Setup metrics tracking
        metrics = {
            'generation': [],
            'best_reward': [],
            'mean_reward': [],
            'worst_reward': []
        }
        
        # Create directory for checkpoints
        os.makedirs('checkpoints/cma', exist_ok=True)

        model = self.model_to_train
        initial_params = torch.nn.utils.parameters_to_vector(
            model.parameters()).detach().cpu().numpy()

        # Initialize CMA-ES optimizer
        es = cma.CMAEvolutionStrategy(initial_params, INITIAL_SIGMA, {'popsize': popsize})
        
        # Main training loop
        for generation in range(141, max_generations+1):
            # Generate population for this generation
            solutions = es.ask()  

            # Evaluate each solution multiple times and average
            mean_rewards = []
            for _ in range(rollouts):
                rewards = self.eval(solutions, controller_class, max_steps, memory)
                mean_rewards.append(rewards)
            mean_rewards = np.mean(mean_rewards, axis=0)       

            # Update CMA-ES with rewards (negative because CMA-ES minimizes)
            es.tell(solutions, [-r for r in mean_rewards])
            
            # Calculate and log training statistics
            log = (f'Generation ({generation}/{max_generations}) | '
                f'Best Reward: {round(np.max(mean_rewards))} | '
                f'Avg Reward: {np.mean(mean_rewards):.2f} | '
                f'Worst: {round(np.min(mean_rewards))} | '
                f'Sigma: {es.sigma:.4f}')
            print(log)

            # Update metrics
            metrics['generation'].append(generation)
            metrics['best_reward'].append(np.max(mean_rewards))
            metrics['worst_reward'].append(np.min(mean_rewards))
            metrics['mean_reward'].append(np.mean(mean_rewards))
                    
            # Save metrics and best controller at checkpoints
            if generation % checkpoint == 0:
                best_controller = Controller(state_dim=LATENT_DIM + HIDDEN_DIM, 
                                        action_dim=ACTION_DIM).to(device)
                torch.nn.utils.vector_to_parameters(
                    torch.tensor(es.result.xbest, dtype=torch.float32).to(device),
                    best_controller.parameters())
                torch.save(best_controller.state_dict(), 
                        f'checkpoints/cma/controller_{generation}.pth')
                print('--Checkpoint: best controller saved')
        
        # Save final model
        best_controller = Controller(state_dim=LATENT_DIM + HIDDEN_DIM, 
                                action_dim=ACTION_DIM).to(device)
        torch.nn.utils.vector_to_parameters(
            torch.tensor(es.result.xbest, dtype=torch.float32).to(device),
            best_controller.parameters())
        torch.save(best_controller.state_dict(), f'checkpoints/controller.pth')
        return es, metrics, es.result.xbest
        
        


    def eval(self, render: bool = False, num_episodes: int = 1, max_steps: int = 1000):
        """
        Evaluate the trained models in the environment.
        
        Args:
            render: Whether to render the environment
            num_episodes: Number of episodes to run for evaluation
            max_steps: Maximum steps per episode
            
        Note: Full evaluation requires additional constants and models to be properly configured.
        Constants needed: LATENT_DIM, HIDDEN_DIM
        """
        if not hasattr(self, 'vision') or not hasattr(self, 'predictor') or not hasattr(self, 'controller'):
            print("Models not properly initialized for evaluation.")
            return
        
        env = self.env
        total_rewards = []
        
        for episode in range(num_episodes):
            done = False
            cumulative_reward = 0
            obs, _ = env.reset()
            step_count = 0
            
            # Note: For full implementation, you need to:
            # 1. Define HIDDEN_DIM constant
            # 2. Initialize hidden state: h = (torch.zeros(1, HIDDEN_DIM), torch.zeros(1, HIDDEN_DIM))
            # 3. Encode observation: z = self.vision.encode(obs[np.newaxis, ...])
            # 4. Get action from controller
            # 5. Update LSTM hidden state
            
            while not done and step_count < max_steps:
                # Placeholder: just take a random action for now
                action = env.action_space.sample()
                obs, reward, done, truncated, info = env.step(action)
                
                if render:
                    env.render()
                
                cumulative_reward += reward
                step_count += 1
            
            total_rewards.append(cumulative_reward)
            print(f'Episode {episode + 1} | Reward: {cumulative_reward:.2f} | Steps: {step_count}')
        
        if render:
            env.close()
        
        print(f'Average Reward: {np.mean(total_rewards):.2f} | Std: {np.std(total_rewards):.2f}')
        return np.mean(total_rewards)