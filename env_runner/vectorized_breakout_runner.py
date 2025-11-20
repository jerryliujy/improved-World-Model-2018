import gymnasium as gym
import numpy as np
from gymnasium.vector import AsyncVectorEnv
import torch
import ale_py
from common.img_process import preprocess_breakout_image

class VectorizedBreakoutRunner:
    """
    Vectorized environment runner for parallel policy evaluation.
    """
    def __init__(self, 
                 env_name='ALE/Breakout-v5', 
                 num_envs=8, 
                 max_steps=1000, 
                 device='cuda',
                 to_grayscale=True):
        self.env_name = env_name
        self.num_envs = num_envs
        self.max_steps = max_steps
        self.device = device
        self.to_grayscale = to_grayscale

        # Create vectorized environments
        def make_env(env_id):
            def _make():
                return gym.make(self.env_name, render_mode='rgb_array')
            return _make
        
        self.envs = AsyncVectorEnv([make_env(i) for i in range(num_envs)])
        self.obs, _ = self.envs.reset()

    def evaluate(self, vision, predictor, controller_params_list,
                 controller_class, num_rollouts=1, max_steps=1000,
                 state_dim=288, action_dim=3):
        """
        Evaluate multiple controller parameters in parallel.
        
        Args:
            vision: Vision encoder model
            predictor: MDNRNN model
            controller_params_list: List of parameter vectors (popsize)
            controller_class: Controller class
            num_rollouts: Number of rollouts per policy
            
        Returns:
            rewards: List of average rewards for each policy
        """
        num_policies = len(controller_params_list)
        rewards = np.zeros(num_policies)
        
        # Evaluate each policy multiple times
        for rollout in range(num_rollouts):
            # Create batch of controller instances with different parameters
            controllers = self._load_weights(
                controller_class, controller_params_list, state_dim=state_dim, action_dim=action_dim
            )
            
            # Run parallel environments
            rollout_rewards = self._run_parallel_envs(
                vision, predictor, controllers, max_steps=max_steps
            )
            rewards += np.array(rollout_rewards)
        
        return (rewards / num_rollouts).tolist()
    
    def _load_weights(self, controller_class, params_list, state_dim, action_dim):
        """
        Load weights for all controllers without repeated tensor creation.
        """
        controllers = []
        with torch.no_grad():
            for params in params_list:
                ctrl = controller_class(state_dim=state_dim, action_dim=action_dim)
                ctrl.to(self.device)
                ctrl.eval()
                
                # load weights
                param_tensor = torch.tensor(params, dtype=torch.float32, device=self.device)
                torch.nn.utils.vector_to_parameters(param_tensor, ctrl.parameters())
                controllers.append(ctrl)
        
        return controllers
    
    def _process_actions(self, controllers, x):
        """
        Process actions for all controllers at once in one generation.
        
        Args:
            controllers (list): List of controller models
            x (torch.Tensor): Batch of input states
            
        Returns:
            torch.Tensor: Batch of actions
        """
        return torch.stack([controllers[i].get_action(x[i:i+1]) 
                            for i in range(x.size(0))], dim=0)

    
    def _run_parallel_envs(self, vision, predictor, controllers, max_steps=1000):
        """
        Run vectorized environments with multiple controllers.
        Key optimization: Process all environments simultaneously.
        """
        num_envs = self.num_envs
        num_policies = len(controllers)
        
        # Repeat controllers to match number of environments
        if num_policies != num_envs:
            controllers = controllers * (num_envs // num_policies + 1)
            controllers = controllers[:num_envs]
        
        cumulative_rewards = np.zeros(num_envs)
        dones = np.zeros(num_envs, dtype=bool)

        vision = vision.to(self.device)
        predictor = predictor.to(self.device)
        controllers = [ctrl.to(self.device) for ctrl in controllers]
        
        # Initialize hidden states once for all environments
        hidden = (torch.zeros(1, num_envs, predictor.hidden_dim, device=self.device),
                  torch.zeros(1, num_envs, predictor.hidden_dim, device=self.device))

        obs_tensor = torch.stack([
            preprocess_breakout_image(img, to_grayscale=self.to_grayscale)
            for img in self.obs
        ]).to(self.device)
        
        with torch.no_grad():
            for step in range(self.max_steps):
                if np.all(dones):
                    break
                
                # vision
                z_batch = vision.encode(obs_tensor)  # [num_envs, latent_dim*36]
                
                h = hidden[0].squeeze(0)
                x = torch.cat([z_batch, h], dim=-1)

                state_batch = torch.cat([z_batch, h], dim=-1)  # [num_envs, state_dim]
                
                # controller
                actions = self._process_actions(controllers, state_batch)
                actions = actions.detach().cpu().numpy()
                
                obs, rewards, dones_new, truncated, _ = self.envs.step(actions)
                
                # predictor
                z_batch_seq = z_batch.unsqueeze(1)  # [num_envs, 1, latent_dim*36]
                actions_seq = torch.from_numpy(actions).float().to(self.device).unsqueeze(1)
                _, _, _, hidden = predictor(z_batch_seq, actions_seq, h=hidden)
                
                cumulative_rewards += rewards * (~dones)
                dones = np.logical_or(dones, dones_new | truncated)
                obs_tensor = torch.stack([
                    preprocess_breakout_image(img, to_grayscale=self.to_grayscale)
                    for img in obs
                ]).to(self.device)

        if num_policies != num_envs:
            policy_rewards = []
            for i in range(num_policies):
                env_idx = i % num_envs  
                policy_rewards.append(cumulative_rewards[env_idx])
            return policy_rewards
        else:
            return cumulative_rewards.tolist()
        
        
    def close(self):
        self.envs.close()