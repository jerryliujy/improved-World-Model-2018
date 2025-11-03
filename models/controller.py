import torch
import torch.nn as nn

class Controller(nn.Module):
    """
    Controller neural network that maps state (z + h) to actions.
    
    Args:
        state_dim (int): Dimension of the input state (latent + hidden state)
        action_dim (int): Dimension of the output action space
    """
    def __init__(self, state_dim, action_dim=3):
        super(Controller, self).__init__()
        
        self.model = nn.Sequential(
            nn.Linear(state_dim, action_dim),
        )

    def forward(self, x):
        """
        Forward pass through the controller network.
        
        Args:
            x (torch.Tensor): Input tensor containing latent state + hidden state
            
        Returns:
            torch.Tensor: Action values for steering, acceleration, and brake
        """
        raw_actions = self.model(x)
        
        # Apply appropriate activations to each action dimension
        steering = torch.tanh(raw_actions[:, 0:1])        # [-1, 1]
        gas = torch.sigmoid(raw_actions[:, 1:2])          # [0, 1]
        brake = torch.sigmoid(raw_actions[:, 2:3]) * 0.8  # [0, 0.8]

        # Ensure brake is reduced when accelerating (realistic vehicle behavior)
        brake = brake * (1-gas)
        
        # Combine all actions into one tensor
        actions = torch.cat([steering, gas, brake], dim=1)
    
        return actions

    def get_action(self, state):
        """
        Get action for a given state without gradient computation.
        
        Args:
            state (torch.Tensor): Current state (z + h), shape (state_dim,) or (batch, state_dim)
            
        Returns:
            torch.Tensor: Action to take, shape (action_dim,)
        """
        with torch.no_grad():
            # Handle 1D input (single step)
            if state.dim() == 1:
                state = state.unsqueeze(0)
            action = self.forward(state)
        return action.squeeze()
    
    
    def loss(self, predicted_actions, target_actions):
        """
        Compute controller loss (MSE between predicted and target actions).
        
        Args:
            predicted_actions: Actions predicted by the controller
            target_actions: Target/true actions from the dataset
            
        Returns:
            loss: Scalar MSE loss value
        """
        return nn.functional.mse_loss(predicted_actions, target_actions)