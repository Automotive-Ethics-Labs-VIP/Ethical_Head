import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class RewardModel(nn.Module):
    """
    A 3-layer feed-forward reward model for ethical decision scoring.
    Scores (state, action) pairs with a scalar reward.

    Architecture:
        Input: [state (40) + action one-hot (5)] = 45
        Hidden: Linear(64 → 32) + ReLU
        Output: Linear(32 → 1)

    Output:
        A single scalar V(s) representing the expected return from the given state.
    """

    def __init__(self, state_dim=40, action_dim=5):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        input_dim = state_dim + action_dim

        #### Define layers
        
        # hidden layers
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        
        # output value
        self.fc3 = nn.Linear(32, 1)

    def forward(self, states, actions):
        """
        Args:
            states: Tensor [batch, 40]
            actions: Tensor [batch] (integer action indices)
        Returns:
            rewards: Tensor [batch] (scalar reward per state-action)
        """

        # Validate inputs
        if states.dim() != 2 or states.shape[1] != self.state_dim:
            raise ValueError(f"Expected states [batch, {self.state_dim}], got {states.shape}")
        
        if actions.dim() != 1 or actions.shape[0] != states.shape[0]:
            raise ValueError(f"Expected actions [batch], got {actions.shape}")
        
        # One-hot encode actions
        actions_onehot = F.one_hot(actions.long(), num_classes=self.action_dim).float()
        
        # Concatenate and forward pass
        x = torch.cat([states, actions_onehot], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        reward = self.fc3(x).squeeze(-1)
        
        # Check for non-finite outputs
        if not torch.isfinite(reward).all():
            raise RuntimeError("RewardModel produced non-finite output")
        
        return reward