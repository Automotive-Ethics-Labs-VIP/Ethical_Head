"""
Replay Buffer Module
====================
This module implements the `ReplayBuffer` class, which serves as a temporary storage
for experience collection during the PPO training loop.

Purpose:
- Stores trajectories of (state, action, reward, value, log_prob, done).
- Allows for the computation of Generalized Advantage Estimation (GAE) over stored trajectories.
- Provides batch retrieval for PPO updates.

Key Concept:
PPO is an on-policy algorithm, meaning it learns from data collected by the *current* policy.
Therefore, this buffer is typically filled (e.g., to 2048 steps), used for an update,
and then immediately cleared. It does not store historical data like a replay buffer in DQN.
"""

import torch
import numpy as np
from algorithms.gae import compute_gae

class ReplayBuffer:
    """
    A buffer to store rollout data (trajectories) for PPO updates.
    
    Attributes:
        capacity (int): Maximum number of steps to store (default 2048).
        state_dim (int): Dimensionality of the state space (default 40).
        ptr (int): Current insertion index / count of elements.
    """
    def __init__(self, capacity: int = 2048, state_dim: int = 40):
        self.capacity = capacity
        self.state_dim = state_dim
        self.clear()

    def clear(self):
        """Resets the buffer."""
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        self.ptr = 0

    def store(self, state, action, reward, value, log_prob, done):
        """
        Stores a single transition.
        Inputs can be tensors or floats/ints. Convert to consistent format later.
        """
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
        self.ptr += 1

    def get_batch(self):
        """
        Returns all stored data as tensors.
        """
        # Convert lists to tensors
        # Assuming states are [40] tensors
        states_tensor = torch.stack(self.states)
        
        # Actions might be ints or tensors
        actions_tensor = torch.tensor(self.actions, dtype=torch.long)
        
        # Rewards, values, log_probs, dones are scalars
        rewards_tensor = torch.tensor(self.rewards, dtype=torch.float32)
        values_tensor = torch.tensor(self.values, dtype=torch.float32)
        log_probs_tensor = torch.tensor(self.log_probs, dtype=torch.float32)
        dones_tensor = torch.tensor(self.dones, dtype=torch.float32)
        
        return states_tensor, actions_tensor, rewards_tensor, values_tensor, log_probs_tensor, dones_tensor

    def compute_advantages_and_returns(self, last_value: float, gamma: float = 0.99, gae_lambda: float = 0.95):
        """
        Computes GAE advantages and returns for the stored trajectory.
        
        Args:
            last_value: Value estimate of the state AFTER the last step (for bootstrapping).
            gamma: Discount factor.
            gae_lambda: GAE smoothing parameter.
            
        Returns:
            advantages: Tensor [T]
            returns: Tensor [T]
        """
        # We need values for t=0...T (stored) AND t=T+1 (last_value)
        # self.values has T elements.
        values_list = self.values + [last_value]
        values_tensor = torch.tensor(values_list, dtype=torch.float32)
        
        rewards_tensor = torch.tensor(self.rewards, dtype=torch.float32)
        dones_tensor = torch.tensor(self.dones, dtype=torch.float32)
        
        # Call GAE function
        advantages, returns = compute_gae(
            rewards_tensor, 
            values_tensor, 
            dones_tensor, 
            gamma=gamma, 
            gae_lambda=gae_lambda
        )
        
        return advantages, returns

    def __len__(self):
        return self.ptr