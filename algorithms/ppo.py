"""
Proximal Policy Optimization (PPO) Module
=========================================
This module implements the PPO algorithm, a policy gradient method for reinforcement learning.

PPO alternates between sampling data through interaction with the environment and optimizing 
a "surrogate" objective function using stochastic gradient ascent.

Key Components:
1.  **Clipped Surrogate Objective**:
    - Maximizes the probability of good actions while limiting the change in policy.
    - Uses a clipping parameter (epsilon) to prevent large policy updates.
    - Loss = -min(r * A, clip(r, 1-eps, 1+eps) * A)

2.  **Value Function Loss**:
    - Trains the critic (ValueNetwork) to predict expected returns.
    - Uses Mean Squared Error (MSE) between predicted values and actual returns.

3.  **Entropy Bonus**:
    - Adds an entropy term to the loss to encourage exploration.
    - Prevents premature convergence to suboptimal deterministic policies.

Functions:
- `compute_policy_loss`: Calculates actor loss and entropy.
- `compute_value_loss`: Calculates critic loss.
- `update`: Performs a single optimization step for both networks.

Reference:
Schulman et al. (2017). Proximal Policy Optimization Algorithms. arXiv:1707.06347
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

def compute_policy_loss(
    policy_net: nn.Module,
    states: torch.Tensor,
    actions: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    epsilon: float = 0.2,
    entropy_coef: float = 0.01
):
    """
    Computes the PPO policy loss (clipped surrogate objective) and entropy bonus.
    
    Returns:
        total_loss: scalar tensor
        policy_loss_val: float (for logging)
        entropy_val: float (for logging)
    """
    # 1. New Probabilities
    probs = policy_net(states)
    dist = Categorical(probs)
    new_log_probs = dist.log_prob(actions)
    
    # 2. Ratio
    ratio = torch.exp(new_log_probs - old_log_probs)
    
    # 3. Surrogate Objectives
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon) * advantages
    
    # 4. Loss (Max objective -> Min negative)
    policy_loss = -torch.min(surr1, surr2).mean()
    
    # 5. Entropy
    entropy = dist.entropy().mean()
    
    # Total
    total_loss = policy_loss - (entropy_coef * entropy)
    
    return total_loss, policy_loss.item(), entropy.item()


def compute_value_loss(
    value_net: nn.Module,
    states: torch.Tensor,
    returns: torch.Tensor
):
    """
    Computes the Value function loss (MSE between predicted value and actual returns).
    
    Returns:
        loss: scalar tensor
    """
    # Predict values
    values = value_net(states)
    
    # Ensure shapes match (batch, 1) vs (batch) or similar
    if values.dim() > 1 and returns.dim() == 1:
        values = values.squeeze(-1)
        
    loss = F.mse_loss(values, returns)
    return loss


def update(
    policy_net: nn.Module,
    value_net: nn.Module,
    policy_optimizer: torch.optim.Optimizer,
    value_optimizer: torch.optim.Optimizer,
    states: torch.Tensor,
    actions: torch.Tensor,
    old_log_probs: torch.Tensor,
    returns: torch.Tensor,
    advantages: torch.Tensor,
    epsilon: float = 0.2,
    entropy_coef: float = 0.01,
    max_grad_norm: float = 0.5
):
    """
    Performs one update step for both Policy and Value networks.
    
    Args:
        policy_net: Actor network
        value_net: Critic network
        policy_optimizer: Optimizer for actor
        value_optimizer: Optimizer for critic
        states: Batch of states
        actions: Batch of actions taken
        old_log_probs: Log probs from rollout
        returns: Calculated returns (G_t) for value target
        advantages: Calculated advantages (often normalized)
        
    Returns:
        dict of metrics (policy_loss, value_loss, entropy)
    """
    
    # --- Policy Update ---
    policy_optimizer.zero_grad()
    p_loss, p_loss_val, entropy_val = compute_policy_loss(
        policy_net, states, actions, old_log_probs, advantages, epsilon, entropy_coef
    )
    p_loss.backward()
    nn.utils.clip_grad_norm_(policy_net.parameters(), max_grad_norm)
    policy_optimizer.step()
    
    # --- Value Update ---
    value_optimizer.zero_grad()
    v_loss = compute_value_loss(value_net, states, returns)
    v_loss.backward()
    nn.utils.clip_grad_norm_(value_net.parameters(), max_grad_norm)
    value_optimizer.step()
    
    return {
        "policy_loss": p_loss_val,
        "value_loss": v_loss.item(),
        "entropy": entropy_val
    }
