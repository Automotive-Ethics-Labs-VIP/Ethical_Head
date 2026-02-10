import torch
import numpy as np
from typing import Union, Tuple


def compute_gae(
    rewards: Union[torch.Tensor, np.ndarray],
    values: Union[torch.Tensor, np.ndarray],
    dones: Union[torch.Tensor, np.ndarray],
    gamma: float = 0.99,
    gae_lambda: float = 0.95
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Generalized Advantage Estimation (GAE).
    
    GAE measures how much better an action was compared to what the value
    function expected. It provides a variance-reduced estimate of the advantage
    function for policy gradient methods.
    
    Algorithm:
        δ_t = r_t + γ * V(s_{t+1}) * (1 - done_t) - V(s_t)
        A_t = δ_t + (γλ) * (1 - done_t) * A_{t+1}
        R_t = A_t + V(s_t)
    
    Args:
        rewards: Rewards received at each timestep [T]
        values: Value estimates at each timestep [T+1] 
                (includes value of final next state for bootstrapping)
        dones: Episode termination flags [T]. 1.0 if episode ended, 0.0 otherwise
        gamma: Discount factor for future rewards (0.99 is standard)
        gae_lambda: GAE smoothing parameter (0.95 is standard)
                   - λ=0: pure TD (high bias, low variance)
                   - λ=1: Monte Carlo (low bias, high variance)
    
    Returns:
        advantages: Advantage estimates [T]
        returns: Target values for value function [T]
    
    Example:
        >>> rewards = torch.tensor([1.0, 0.5, 0.0, 1.0])
        >>> values = torch.tensor([0.5, 0.6, 0.7, 0.8, 0.0])  # T+1 values
        >>> dones = torch.tensor([0.0, 0.0, 0.0, 1.0])
        >>> advantages, returns = compute_gae(rewards, values, dones)
    """
    # Convert to tensors if needed
    if isinstance(rewards, np.ndarray):
        rewards = torch.from_numpy(rewards).float()
    if isinstance(values, np.ndarray):
        values = torch.from_numpy(values).float()
    if isinstance(dones, np.ndarray):
        dones = torch.from_numpy(dones).float()
    
    # Validate inputs
    T = len(rewards)
    
    if len(values) != T + 1:
        raise ValueError(
            f"Values must have length T+1 (got {len(values)}), "
            f"where T={T} is the length of rewards. "
            f"The extra value is for bootstrapping the final state."
        )
    
    if len(dones) != T:
        raise ValueError(
            f"Dones must have same length as rewards. "
            f"Got dones={len(dones)}, rewards={len(rewards)}"
        )
    
    # Check for valid ranges
    if not ((dones == 0.0) | (dones == 1.0)).all():
        raise ValueError("Dones must be binary (0.0 or 1.0)")
    
    if not (0.0 <= gamma <= 1.0):
        raise ValueError(f"Gamma must be in [0, 1], got {gamma}")
    
    if not (0.0 <= gae_lambda <= 1.0):
        raise ValueError(f"GAE lambda must be in [0, 1], got {gae_lambda}")
    
    # Initialize advantage accumulator
    advantages = torch.zeros_like(rewards)
    last_gae = 0.0
    
    # Compute advantages backward through time
    for t in reversed(range(T)):
        # TD error: δ_t = r_t + γ * V(s_{t+1}) * (1 - done) - V(s_t)
        # When done=1, next_value should be 0 (no future rewards)
        next_value = values[t + 1] * (1.0 - dones[t])
        delta = rewards[t] + gamma * next_value - values[t]
        
        # GAE: A_t = δ_t + (γλ) * (1 - done) * A_{t+1}
        # When done=1, don't propagate advantage from future
        last_gae = delta + gamma * gae_lambda * (1.0 - dones[t]) * last_gae
        advantages[t] = last_gae
    
    # Returns are advantages + baseline
    # These are the targets for value function training
    returns = advantages + values[:-1]  # Don't include final bootstrap value
    
    return advantages, returns


def compute_gae_batch(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float = 0.99,
    gae_lambda: float = 0.95
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Vectorized GAE computation for multiple trajectories (more efficient).
    
    This is useful when you have multiple parallel environments.
    
    Args:
        rewards: [batch_size, T] rewards
        values: [batch_size, T+1] value estimates
        dones: [batch_size, T] termination flags
        gamma: Discount factor
        gae_lambda: GAE smoothing parameter
    
    Returns:
        advantages: [batch_size, T]
        returns: [batch_size, T]
    """
    batch_size, T = rewards.shape
    
    if values.shape != (batch_size, T + 1):
        raise ValueError(
            f"Values shape {values.shape} doesn't match expected "
            f"[{batch_size}, {T + 1}]"
        )
    
    if dones.shape != (batch_size, T):
        raise ValueError(
            f"Dones shape {dones.shape} doesn't match expected "
            f"[{batch_size}, {T}]"
        )
    
    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros(batch_size)
    
    for t in reversed(range(T)):
        next_values = values[:, t + 1] * (1.0 - dones[:, t])
        delta = rewards[:, t] + gamma * next_values - values[:, t]
        last_gae = delta + gamma * gae_lambda * (1.0 - dones[:, t]) * last_gae
        advantages[:, t] = last_gae
    
    returns = advantages + values[:, :-1]
    
    return advantages, returns


def normalize_advantages(advantages: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Normalize advantages to have mean 0 and std 1.
    
    This is a common trick to stabilize policy gradient training.
    
    Args:
        advantages: Advantage estimates (any shape)
        eps: Small constant for numerical stability
    
    Returns:
        Normalized advantages (same shape as input)
    """
    mean = advantages.mean()
    std = advantages.std()
    return (advantages - mean) / (std + eps)


def compute_returns_monte_carlo(
    rewards: Union[torch.Tensor, np.ndarray],
    dones: Union[torch.Tensor, np.ndarray],
    gamma: float = 0.99
) -> torch.Tensor:
    """
    Compute Monte Carlo returns (sum of discounted future rewards).
    
    This is a special case of GAE with λ=1.0 and no value function.
    Useful for testing or simple environments.
    
    Args:
        rewards: Rewards at each timestep [T]
        dones: Episode termination flags [T]
        gamma: Discount factor
    
    Returns:
        returns: Discounted return at each timestep [T]
    
    Example:
        >>> rewards = torch.tensor([1.0, 1.0, 1.0])
        >>> dones = torch.tensor([0.0, 0.0, 1.0])
        >>> returns = compute_returns_monte_carlo(rewards, dones, gamma=0.9)
        >>> # returns[0] = 1 + 0.9*1 + 0.9^2*1 = 2.71
    """
    if isinstance(rewards, np.ndarray):
        rewards = torch.from_numpy(rewards).float()
    if isinstance(dones, np.ndarray):
        dones = torch.from_numpy(dones).float()
    
    T = len(rewards)
    returns = torch.zeros_like(rewards)
    running_return = 0.0
    
    for t in reversed(range(T)):
        # If episode ended, reset return to 0
        if dones[t]:
            running_return = 0.0
        
        running_return = rewards[t] + gamma * running_return
        returns[t] = running_return
    
    return returns


# Utility function for debugging
def explain_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float = 0.99,
    gae_lambda: float = 0.95
):
    """
    Compute GAE with detailed step-by-step explanation.
    
    Useful for understanding and debugging GAE computation.
    
    Args:
        Same as compute_gae()
    
    Prints:
        Step-by-step breakdown of GAE computation
    """
    T = len(rewards)
    print("=" * 60)
    print("GAE Computation Breakdown")
    print("=" * 60)
    print(f"Gamma (γ): {gamma}")
    print(f"Lambda (λ): {gae_lambda}")
    print(f"Trajectory length: {T}")
    print()
    
    advantages = torch.zeros_like(rewards)
    last_gae = 0.0
    
    print(f"{'t':<3} {'r_t':<8} {'V(s_t)':<8} {'V(s_t+1)':<10} {'done':<6} {'δ_t':<8} {'A_t':<8}")
    print("-" * 60)
    
    for t in reversed(range(T)):
        next_value = values[t + 1] * (1.0 - dones[t])
        delta = rewards[t] + gamma * next_value - values[t]
        last_gae = delta + gamma * gae_lambda * (1.0 - dones[t]) * last_gae
        advantages[t] = last_gae
        
        print(f"{t:<3} {rewards[t].item():<8.3f} {values[t].item():<8.3f} "
              f"{next_value.item():<10.3f} {dones[t].item():<6.0f} "
              f"{delta.item():<8.3f} {last_gae:<8.3f}")
    
    returns = advantages + values[:-1]
    
    print()
    print("Summary:")
    print(f"  Mean advantage: {advantages.mean().item():.3f}")
    print(f"  Std advantage: {advantages.std().item():.3f}")
    print(f"  Mean return: {returns.mean().item():.3f}")
    print("=" * 60)
    
    return advantages, returns