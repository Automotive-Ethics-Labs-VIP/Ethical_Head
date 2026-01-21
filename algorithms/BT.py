import torch
import torch.nn.functional as F

def trajectory_reward(reward_model, states, actions):
    """
    Computes cumulative reward for a trajectory.
    
    Args:
        reward_model: RewardModel instance
        states: Tensor [T, 40] - trajectory states
        actions: Tensor [T] - trajectory actions
    
    Returns:
        Scalar tensor - sum of rewards over trajectory
    """
    rewards = reward_model(states, actions)  # [T]
    return rewards.sum()


def bradley_terry_loss(reward_model, traj_A, traj_B, preference):
    """
    Bradley-Terry preference loss for comparing two trajectories.
    
    The model learns: P(A > B) = sigmoid(R_A - R_B)
    
    Args:
        reward_model: RewardModel instance
        traj_A: Tuple of (states [T_A, 40], actions [T_A])
        traj_B: Tuple of (states [T_B, 40], actions [T_B])
        preference: Float in {0.0, 1.0}
            1.0 if A is preferred over B
            0.0 if B is preferred over A
    
    Returns:
        Scalar loss tensor
    """
    states_A, actions_A = traj_A
    states_B, actions_B = traj_B
    
    R_A = trajectory_reward(reward_model, states_A, actions_A)
    R_B = trajectory_reward(reward_model, states_B, actions_B)
    
    # logits = R_A - R_B
    # P(A > B) = sigmoid(logits)
    logits = R_A - R_B
    
    # Convert preference to tensor on same device
    label = torch.tensor(
        [preference], 
        dtype=torch.float32, 
        device=logits.device
    )
    
    # Binary cross-entropy: -[y*log(p) + (1-y)*log(1-p)]
    loss = F.binary_cross_entropy_with_logits(
        logits.unsqueeze(0), 
        label
    )
    
    return loss


def bradley_terry_batch_loss(reward_model, batch_traj_A, batch_traj_B, batch_preferences):
    """
    Batched version of Bradley-Terry loss for efficiency.
    
    Args:
        reward_model: RewardModel instance
        batch_traj_A: List of (states, actions) tuples
        batch_traj_B: List of (states, actions) tuples  
        batch_preferences: Tensor [batch_size] of 0s and 1s
    
    Returns:
        Mean loss over batch
    """
    losses = []
    for traj_A, traj_B, pref in zip(batch_traj_A, batch_traj_B, batch_preferences):
        loss = bradley_terry_loss(reward_model, traj_A, traj_B, pref)
        losses.append(loss)
    
    return torch.stack(losses).mean()