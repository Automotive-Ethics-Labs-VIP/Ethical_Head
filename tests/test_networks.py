import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn.functional as F

from networks.policy import PolicyNetwork
from networks.value import ValueNetwork
from networks.reward import RewardModel
from algorithms.BT import trajectory_reward, bradley_terry_loss


def test_basic_networks():
    """Test policy and value networks (your original tests)"""
    print("=" * 60)
    print("=== Testing Policy and Value Networks ===")
    print("=" * 60)
    
    policy = PolicyNetwork()
    value_net = ValueNetwork()

    print(f"Policy params: {sum(p.numel() for p in policy.parameters()):,}")
    print(f"Value params:  {sum(p.numel() for p in value_net.parameters()):,}")

    # Single test state
    state = torch.randn(40)

    print("\n--- Forward pass test ---")

    # Policy forward
    probs = policy(state)
    action, logprob = policy.sample_action(state)

    print(f"Policy probs shape: {probs.shape}")
    print(f"Policy probs: {probs}")
    print(f"Sum probs: {probs.sum().item():.6f}")
    print(f"Sampled action: {action}")
    print(f"Log prob: {logprob.item():.4f}")

    # Value forward
    value = value_net(state)
    print(f"Value estimate: {value.item():.4f}")

    print("\n--- Backward pass (smoke test) ---")

    optimizer = torch.optim.Adam(
        list(policy.parameters()) + list(value_net.parameters()),
        lr=1e-3
    )

    # Fake reward
    reward = torch.tensor([1.0])  # Make it [1] to match value shape
    advantage = reward - value.detach()

    policy_loss = -logprob * advantage
    value_loss = F.mse_loss(value, reward)
    loss = policy_loss + value_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Loss: {loss.item():.4f}")
    print("✓ Policy and Value networks passed")


def test_reward_model():
    """Test reward model with single and batch inputs"""
    print("\n" + "=" * 60)
    print("=== Testing Reward Model ===")
    print("=" * 60)
    
    reward_model = RewardModel()
    print(f"Reward model params: {sum(p.numel() for p in reward_model.parameters()):,}")
    
    print("\n--- Single state-action pair ---")
    state = torch.randn(1, 40)  # Batch size 1
    action = torch.tensor([2])  # Action index (swerve_left)
    
    reward = reward_model(state, action)
    print(f"State shape: {state.shape}")
    print(f"Action: {action.item()}")
    print(f"Reward: {reward.item():.4f}")
    
    print("\n--- Batch of state-action pairs ---")
    batch_size = 8
    states = torch.randn(batch_size, 40)
    actions = torch.randint(0, 5, (batch_size,))
    
    rewards = reward_model(states, actions)
    print(f"Batch size: {batch_size}")
    print(f"Rewards shape: {rewards.shape}")
    print(f"Rewards: {rewards}")
    print(f"Mean reward: {rewards.mean().item():.4f}")
    
    print("\n--- Gradient flow test ---")
    optimizer = torch.optim.Adam(reward_model.parameters(), lr=1e-3)
    
    # Fake target rewards
    target_rewards = torch.ones(batch_size)
    loss = F.mse_loss(rewards, target_rewards)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f"Loss: {loss.item():.4f}")
    print("✓ Reward model passed")


def test_trajectory_reward():
    """Test trajectory reward computation"""
    print("\n" + "=" * 60)
    print("=== Testing Trajectory Reward ===")
    print("=" * 60)
    
    reward_model = RewardModel()
    
    # Create a fake trajectory
    traj_length = 10
    states = torch.randn(traj_length, 40)
    actions = torch.randint(0, 5, (traj_length,))
    
    total_reward = trajectory_reward(reward_model, states, actions)
    
    print(f"Trajectory length: {traj_length}")
    print(f"Total reward: {total_reward.item():.4f}")
    
    # Verify it's the sum of individual rewards
    individual_rewards = reward_model(states, actions)
    expected_sum = individual_rewards.sum()
    
    print(f"Individual rewards sum: {expected_sum.item():.4f}")
    print(f"Match: {torch.allclose(total_reward, expected_sum)}")
    
    print("✓ Trajectory reward passed")


def test_bradley_terry_loss():
    """Test Bradley-Terry preference loss"""
    print("\n" + "=" * 60)
    print("=== Testing Bradley-Terry Loss ===")
    print("=" * 60)
    
    reward_model = RewardModel()
    
    # Create two trajectories of different lengths
    traj_A_length = 8
    traj_B_length = 12
    
    states_A = torch.randn(traj_A_length, 40)
    actions_A = torch.randint(0, 5, (traj_A_length,))
    
    states_B = torch.randn(traj_B_length, 40)
    actions_B = torch.randint(0, 5, (traj_B_length,))
    
    traj_A = (states_A, actions_A)
    traj_B = (states_B, actions_B)
    
    # Test preference for A
    print("\n--- Preference: A > B ---")
    preference = 1.0
    loss_A_preferred = bradley_terry_loss(reward_model, traj_A, traj_B, preference)
    print(f"Loss when A preferred: {loss_A_preferred.item():.4f}")
    
    # Test preference for B
    print("\n--- Preference: B > A ---")
    preference = 0.0
    loss_B_preferred = bradley_terry_loss(reward_model, traj_A, traj_B, preference)
    print(f"Loss when B preferred: {loss_B_preferred.item():.4f}")
    
    # Test gradient flow
    print("\n--- Gradient flow test ---")
    optimizer = torch.optim.Adam(reward_model.parameters(), lr=1e-3)
    
    preference = 1.0
    loss = bradley_terry_loss(reward_model, traj_A, traj_B, preference)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f"Loss before step: {loss.item():.4f}")
    
    # Compute loss again after update
    with torch.no_grad():
        loss_after = bradley_terry_loss(reward_model, traj_A, traj_B, preference)
    print(f"Loss after step: {loss_after.item():.4f}")
    
    print("✓ Bradley-Terry loss passed")


def test_reward_model_training():
    """Test a mini training loop for the reward model"""
    print("\n" + "=" * 60)
    print("=== Testing Reward Model Training Loop ===")
    print("=" * 60)
    
    reward_model = RewardModel()
    optimizer = torch.optim.Adam(reward_model.parameters(), lr=1e-3)
    
    # Create synthetic preference dataset
    num_pairs = 5
    
    print(f"\nTraining on {num_pairs} preference pairs...")
    
    for epoch in range(3):
        total_loss = 0.0
        
        for i in range(num_pairs):
            # Generate random trajectories
            len_A = torch.randint(5, 15, (1,)).item()
            len_B = torch.randint(5, 15, (1,)).item()
            
            states_A = torch.randn(len_A, 40)
            actions_A = torch.randint(0, 5, (len_A,))
            
            states_B = torch.randn(len_B, 40)
            actions_B = torch.randint(0, 5, (len_B,))
            
            # Random preference
            preference = torch.randint(0, 2, (1,)).float().item()
            
            traj_A = (states_A, actions_A)
            traj_B = (states_B, actions_B)
            
            # Compute loss and update
            optimizer.zero_grad()
            loss = bradley_terry_loss(reward_model, traj_A, traj_B, preference)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / num_pairs
        print(f"Epoch {epoch+1}/3 | Avg Loss: {avg_loss:.4f}")
    
    print("✓ Training loop passed")


def test_all_together():
    """Integration test: use all networks together"""
    print("\n" + "=" * 60)
    print("=== Integration Test: All Networks Together ===")
    print("=" * 60)
    
    # Initialize all networks
    policy = PolicyNetwork()
    value_net = ValueNetwork()
    reward_model = RewardModel()
    
    # Create a mini rollout
    num_steps = 5
    state = torch.randn(40)
    
    print(f"\nSimulating {num_steps}-step rollout...")
    
    states_list = []
    actions_list = []
    values_list = []
    rewards_list = []
    
    for step in range(num_steps):
        # Policy chooses action
        action, log_prob = policy.sample_action(state)
        
        # Value network estimates value
        value = value_net(state)
        
        # Reward model scores the (state, action) pair
        state_batch = state.unsqueeze(0)
        # Convert action to tensor if it's an int
        if isinstance(action, int):
            action_tensor = torch.tensor([action])
        else:
            action_tensor = action.unsqueeze(0) if action.dim() == 0 else action
        learned_reward = reward_model(state_batch, action_tensor)
        
        print(f"Step {step}: action={action if isinstance(action, int) else action.item()}, "
              f"value={value.item():.3f}, "
              f"learned_reward={learned_reward.item():.3f}")
        
        states_list.append(state_batch)
        actions_list.append(action_tensor)
        values_list.append(value)
        rewards_list.append(learned_reward)
        
        # Simulate next state (random walk)
        state = state + torch.randn(40) * 0.1
    
    # Compute trajectory reward
    all_states = torch.cat(states_list, dim=0)
    all_actions = torch.cat(actions_list, dim=0).squeeze()
    
    traj_reward = trajectory_reward(reward_model, all_states, all_actions)
    print(f"\nTotal trajectory reward: {traj_reward.item():.4f}")
    
    print("✓ Integration test passed")


def main():
    print("\n" + "=" * 60)
    print("RUNNING ALL NETWORK TESTS")
    print("=" * 60)
    
    try:
        test_basic_networks()
        test_reward_model()
        test_trajectory_reward()
        test_bradley_terry_loss()
        test_reward_model_training()
        test_all_together()
        
        print("\n" + "=" * 60)
        print("✓✓✓ ALL TESTS PASSED ✓✓✓")
        print("=" * 60)
        
    except Exception as e:
        print("\n" + "=" * 60)
        print(f"✗✗✗ TEST FAILED ✗✗✗")
        print("=" * 60)
        print(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()