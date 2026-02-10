import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from networks.value import ValueNetwork
from networks.policy import PolicyNetwork
from algorithms.gae import compute_gae

def test_gae_with_networks():
    """Verify GAE integrates with existing networks."""
    
    # Initialize networks
    value_net = ValueNetwork()
    policy = PolicyNetwork()
    
    # Simulate rollout
    T = 20
    states = torch.randn(T, 40)
    
    # Get values from network - FIX: squeeze to get scalars
    values = []
    for state in states:
        value = value_net(state).squeeze()  # ◄── Add .squeeze()
        values.append(value)
    values.append(torch.tensor(0.0))  # Bootstrap (scalar, not [0.0])
    values = torch.stack(values)
    
    # Simulate rewards and dones
    rewards = torch.randn(T)
    dones = torch.zeros(T)
    
    # Compute GAE
    advantages, returns = compute_gae(rewards, values, dones)
    
    assert advantages.shape == (T,), f"Expected ({T},), got {advantages.shape}"
    assert returns.shape == (T,), f"Expected ({T},), got {returns.shape}"
    print("✓ GAE successfully integrates with networks!")
    print(f"  Advantages: {advantages.shape}")
    print(f"  Returns: {returns.shape}")

if __name__ == "__main__":
    test_gae_with_networks()
