import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import torch.optim as optim
from algorithms.ppo import compute_policy_loss, compute_value_loss, update
from networks.policy import PolicyNetwork
from networks.value import ValueNetwork

def test_compute_policy_loss():
    print("\n=== Testing Compute Policy Loss ===")
    policy = PolicyNetwork()
    batch = 4
    states = torch.randn(batch, 40)
    actions = torch.randint(0, 5, (batch,))
    old_log_probs = torch.randn(batch)
    advantages = torch.randn(batch)
    
    loss, p_val, ent = compute_policy_loss(policy, states, actions, old_log_probs, advantages)
    
    print(f"Policy Loss (Tensor): {loss}")
    print(f"Policy Loss (Val): {p_val}")
    print(f"Entropy: {ent}")
    
    assert isinstance(p_val, float)
    assert isinstance(ent, float)
    assert loss.requires_grad
    print("✓ Policy loss passed")

def test_compute_value_loss():
    print("\n=== Testing Compute Value Loss ===")
    value_net = ValueNetwork()
    batch = 4
    states = torch.randn(batch, 40)
    returns = torch.randn(batch) # Returns are scalar targets
    
    loss = compute_value_loss(value_net, states, returns)
    
    print(f"Value Loss: {loss.item()}")
    
    assert loss.requires_grad
    # Just check if it's MSE-like (positive)
    assert loss.item() >= 0
    print("✓ Value loss passed")

def test_full_update():
    print("\n=== Testing Full Update Step ===")
    policy = PolicyNetwork()
    value_net = ValueNetwork()
    
    p_opt = optim.Adam(policy.parameters(), lr=1e-3)
    v_opt = optim.Adam(value_net.parameters(), lr=1e-3)
    
    batch = 4
    states = torch.randn(batch, 40)
    actions = torch.randint(0, 5, (batch,))
    old_log_probs = torch.randn(batch)
    advantages = torch.randn(batch)
    returns = torch.randn(batch)
    
    # Clone params to check update
    p_params_before = [p.clone() for p in policy.parameters()]
    v_params_before = [p.clone() for p in value_net.parameters()]
    
    metrics = update(
        policy, value_net, p_opt, v_opt,
        states, actions, old_log_probs, returns, advantages
    )
    
    print(f"Metrics: {metrics}")
    
    # Check if params changed
    p_changed = any(not torch.allclose(p1, p2) for p1, p2 in zip(p_params_before, policy.parameters()))
    v_changed = any(not torch.allclose(p1, p2) for p1, p2 in zip(v_params_before, value_net.parameters()))
    
    print(f"Policy Updated: {p_changed}")
    print(f"Value Updated: {v_changed}")
    
    assert p_changed, "Policy network did not update"
    assert v_changed, "Value network did not update"
    print("✓ Full update passed")

if __name__ == "__main__":
    test_compute_policy_loss()
    test_compute_value_loss()
    test_full_update()
    print("\n✓✓✓ ALL PPO REFACTOR TESTS PASSED ✓✓✓")
