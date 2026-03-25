import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
from data.replay_buffer import ReplayBuffer

def test_storage_and_retrieval():
    print("\n=== Testing Buffer Storage and Retrieval ===")
    buffer = ReplayBuffer(capacity=10)
    
    # Store some fake data
    for i in range(5):
        state = torch.randn(40)
        action = 1
        reward = 1.0
        value = 0.5
        log_prob = -0.1
        done = 0
        buffer.store(state, action, reward, value, log_prob, done)
        
    assert len(buffer) == 5
    
    states, actions, rewards, values, log_probs, dones = buffer.get_batch()
    
    print(f"States shape: {states.shape}")
    assert states.shape == (5, 40)
    assert actions.shape == (5,)
    assert rewards.shape == (5,)
    
    print("✓ Storage and retrieval passed")

def test_gae_computation():
    print("\n=== Testing GAE Computation Integration ===")
    buffer = ReplayBuffer(capacity=10)
    
    # Store data
    # Simple case: Reward=1 every step, Value=0.9, Gamma=0.99
    for i in range(3):
        buffer.store(torch.randn(40), 1, 1.0, 0.0, -0.1, 0)
        
    last_val = 0.0
    adv, returns = buffer.compute_advantages_and_returns(last_val, gamma=0.99, gae_lambda=0.95)
    
    print(f"Advantages: {adv}")
    print(f"Returns: {returns}")
    
    assert adv.shape == (3,)
    assert returns.shape == (3,)
    # Verify basic property: Advantages are normalized
    # ReplayBuffer.compute_advantages_and_returns normalizes advantages, so they should roughly have mean 0 and std 1
    assert torch.allclose(adv.mean(), torch.tensor(0.0), atol=1e-5)
    assert torch.allclose(adv.std(), torch.tensor(1.0), atol=1e-4)
    
    print("✓ GAE integration passed")

def test_clearing():
    print("\n=== Testing Buffer Clearing ===")
    buffer = ReplayBuffer(capacity=10)
    buffer.store(torch.randn(40), 1, 1.0, 0.5, -0.1, 0)
    assert len(buffer) == 1
    
    buffer.clear()
    assert len(buffer) == 0
    print("✓ Clearing passed")

if __name__ == "__main__":
    test_storage_and_retrieval()
    test_gae_computation()
    test_clearing()
    print("\n✓✓✓ ALL BUFFER TESTS PASSED ✓✓✓")
