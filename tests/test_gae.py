import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
import pytest


from algorithms.gae import (
    compute_gae,
    compute_gae_batch,
    normalize_advantages,
    compute_returns_monte_carlo,
    explain_gae
)


def test_gae_shape():
    """Test that GAE returns correct shapes."""
    T = 10
    rewards = torch.randn(T)
    values = torch.randn(T + 1)
    dones = torch.zeros(T)
    
    advantages, returns = compute_gae(rewards, values, dones)
    
    assert advantages.shape == (T,), f"Expected shape ({T},), got {advantages.shape}"
    assert returns.shape == (T,), f"Expected shape ({T},), got {returns.shape}"
    print("✓ Shape test passed")


def test_gae_with_terminal_state():
    """Test that GAE correctly handles terminal states (done=1)."""
    rewards = torch.tensor([1.0, 1.0, 1.0, 1.0])
    values = torch.tensor([0.5, 0.5, 0.5, 0.5, 0.5])
    dones = torch.tensor([0.0, 0.0, 1.0, 0.0])  # Episode ends at t=2
    
    advantages, returns = compute_gae(rewards, values, dones, gamma=0.9, gae_lambda=0.95)
    
    # At terminal state (t=2), next_value should be 0
    # So advantage at t=2 should be: r[2] + 0.9*0 - 0.5 = 0.5
    assert abs(advantages[2] - 0.5) < 0.01, f"Expected ~0.5, got {advantages[2]}"
    
    print("✓ Terminal state test passed")


def test_gae_simple_case():
    """Test GAE with a simple, manually verifiable case."""
    # Simple trajectory: constant rewards and values
    rewards = torch.tensor([1.0, 1.0, 1.0])
    values = torch.tensor([0.0, 0.0, 0.0, 0.0])  # Value function predicts 0
    dones = torch.tensor([0.0, 0.0, 1.0])  # Episode ends after 3 steps
    
    gamma = 1.0  # No discounting
    gae_lambda = 1.0  # Pure Monte Carlo
    
    advantages, returns = compute_gae(rewards, values, dones, gamma, gae_lambda)
    
    # With γ=1, λ=1, V=0, and terminal at t=2:
    # A[2] = 1 + 0 - 0 = 1
    # A[1] = 1 + 1*1*(1 - 0) - 0 = 2
    # A[0] = 1 + 1*1*(1 - 0)*2 - 0 = 3
    
    expected_advantages = torch.tensor([3.0, 2.0, 1.0])
    
    assert torch.allclose(advantages, expected_advantages, atol=1e-5), \
        f"Expected {expected_advantages}, got {advantages}"
    
    print("✓ Simple case test passed")


def test_gae_discounting():
    """Test that gamma properly discounts future rewards."""
    rewards = torch.ones(5)
    values = torch.zeros(6)
    dones = torch.zeros(5)
    dones[-1] = 1.0  # Terminal state at end
    
    gamma = 0.5  # Strong discounting
    gae_lambda = 1.0
    
    advantages, returns = compute_gae(rewards, values, dones, gamma, gae_lambda)
    
    # First advantage should be largest (gets all future rewards)
    # Last advantage should be smallest (only immediate reward)
    assert advantages[0] > advantages[-1], "Discounting not working correctly"
    assert advantages[0] > advantages[1], "Advantages should decrease"
    
    print("✓ Discounting test passed")


def test_gae_with_nonzero_values():
    """Test GAE with realistic value function estimates."""
    rewards = torch.tensor([1.0, 0.5, 0.0, 1.0])
    values = torch.tensor([0.8, 0.6, 0.4, 0.5, 0.0])
    dones = torch.tensor([0.0, 0.0, 0.0, 1.0])
    
    advantages, returns = compute_gae(rewards, values, dones, gamma=0.99, gae_lambda=0.95)
    
    # Returns should be advantages + values
    expected_returns = advantages + values[:-1]
    assert torch.allclose(returns, expected_returns, atol=1e-5)
    
    print("✓ Nonzero values test passed")


def test_gae_input_validation():
    """Test that GAE validates inputs correctly."""
    # Wrong values length
    with pytest.raises(ValueError, match="Values must have length T\\+1"):
        compute_gae(
            torch.ones(5),
            torch.ones(5),  # Should be length 6
            torch.zeros(5)
        )
    
    # Wrong dones length
    with pytest.raises(ValueError, match="same length as rewards"):
        compute_gae(
            torch.ones(5),
            torch.ones(6),
            torch.zeros(3)  # Should be length 5
        )
    
    # Invalid gamma
    with pytest.raises(ValueError, match="Gamma must be in"):
        compute_gae(
            torch.ones(5),
            torch.ones(6),
            torch.zeros(5),
            gamma=1.5
        )
    
    # Invalid lambda
    with pytest.raises(ValueError, match="GAE lambda must be in"):
        compute_gae(
            torch.ones(5),
            torch.ones(6),
            torch.zeros(5),
            gae_lambda=2.0
        )
    
    print("✓ Input validation test passed")


def test_gae_batch():
    """Test batched GAE computation."""
    batch_size = 4
    T = 10
    
    rewards = torch.randn(batch_size, T)
    values = torch.randn(batch_size, T + 1)
    dones = torch.zeros(batch_size, T)
    
    advantages, returns = compute_gae_batch(rewards, values, dones)
    
    assert advantages.shape == (batch_size, T)
    assert returns.shape == (batch_size, T)
    
    # Verify batch computation matches single computation
    for i in range(batch_size):
        adv_single, ret_single = compute_gae(
            rewards[i], values[i], dones[i]
        )
        assert torch.allclose(advantages[i], adv_single, atol=1e-5)
        assert torch.allclose(returns[i], ret_single, atol=1e-5)
    
    print("✓ Batch computation test passed")


def test_normalize_advantages():
    """Test advantage normalization."""
    advantages = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    
    normalized = normalize_advantages(advantages)
    
    # Should have mean ~0 and std ~1
    assert abs(normalized.mean().item()) < 1e-5
    assert abs(normalized.std().item() - 1.0) < 1e-5
    
    print("✓ Normalization test passed")


def test_monte_carlo_returns():
    """Test Monte Carlo return computation."""
    rewards = torch.tensor([1.0, 1.0, 1.0])
    dones = torch.tensor([0.0, 0.0, 1.0])
    gamma = 0.9
    
    returns = compute_returns_monte_carlo(rewards, dones, gamma)
    
    # Manual calculation:
    # returns[2] = 1.0
    # returns[1] = 1.0 + 0.9 * 1.0 = 1.9
    # returns[0] = 1.0 + 0.9 * 1.9 = 2.71
    
    expected = torch.tensor([2.71, 1.9, 1.0])
    assert torch.allclose(returns, expected, atol=1e-5)
    
    print("✓ Monte Carlo returns test passed")


def test_gae_vs_monte_carlo():
    """Test that GAE with λ=1 and V=0 matches Monte Carlo."""
    rewards = torch.tensor([1.0, 0.5, 0.8, 1.2])
    values = torch.zeros(5)  # Zero value function
    dones = torch.tensor([0.0, 0.0, 0.0, 1.0])
    gamma = 0.95
    
    # GAE with λ=1 should match Monte Carlo
    advantages_gae, returns_gae = compute_gae(
        rewards, values, dones, gamma=gamma, gae_lambda=1.0
    )
    
    returns_mc = compute_returns_monte_carlo(rewards, dones, gamma=gamma)
    
    # Since V=0, advantages = returns for GAE
    assert torch.allclose(advantages_gae, returns_mc, atol=1e-4)
    
    print("✓ GAE vs Monte Carlo test passed")


def test_gae_numerical_stability():
    """Test GAE with extreme values."""
    # Large rewards
    rewards = torch.ones(100) * 100
    values = torch.zeros(101)
    dones = torch.zeros(100)
    
    advantages, returns = compute_gae(rewards, values, dones)
    
    assert torch.isfinite(advantages).all(), "GAE produced non-finite values"
    assert torch.isfinite(returns).all(), "Returns produced non-finite values"
    
    print("✓ Numerical stability test passed")


def test_numpy_input():
    """Test that GAE works with numpy arrays."""
    rewards_np = np.array([1.0, 1.0, 1.0])
    values_np = np.array([0.5, 0.5, 0.5, 0.0])
    dones_np = np.array([0.0, 0.0, 1.0])
    
    advantages, returns = compute_gae(rewards_np, values_np, dones_np)
    
    assert isinstance(advantages, torch.Tensor)
    assert isinstance(returns, torch.Tensor)
    
    print("✓ NumPy input test passed")


def test_explain_gae():
    """Test the explain_gae utility function."""
    rewards = torch.tensor([1.0, 0.5, 1.0])
    values = torch.tensor([0.8, 0.6, 0.4, 0.0])
    dones = torch.tensor([0.0, 0.0, 1.0])
    
    print("\n" + "=" * 60)
    print("Testing explain_gae() output:")
    advantages, returns = explain_gae(rewards, values, dones)
    
    # Should match regular GAE output
    advantages_regular, returns_regular = compute_gae(rewards, values, dones)
    
    assert torch.allclose(advantages, advantages_regular)
    assert torch.allclose(returns, returns_regular)
    
    print("✓ explain_gae test passed")


def run_all_tests():
    """Run all GAE tests."""
    print("\n" + "=" * 60)
    print("RUNNING GAE TESTS")
    print("=" * 60 + "\n")
    
    tests = [
        test_gae_shape,
        test_gae_with_terminal_state,
        test_gae_simple_case,
        test_gae_discounting,
        test_gae_with_nonzero_values,
        test_gae_input_validation,
        test_gae_batch,
        test_normalize_advantages,
        test_monte_carlo_returns,
        test_gae_vs_monte_carlo,
        test_gae_numerical_stability,
        test_numpy_input,
        test_explain_gae,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ {test.__name__} FAILED: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)
    
    if failed == 0:
        print("✓✓✓ ALL GAE TESTS PASSED ✓✓✓")
    
    return failed == 0


if __name__ == "__main__":
    run_all_tests()