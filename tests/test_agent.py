"""Test the inference agent with real checkpoints."""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
import tempfile

from inference.agent import EthicalAgent, quick_inference
from training.trainer import RLTrainer
from training.config import Config


def create_test_checkpoint():
    """Create a small trained checkpoint for testing."""
    print("Creating test checkpoint...")
    config = Config()
    config.buffer_size = 64
    config.total_timesteps = 128
    config.log_interval = 100  # Suppress logs
    
    trainer = RLTrainer(config)
    
    # Quick training
    trainer.train(total_timesteps=128)
    
    # Save checkpoint
    checkpoint_path = Path("checkpoints/test_agent_checkpoint.pt")
    trainer.save_checkpoint(checkpoint_path)
    
    print(f"✓ Test checkpoint created at {checkpoint_path}")
    return checkpoint_path


def test_agent_initialization():
    """Test agent loads checkpoint correctly."""
    print("\n--- Test: Agent Initialization ---")
    
    checkpoint_path = create_test_checkpoint()
    
    # Load agent
    agent = EthicalAgent(checkpoint_path)
    
    assert agent.policy is not None
    assert agent.value_net is None  # Not loaded by default
    assert agent.reward_model is None
    
    print("✓ Agent initialized successfully")
    return checkpoint_path


def test_agent_with_value_and_reward():
    """Test agent loading all networks."""
    print("\n--- Test: Load All Networks ---")
    
    checkpoint_path = Path("checkpoints/test_agent_checkpoint.pt")
    
    agent = EthicalAgent(
        checkpoint_path,
        load_value_network=True,
        load_reward_model=True
    )
    
    assert agent.policy is not None
    assert agent.value_net is not None
    assert agent.reward_model is not None
    
    print("✓ All networks loaded successfully")


def test_get_action():
    """Test basic action prediction."""
    print("\n--- Test: Get Action ---")
    
    checkpoint_path = Path("checkpoints/test_agent_checkpoint.pt")
    agent = EthicalAgent(checkpoint_path)
    
    # Test with numpy array
    state_np = np.random.randn(40)
    action = agent.get_action(state_np)
    
    assert isinstance(action, int)
    assert 0 <= action <= 4
    print(f"Action from numpy input: {action} ({agent.ACTION_NAMES[action]})")
    
    # Test with list
    state_list = state_np.tolist()
    action = agent.get_action(state_list)
    assert isinstance(action, int)
    print(f"Action from list input: {action} ({agent.ACTION_NAMES[action]})")
    
    # Test with tensor
    state_tensor = torch.FloatTensor(state_np)
    action = agent.get_action(state_tensor)
    assert isinstance(action, int)
    print(f"Action from tensor input: {action} ({agent.ACTION_NAMES[action]})")
    
    print("✓ get_action works with all input types")


def test_deterministic_vs_stochastic():
    """Test deterministic vs stochastic action selection."""
    print("\n--- Test: Deterministic vs Stochastic ---")
    
    checkpoint_path = Path("checkpoints/test_agent_checkpoint.pt")
    agent = EthicalAgent(checkpoint_path)
    
    state = np.random.randn(40)
    
    # Deterministic should be consistent
    action1 = agent.get_action(state, deterministic=True)
    action2 = agent.get_action(state, deterministic=True)
    assert action1 == action2, "Deterministic actions should be identical"
    print(f"Deterministic action: {action1} (consistent)")
    
    # Stochastic might vary (run multiple times to check)
    stochastic_actions = [agent.get_action(state, deterministic=False) for _ in range(10)]
    print(f"Stochastic actions: {stochastic_actions}")
    print(f"  (May vary due to sampling)")
    
    print("✓ Deterministic and stochastic modes work")


def test_get_action_probabilities():
    """Test probability distribution output."""
    print("\n--- Test: Action Probabilities ---")
    
    checkpoint_path = Path("checkpoints/test_agent_checkpoint.pt")
    agent = EthicalAgent(checkpoint_path)
    
    state = np.random.randn(40)
    probs = agent.get_action_probabilities(state)
    
    assert len(probs) == 5
    assert all(0 <= p <= 1 for p in probs.values())
    assert abs(sum(probs.values()) - 1.0) < 1e-5, "Probabilities should sum to 1"
    
    print("Action probabilities:")
    for action, prob in probs.items():
        print(f"  {action}: {prob:.2%}")
    
    print("✓ Probability distribution correct")


def test_get_action_with_confidence():
    """Test action with confidence score."""
    print("\n--- Test: Action with Confidence ---")
    
    checkpoint_path = Path("checkpoints/test_agent_checkpoint.pt")
    agent = EthicalAgent(checkpoint_path)
    
    state = np.random.randn(40)
    action, confidence, probs = agent.get_action_with_confidence(state)
    
    assert isinstance(action, int)
    assert 0 <= confidence <= 1
    assert len(probs) == 5
    
    print(f"Action: {agent.ACTION_NAMES[action]}")
    print(f"Confidence: {confidence:.2%}")
    print(f"All probabilities: {probs}")
    
    print("✓ Confidence scoring works")


def test_value_estimate():
    """Test value network predictions."""
    print("\n--- Test: Value Estimate ---")
    
    checkpoint_path = Path("checkpoints/test_agent_checkpoint.pt")
    agent = EthicalAgent(checkpoint_path, load_value_network=True)
    
    state = np.random.randn(40)
    value = agent.get_value_estimate(state)
    
    assert isinstance(value, float)
    print(f"Value estimate: {value:.4f}")
    
    print("✓ Value network predictions work")


def test_ethical_reward():
    """Test reward model scoring."""
    print("\n--- Test: Ethical Reward ---")
    
    checkpoint_path = Path("checkpoints/test_agent_checkpoint.pt")
    agent = EthicalAgent(checkpoint_path, load_reward_model=True)
    
    state = np.random.randn(40)
    
    # Test all actions
    for action in range(5):
        reward = agent.get_ethical_reward(state, action)
        assert isinstance(reward, float)
        print(f"  Action {action} ({agent.ACTION_NAMES[action]}): reward = {reward:.4f}")
    
    print("✓ Reward model scoring works")


def test_analyze_decision():
    """Test comprehensive decision analysis."""
    print("\n--- Test: Decision Analysis ---")
    
    checkpoint_path = Path("checkpoints/test_agent_checkpoint.pt")
    agent = EthicalAgent(
        checkpoint_path,
        load_value_network=True,
        load_reward_model=True
    )
    
    state = np.random.randn(40)
    analysis = agent.analyze_decision(state)
    
    assert 'action' in analysis
    assert 'action_name' in analysis
    assert 'confidence' in analysis
    assert 'action_probabilities' in analysis
    assert 'value_estimate' in analysis
    assert 'ethical_rewards' in analysis
    
    print("\nComplete Decision Analysis:")
    print(f"  Chosen action: {analysis['action_name']}")
    print(f"  Confidence: {analysis['confidence']:.2%}")
    print(f"  Value estimate: {analysis['value_estimate']:.4f}")
    print(f"  Ethical rewards:")
    for action_name, reward in analysis['ethical_rewards'].items():
        print(f"    {action_name}: {reward:.4f}")
    
    print("✓ Full decision analysis works")


def test_batch_predict():
    """Test batch prediction."""
    print("\n--- Test: Batch Prediction ---")
    
    checkpoint_path = Path("checkpoints/test_agent_checkpoint.pt")
    agent = EthicalAgent(checkpoint_path)
    
    # Batch of states
    batch_size = 10
    states = np.random.randn(batch_size, 40)
    
    actions = agent.batch_predict(states, deterministic=True)
    
    assert actions.shape == (batch_size,)
    assert all(0 <= a <= 4 for a in actions)
    
    print(f"Predicted {batch_size} actions: {actions}")
    print("✓ Batch prediction works")


def test_quick_inference():
    """Test convenience function."""
    print("\n--- Test: Quick Inference ---")
    
    checkpoint_path = Path("checkpoints/test_agent_checkpoint.pt")
    state = np.random.randn(40)
    
    action = quick_inference(str(checkpoint_path), state)
    
    assert isinstance(action, int)
    assert 0 <= action <= 4
    
    print(f"Quick inference action: {action}")
    print("✓ Quick inference works")


def test_model_info():
    """Test model information retrieval."""
    print("\n--- Test: Model Info ---")
    
    checkpoint_path = Path("checkpoints/test_agent_checkpoint.pt")
    agent = EthicalAgent(
        checkpoint_path,
        load_value_network=True,
        load_reward_model=True
    )
    
    info = agent.get_model_info()
    
    print("\nModel Information:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    assert 'policy_parameters' in info
    assert info['policy_parameters'] > 0
    
    print("✓ Model info retrieval works")


def test_integration_with_carla_format():
    """Test agent works with CARLA-like state format."""
    print("\n--- Test: CARLA Integration Format ---")
    
    checkpoint_path = Path("checkpoints/test_agent_checkpoint.pt")
    agent = EthicalAgent(checkpoint_path)
    
    # Simulate CARLA providing state as list
    carla_state = {
        'velocity_ego': 25.0,
        'num_passengers': 2,
        'lane_position': 0.0,
        'velocity_delta': 5.0,
        'num_ped_if_straight': 0,
        'num_ped_if_left': 1,
        'num_ped_if_right': 0,
        # ... obstacle features (33 more values)
    }
    
    # Convert to 40-dim vector (this is what CARLA team will do)
    state_vector = [carla_state['velocity_ego'], carla_state['num_passengers']] + [0] * 38
    
    action = agent.get_action(state_vector)
    
    print(f"CARLA state -> Action: {agent.ACTION_NAMES[action]}")
    print("✓ CARLA integration format works")


def run_all_tests():
    """Run all agent tests."""
    print("\n" + "=" * 60)
    print("RUNNING AGENT TESTS")
    print("=" * 60)
    
    tests = [
        test_agent_initialization,
        test_agent_with_value_and_reward,
        test_get_action,
        test_deterministic_vs_stochastic,
        test_get_action_probabilities,
        test_get_action_with_confidence,
        test_value_estimate,
        test_ethical_reward,
        test_analyze_decision,
        test_batch_predict,
        test_quick_inference,
        test_model_info,
        test_integration_with_carla_format,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ {test.__name__} FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)
    
    if failed == 0:
        print("✓✓✓ ALL AGENT TESTS PASSED ✓✓✓")
        print("\n🎉 DEPLOYMENT INTERFACE READY! 🎉")
    
    return failed == 0


if __name__ == "__main__":
    run_all_tests()