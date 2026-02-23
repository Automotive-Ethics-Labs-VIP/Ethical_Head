"""Test main RL trainer."""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import tempfile
from training.trainer import RLTrainer
from training.config import Config
from training.reward_trainer import create_synthetic_preferences


def test_trainer_initialization():
    """Test trainer initializes correctly."""
    config = Config()
    config.total_timesteps = 1000  # Small for testing
    
    trainer = RLTrainer(config)
    
    assert trainer.policy_net is not None
    assert trainer.value_net is not None
    assert trainer.reward_model is not None
    assert trainer.buffer is not None
    assert trainer.total_timesteps == 0
    assert trainer.num_updates == 0
    
    print("✓ Trainer initialization test passed")


def test_collect_rollout():
    """Test rollout collection."""
    print("\n--- Testing rollout collection ---")
    
    config = Config()
    trainer = RLTrainer(config)
    
    # Collect small rollout
    stats = trainer.collect_rollout(n_steps=100)
    
    assert stats['steps_collected'] == 100
    assert 'episodes_completed' in stats
    assert 'mean_episode_reward' in stats
    
    print(f"Collected {stats['steps_collected']} steps")
    print(f"Episodes completed: {stats['episodes_completed']}")
    print(f"Mean reward: {stats['mean_episode_reward']:.2f}")
    print("✓ Rollout collection test passed")


def test_combined_rewards():
    """Test combined reward computation."""
    config = Config()
    trainer = RLTrainer(config)
    
    # Create fake data
    states = torch.randn(10, 40)
    actions = torch.randint(0, 5, (10,))
    base_rewards = torch.randn(10)
    
    # Compute combined rewards
    combined = trainer.compute_combined_rewards(states, actions, base_rewards)
    
    assert combined.shape == (10,)
    assert torch.isfinite(combined).all()
    
    print(f"Base rewards mean: {base_rewards.mean():.3f}")
    print(f"Combined rewards mean: {combined.mean():.3f}")
    print("✓ Combined rewards test passed")


def test_policy_update():
    """Test policy update mechanism."""
    print("\n--- Testing policy update ---")
    
    config = Config()
    config.buffer_size = 64  # Small buffer for testing
    trainer = RLTrainer(config)
    
    # Collect small rollout
    trainer.collect_rollout(n_steps=64)
    
    # Update policy
    metrics = trainer.update_policy()
    
    assert 'policy_loss' in metrics
    assert 'value_loss' in metrics
    assert 'entropy' in metrics
    assert trainer.num_updates == 1
    
    print(f"Policy loss: {metrics['policy_loss']:.4f}")
    print(f"Value loss: {metrics['value_loss']:.4f}")
    print(f"Entropy: {metrics['entropy']:.4f}")
    print("✓ Policy update test passed")


def test_training_loop():
    """Test main training loop."""
    print("\n--- Testing training loop ---")
    
    config = Config()
    config.buffer_size = 64
    config.total_timesteps = 256  # Very short training
    config.log_interval = 1
    config.save_interval = 100
    
    trainer = RLTrainer(config)
    
    # Run short training
    trainer.train(total_timesteps=256)
    
    assert trainer.total_timesteps >= 256
    assert trainer.num_updates > 0
    assert len(trainer.training_history['policy_loss']) > 0
    
    print(f"Total timesteps: {trainer.total_timesteps}")
    print(f"Total updates: {trainer.num_updates}")
    print("✓ Training loop test passed")


def test_training_with_preferences():
    """Test training with reward model updates."""
    print("\n--- Testing training with preference updates ---")
    
    config = Config()
    config.buffer_size = 64
    config.total_timesteps = 256
    config.reward_model_update_freq = 2  # Update often for testing
    config.log_interval = 5
    
    trainer = RLTrainer(config)
    
    # Create synthetic preferences
    preferences = create_synthetic_preferences(num_pairs=20)
    
    # Run training with preferences
    trainer.train(total_timesteps=256, preference_dataset=preferences)
    
    assert trainer.total_timesteps >= 256
    print("✓ Training with preferences test passed")


def test_checkpoint_save_load():
    """Test checkpoint saving and loading."""
    print("\n--- Testing checkpoint save/load ---")
    
    config = Config()
    config.buffer_size = 64
    trainer = RLTrainer(config)
    
    # Collect some experience
    trainer.collect_rollout(n_steps=64)
    trainer.update_policy()
    
    initial_timesteps = trainer.total_timesteps
    initial_updates = trainer.num_updates
    
    # Save checkpoint
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        checkpoint_path = Path(f.name)
        trainer.save_checkpoint(checkpoint_path)
        
        # Create new trainer and load
        new_trainer = RLTrainer(config)
        new_trainer.load_checkpoint(checkpoint_path)
        
        assert new_trainer.total_timesteps == initial_timesteps
        assert new_trainer.num_updates == initial_updates
    
    print("✓ Checkpoint save/load test passed")


def test_mock_environment():
    """Test that mock environment works correctly."""
    config = Config()
    trainer = RLTrainer(config)
    
    # Test initial state
    state = trainer._get_initial_state()
    assert state.shape == (40,)
    
    # Test step
    action = 2
    next_state, reward, done, info = trainer._step_environment(state, action)
    
    assert next_state.shape == (40,)
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert isinstance(info, dict)
    
    print("✓ Mock environment test passed")


def test_integration_full_pipeline():
    """Integration test: full training pipeline."""
    print("\n--- Integration Test: Full Pipeline ---")
    
    config = Config()
    config.buffer_size = 128
    config.total_timesteps = 512
    config.log_interval = 2
    config.save_interval = 100
    config.reward_model_update_freq = 3
    
    trainer = RLTrainer(config)
    
    # Create preferences
    preferences = create_synthetic_preferences(num_pairs=30)
    
    # Track metrics
    initial_policy_params = list(trainer.policy_net.parameters())[0].clone()
    
    # Run training
    print("\nRunning training...")
    trainer.train(total_timesteps=512, preference_dataset=preferences)
    
    # Verify training occurred
    final_policy_params = list(trainer.policy_net.parameters())[0]
    params_changed = not torch.allclose(initial_policy_params, final_policy_params)
    
    assert params_changed, "Policy parameters should have changed"
    assert trainer.num_updates > 0
    assert len(trainer.training_history['policy_loss']) > 0
    
    print("\n✓ Full pipeline integration test passed")
    print(f"  Updates: {trainer.num_updates}")
    print(f"  Timesteps: {trainer.total_timesteps}")
    print(f"  History length: {len(trainer.training_history['policy_loss'])}")


def run_all_tests():
    """Run all trainer tests."""
    print("\n" + "=" * 60)
    print("RUNNING MAIN TRAINER TESTS")
    print("=" * 60 + "\n")
    
    tests = [
        test_trainer_initialization,
        test_collect_rollout,
        test_combined_rewards,
        test_policy_update,
        test_training_loop,
        test_training_with_preferences,
        test_checkpoint_save_load,
        test_mock_environment,
        test_integration_full_pipeline,
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
        print("✓✓✓ ALL TRAINER TESTS PASSED ✓✓✓")
        print("\n🎉 COMPLETE RL TRAINING PIPELINE WORKING! 🎉")
    
    return failed == 0


if __name__ == "__main__":
    run_all_tests()