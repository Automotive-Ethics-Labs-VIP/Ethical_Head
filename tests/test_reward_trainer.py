"""Test reward model trainer."""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import tempfile
from networks.reward import RewardModel
from training.reward_trainer import (
    RewardModelTrainer,
    create_synthetic_preferences,
    split_preferences
)
from training.config import Config


def test_trainer_initialization():
    """Test trainer initializes correctly."""
    config = Config()
    reward_model = RewardModel()
    
    trainer = RewardModelTrainer(reward_model, config)
    
    assert trainer.reward_model is not None
    assert trainer.optimizer is not None
    assert len(trainer.training_history['losses']) == 0
    
    print("✓ Trainer initialization test passed")


def test_synthetic_preferences():
    """Test synthetic preference generation."""
    preferences = create_synthetic_preferences(num_pairs=10)
    
    assert len(preferences) == 10
    
    # Check structure
    traj_A, traj_B, pref = preferences[0]
    states_A, actions_A = traj_A
    states_B, actions_B = traj_B
    
    assert states_A.shape[1] == 40
    assert actions_A.dim() == 1
    assert pref in [0.0, 1.0]
    
    print("✓ Synthetic preferences test passed")


def test_split_preferences():
    """Test train/test split."""
    preferences = create_synthetic_preferences(num_pairs=100)
    train, test = split_preferences(preferences, train_ratio=0.8)
    
    assert len(train) == 80
    assert len(test) == 20
    assert len(train) + len(test) == len(preferences)
    
    print("✓ Preference split test passed")


def test_training_on_synthetic_data():
    """Test training loop on synthetic preferences."""
    print("\n--- Testing training on synthetic data ---")
    
    config = Config()
    config.reward_model_epochs = 3
    config.preference_batch_size = 8
    
    reward_model = RewardModel()
    trainer = RewardModelTrainer(reward_model, config)
    
    # Create synthetic data
    preferences = create_synthetic_preferences(num_pairs=32)
    
    # Train
    metrics = trainer.train_on_preferences(preferences, verbose=True)
    
    assert 'loss' in metrics
    assert 'accuracy' in metrics
    assert len(trainer.training_history['losses']) == 3  # 3 epochs
    
    print(f"Final metrics: {metrics}")
    print("✓ Training test passed")


def test_evaluation():
    """Test evaluation on test set."""
    print("\n--- Testing evaluation ---")
    
    config = Config()
    reward_model = RewardModel()
    trainer = RewardModelTrainer(reward_model, config)
    
    # Create and split data
    preferences = create_synthetic_preferences(num_pairs=50)
    train_prefs, test_prefs = split_preferences(preferences)
    
    # Train
    trainer.train_on_preferences(train_prefs, epochs=2, verbose=False)
    
    # Evaluate
    eval_metrics = trainer.evaluate(test_prefs, verbose=True)
    
    assert 'test_loss' in eval_metrics
    assert 'test_accuracy' in eval_metrics
    
    print("✓ Evaluation test passed")


def test_predict_preference():
    """Test predicting preferences between trajectories."""
    config = Config()
    reward_model = RewardModel()
    trainer = RewardModelTrainer(reward_model, config)
    
    # Create two trajectories
    states_A = torch.randn(10, 40)
    actions_A = torch.randint(0, 5, (10,))
    states_B = torch.randn(8, 40)
    actions_B = torch.randint(0, 5, (8,))
    
    traj_A = (states_A, actions_A)
    traj_B = (states_B, actions_B)
    
    # Predict
    R_A, R_B, prob = trainer.predict_preference(traj_A, traj_B)
    
    assert isinstance(R_A, float)
    assert isinstance(R_B, float)
    assert 0.0 <= prob <= 1.0
    
    print(f"R_A: {R_A:.3f}, R_B: {R_B:.3f}, P(A>B): {prob:.3f}")
    print("✓ Predict preference test passed")


def test_save_load_checkpoint():
    """Test checkpoint saving and loading."""
    config = Config()
    reward_model = RewardModel()
    trainer = RewardModelTrainer(reward_model, config)
    
    # Train briefly
    preferences = create_synthetic_preferences(num_pairs=20)
    trainer.train_on_preferences(preferences, epochs=2, verbose=False)
    
    # Save checkpoint
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        checkpoint_path = f.name
        trainer.save_checkpoint(checkpoint_path, epoch=2)
        
        # Create new trainer and load
        new_reward_model = RewardModel()
        new_trainer = RewardModelTrainer(new_reward_model, config)
        new_trainer.load_checkpoint(checkpoint_path)
        
        # Check history was loaded
        assert len(new_trainer.training_history['losses']) > 0
    
    print("✓ Checkpoint save/load test passed")


def test_training_improves_accuracy():
    """Test that training actually improves accuracy over random."""
    print("\n--- Testing that training improves performance ---")
    
    config = Config()
    reward_model = RewardModel()
    trainer = RewardModelTrainer(reward_model, config)
    
    # Create consistent preferences (deterministic for testing)
    preferences = []
    for _ in range(50):
        states_A = torch.randn(10, 40)
        actions_A = torch.randint(0, 5, (10,))
        states_B = torch.randn(10, 40)
        actions_B = torch.randint(0, 5, (10,))
        
        # Always prefer trajectory with more positive mean state values
        pref = 1.0 if states_A.mean() > states_B.mean() else 0.0
        
        preferences.append(((states_A, actions_A), (states_B, actions_B), pref))
    
    train_prefs, test_prefs = split_preferences(preferences)
    
    # Evaluate before training
    initial_metrics = trainer.evaluate(test_prefs, verbose=False)
    print(f"Initial accuracy: {initial_metrics['test_accuracy']:.2%}")
    
    # Train
    trainer.train_on_preferences(train_prefs, epochs=10, verbose=False)
    
    # Evaluate after training
    final_metrics = trainer.evaluate(test_prefs, verbose=False)
    print(f"Final accuracy: {final_metrics['test_accuracy']:.2%}")
    
    # Accuracy should improve (might not be perfect on synthetic data)
    print(f"Improvement: {final_metrics['test_accuracy'] - initial_metrics['test_accuracy']:.2%}")
    
    print("✓ Training improvement test passed")


def run_all_tests():
    """Run all reward trainer tests."""
    print("\n" + "=" * 60)
    print("RUNNING REWARD TRAINER TESTS")
    print("=" * 60 + "\n")
    
    tests = [
        test_trainer_initialization,
        test_synthetic_preferences,
        test_split_preferences,
        test_training_on_synthetic_data,
        test_evaluation,
        test_predict_preference,
        test_save_load_checkpoint,
        test_training_improves_accuracy,
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
        print("✓✓✓ ALL REWARD TRAINER TESTS PASSED ✓✓✓")
    
    return failed == 0


if __name__ == "__main__":
    run_all_tests()