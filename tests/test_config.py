"""Test configuration module."""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from training.config import Config, get_default_config, create_experiment_config


def test_default_config():
    """Test default configuration."""
    config = get_default_config()
    
    # Check key values from design
    assert config.state_dim == 40
    assert config.action_dim == 5
    assert config.buffer_size == 2048
    assert config.alpha == 0.3
    assert config.beta == 0.7
    assert config.gamma == 0.99
    assert config.gae_lambda == 0.95
    
    print("✓ Default config test passed")


def test_config_validation():
    """Test that validation catches errors."""
    try:
        # This should fail: alpha + beta != 1
        Config(alpha=0.5, beta=0.3)
        assert False, "Should have raised assertion error"
    except AssertionError as e:
        assert "alpha + beta" in str(e)
        print("✓ Config validation test passed")


def test_config_save_load():
    """Test saving and loading config."""
    import tempfile
    
    config = get_default_config()
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        config.save_yaml(f.name)
        
        # Load it back
        loaded = Config.from_yaml(f.name)
        
        assert loaded.gamma == config.gamma
        assert loaded.alpha == config.alpha
        assert loaded.buffer_size == config.buffer_size
    
    print("✓ Save/load test passed")


def test_experiment_config():
    """Test creating experiment configs."""
    config = create_experiment_config(
        name="test_exp",
        learning_rate=1e-3,
        gamma=0.95
    )
    
    assert config.learning_rate == 1e-3
    assert config.gamma == 0.95
    assert "test_exp" in config.checkpoint_dir
    
    print("✓ Experiment config test passed")


def test_config_repr():
    """Test config pretty printing."""
    config = get_default_config()
    repr_str = repr(config)
    
    assert "Training Configuration" in repr_str
    assert "gamma" in repr_str
    assert "alpha" in repr_str
    
    print("✓ Config repr test passed")


def run_all_tests():
    """Run all config tests."""
    print("\n" + "=" * 60)
    print("RUNNING CONFIG TESTS")
    print("=" * 60 + "\n")
    
    tests = [
        test_default_config,
        test_config_validation,
        test_config_save_load,
        test_experiment_config,
        test_config_repr,
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
        print("✓✓✓ ALL CONFIG TESTS PASSED ✓✓✓")
        
        # Print example config
        print("\n" + "=" * 60)
        print("EXAMPLE CONFIG OUTPUT:")
        print("=" * 60)
        config = get_default_config()
        print(config)
    
    return failed == 0


if __name__ == "__main__":
    run_all_tests()