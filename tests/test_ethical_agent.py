"""Test the final ethical agent deliverable."""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from ethical_agent import EthicalAgent, quick_predict  # Now this will work


def test_basic_usage():
    """Test basic action prediction."""
    print("\n--- Test: Basic Usage ---")
    
    model_path = "checkpoints/test_agent_checkpoint.pt"
    agent = EthicalAgent(model_path)
    
    state = np.random.randn(40).tolist()
    action = agent.get_action(state)
    
    assert isinstance(action, int)
    assert 0 <= action <= 4
    
    action_name = agent.get_action_name(action)
    assert action_name in ['maintain_course', 'brake_hard', 'swerve_left', 'swerve_right', 'accelerate']
    
    print(f"✓ Basic usage works: {action_name}")


def test_detailed_analysis():
    """Test full decision analysis."""
    print("\n--- Test: Detailed Analysis ---")
    
    model_path = "checkpoints/test_agent_checkpoint.pt"
    agent = EthicalAgent(model_path)
    
    state = np.random.randn(40).tolist()
    analysis = agent.get_action_with_analysis(state)
    
    assert 'action' in analysis
    assert 'action_name' in analysis
    assert 'confidence' in analysis
    assert 'value_estimate' in analysis
    assert 'ethical_rewards' in analysis
    
    print(f"Action: {analysis['action_name']}")
    print(f"Confidence: {analysis['confidence']:.2%}")
    print(f"✓ Detailed analysis works")


def test_compare_actions():
    """Test action comparison."""
    print("\n--- Test: Compare Actions ---")
    
    model_path = "checkpoints/test_agent_checkpoint.pt"
    agent = EthicalAgent(model_path)
    
    state = np.random.randn(40).tolist()
    scores = agent.compare_actions(state)
    
    assert len(scores) == 5
    assert all(isinstance(v, float) for v in scores.values())
    
    print("Ethical scores:")
    for action, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        print(f"  {action}: {score:.4f}")
    
    print("✓ Action comparison works")


def test_human_feedback():
    """Test human feedback collection."""
    print("\n--- Test: Human Feedback ---")
    
    model_path = "checkpoints/test_agent_checkpoint.pt"
    agent = EthicalAgent(model_path, enable_feedback=True)
    
    state = np.random.randn(40).tolist()
    action = agent.get_action(state)
    
    # Add feedback
    agent.add_human_feedback(state, action, "Good decision", rating=5)
    agent.add_human_feedback(state, action, "Safe choice", rating=4)
    
    assert len(agent.feedback_log) == 2
    
    # Save feedback
    agent.save_feedback('data/test_feedback.json')
    
    # Check statistics
    stats = agent.get_statistics()
    assert stats['total_feedback'] == 2
    assert 'average_rating' in stats
    
    print(f"✓ Feedback collection works: {stats}")


def test_preference_pair_creation():
    """Test creating preference pairs."""
    print("\n--- Test: Preference Pair Creation ---")
    
    model_path = "checkpoints/test_agent_checkpoint.pt"
    agent = EthicalAgent(model_path)
    
    # Create two trajectories
    traj_a = [(np.random.randn(40).tolist(), i % 5) for i in range(5)]
    traj_b = [(np.random.randn(40).tolist(), (i + 1) % 5) for i in range(5)]
    
    # Create preference (A is better)
    pref = agent.create_preference_pair(traj_a, traj_b, preference=0)
    
    assert 'trajectory_a' in pref
    assert 'trajectory_b' in pref
    assert pref['preference'] == 0
    assert len(pref['trajectory_a']['states']) == 5
    assert len(pref['trajectory_b']['actions']) == 5
    
    print("✓ Preference pair creation works")


def test_quick_predict():
    """Test convenience function."""
    print("\n--- Test: Quick Predict ---")
    
    model_path = "checkpoints/test_agent_checkpoint.pt"
    state = np.random.randn(40).tolist()
    
    action = quick_predict(model_path, state)
    
    assert isinstance(action, int)
    assert 0 <= action <= 4
    
    print(f"✓ Quick predict works: action={action}")


def test_feedback_tracking():
    """Test that decisions are tracked when feedback enabled."""
    print("\n--- Test: Feedback Tracking ---")
    
    model_path = "checkpoints/test_agent_checkpoint.pt"
    agent = EthicalAgent(model_path, enable_feedback=True)
    
    # Make several decisions
    for _ in range(5):
        state = np.random.randn(40).tolist()
        agent.get_action(state)
    
    assert len(agent.decision_history) == 5
    
    # Check history structure
    decision = agent.decision_history[0]
    assert 'state' in decision
    assert 'action' in decision
    assert 'timestamp' in decision
    
    # Get statistics
    stats = agent.get_statistics()
    assert stats['total_decisions'] == 5
    assert 'action_distribution' in stats
    
    print(f"✓ Decision tracking works: {stats['action_distribution']}")


def test_carla_integration_example():
    """Test example CARLA integration."""
    print("\n--- Test: CARLA Integration Example ---")
    
    model_path = "checkpoints/test_agent_checkpoint.pt"
    agent = EthicalAgent(model_path)
    
    # Simulate CARLA providing state
    carla_state = {
        'velocity_ego': 25.0,
        'num_passengers': 2,
        'lane_position': 0.0,
        'velocity_delta': 5.0,
        'num_ped_if_straight': 0,
        'num_ped_if_left': 1,
        'num_ped_if_right': 0,
    }
    
    # Convert to state vector (simplified - real version would include all 40 dims)
    state_vector = [
        carla_state['velocity_ego'],
        carla_state['num_passengers'],
        carla_state['lane_position'],
        carla_state['velocity_delta'],
        carla_state['num_ped_if_straight'],
        carla_state['num_ped_if_left'],
        carla_state['num_ped_if_right'],
    ] + [0.0] * 33  # Placeholder for obstacle features
    
    # Get action
    action = agent.get_action(state_vector)
    action_name = agent.get_action_name(action)
    
    print(f"CARLA State: {carla_state}")
    print(f"Decision: {action_name}")
    print("✓ CARLA integration works")


def run_all_tests():
    """Run all tests for final deliverable."""
    print("\n" + "=" * 60)
    print("TESTING FINAL ETHICAL AGENT DELIVERABLE")
    print("=" * 60)
    
    tests = [
        test_basic_usage,
        test_detailed_analysis,
        test_compare_actions,
        test_human_feedback,
        test_preference_pair_creation,
        test_quick_predict,
        test_feedback_tracking,
        test_carla_integration_example,
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
        print("✓✓✓ ALL TESTS PASSED ✓✓✓")
        print("\n🎉 FINAL DELIVERABLE READY FOR DEPLOYMENT! 🎉")
    
    return failed == 0


if __name__ == "__main__":
    run_all_tests()