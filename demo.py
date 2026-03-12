"""
Demo Script - Try the Ethical Agent Yourself
=============================================

This script lets you interactively test the ethical agent with different scenarios.

Usage:
    python demo.py
"""

import numpy as np
from ethical_agent import EthicalAgent
from pathlib import Path


def create_test_scenarios():
    """Create some test scenarios with meaningful state vectors."""
    
    scenarios = {
        "normal_driving": {
            "description": "Normal highway driving, 2 passengers, clear road",
            "state": [
                25.0,  # velocity_ego (25 m/s ≈ 55 mph)
                2.0,   # num_passengers
                0.0,   # lane_position (centered)
                0.0,   # velocity_delta (matching traffic)
                0.0,   # num_ped_if_straight
                0.0,   # num_ped_if_left
                0.0,   # num_ped_if_right
            ] + [0.0] * 33  # obstacle features
        },
        
        "pedestrian_ahead": {
            "description": "Pedestrian directly ahead, need to decide",
            "state": [
                15.0,  # velocity_ego (slowing down)
                1.0,   # num_passengers
                0.0,   # lane_position
                -5.0,  # velocity_delta (slower than traffic)
                1.0,   # num_ped_if_straight (pedestrian ahead!)
                0.0,   # num_ped_if_left
                0.0,   # num_ped_if_right
            ] + [0.0] * 33
        },
        
        "left_lane_pedestrian": {
            "description": "Pedestrian in left lane, considering lane change",
            "state": [
                20.0,  # velocity_ego
                3.0,   # num_passengers (full car)
                0.5,   # lane_position (slightly right)
                2.0,   # velocity_delta
                0.0,   # num_ped_if_straight
                1.0,   # num_ped_if_left (pedestrian in left lane!)
                0.0,   # num_ped_if_right
            ] + [0.0] * 33
        },
        
        "multiple_pedestrians": {
            "description": "Pedestrians on both sides, difficult choice",
            "state": [
                10.0,  # velocity_ego (slow)
                2.0,   # num_passengers
                0.0,   # lane_position
                -10.0, # velocity_delta (much slower)
                0.0,   # num_ped_if_straight
                2.0,   # num_ped_if_left (2 pedestrians!)
                1.0,   # num_ped_if_right (1 pedestrian)
            ] + [0.0] * 33
        },
        
        "high_speed": {
            "description": "High speed highway, empty road",
            "state": [
                35.0,  # velocity_ego (35 m/s ≈ 78 mph)
                1.0,   # num_passengers
                0.0,   # lane_position
                5.0,   # velocity_delta (faster than traffic)
                0.0,   # num_ped_if_straight
                0.0,   # num_ped_if_left
                0.0,   # num_ped_if_right
            ] + [0.0] * 33
        },
    }
    
    return scenarios


def demo_basic_usage(agent, scenarios):
    """Demo 1: Basic action prediction."""
    print("\n" + "=" * 70)
    print("DEMO 1: BASIC USAGE - Get Actions for Different Scenarios")
    print("=" * 70)
    
    for name, scenario in scenarios.items():
        print(f"\nScenario: {scenario['description']}")
        print("-" * 70)
        
        action = agent.get_action(scenario['state'])
        action_name = agent.get_action_name(action)
        
        print(f"  State: velocity={scenario['state'][0]:.1f} m/s, "
              f"passengers={int(scenario['state'][1])}, "
              f"peds_ahead={int(scenario['state'][4])}")
        print(f"  → Decision: {action_name.upper()}")


def demo_detailed_analysis(agent, scenario):
    """Demo 2: Detailed decision analysis."""
    print("\n" + "=" * 70)
    print("DEMO 2: DETAILED ANALYSIS - Understand Why Agent Chose This Action")
    print("=" * 70)
    
    print(f"\nAnalyzing scenario: {scenario['description']}")
    print("-" * 70)
    
    analysis = agent.get_action_with_analysis(scenario['state'])
    
    print(f"\nChosen Action: {analysis['action_name'].upper()}")
    print(f"Confidence: {analysis['confidence']:.1%}")
    print(f"Expected Future Return: {analysis['value_estimate']:.4f}")
    
    print("\nAll Action Probabilities:")
    for action, prob in sorted(analysis['action_probabilities'].items(), 
                                key=lambda x: x[1], reverse=True):
        bar = "█" * int(prob * 50)
        print(f"  {action:20s} {prob:6.1%} {bar}")
    
    print("\nEthical Scores for Each Action:")
    for action, score in sorted(analysis['ethical_rewards'].items(), 
                                 key=lambda x: x[1], reverse=True):
        print(f"  {action:20s} {score:7.4f}")


def demo_compare_actions(agent, scenario):
    """Demo 3: Compare all possible actions."""
    print("\n" + "=" * 70)
    print("DEMO 3: COMPARE ACTIONS - See Ethical Scores for All Options")
    print("=" * 70)
    
    print(f"\nScenario: {scenario['description']}")
    print("-" * 70)
    
    scores = agent.compare_actions(scenario['state'])
    
    print("\nEthical Ranking:")
    for i, (action, score) in enumerate(sorted(scores.items(), 
                                                key=lambda x: x[1], 
                                                reverse=True), 1):
        print(f"  {i}. {action:20s} score={score:7.4f}")


def demo_human_feedback(agent):
    """Demo 4: Collecting human feedback."""
    print("\n" + "=" * 70)
    print("DEMO 4: HUMAN FEEDBACK - Rate Agent Decisions")
    print("=" * 70)
    
    # Create agent with feedback enabled
    feedback_agent = EthicalAgent(
        "checkpoints/test_agent_checkpoint.pt",
        enable_feedback=True
    )
    
    print("\nSimulating 3 decisions with human feedback...")
    print("-" * 70)
    
    scenarios_list = list(create_test_scenarios().values())[:3]
    
    for i, scenario in enumerate(scenarios_list, 1):
        action = feedback_agent.get_action(scenario['state'])
        action_name = feedback_agent.get_action_name(action)
        
        print(f"\nDecision {i}: {scenario['description']}")
        print(f"  Agent chose: {action_name}")
        
        # Simulate human rating
        rating = np.random.randint(3, 6)  # Random rating 3-5
        feedback_text = "Good" if rating >= 4 else "Okay"
        
        feedback_agent.add_human_feedback(
            scenario['state'],
            action,
            feedback=feedback_text,
            rating=rating
        )
        print(f"  Human rated: {rating}/5 - '{feedback_text}'")
    
    # Show statistics
    stats = feedback_agent.get_statistics()
    print(f"\nStatistics:")
    print(f"  Total decisions: {stats['total_decisions']}")
    print(f"  Average rating: {stats.get('average_rating', 0):.2f}/5")
    
    # Save feedback
    feedback_agent.save_feedback('data/demo_feedback.json')


def demo_random_states(agent, num_tests=5):
    """Demo 5: Test with random states."""
    print("\n" + "=" * 70)
    print(f"DEMO 5: RANDOM STATES - Test {num_tests} Random Scenarios")
    print("=" * 70)
    
    for i in range(num_tests):
        state = np.random.randn(40) * 10  # Random state
        
        action = agent.get_action(state)
        action_name = agent.get_action_name(action)
        
        print(f"\nRandom Test {i+1}:")
        print(f"  State vector: [{state[0]:.2f}, {state[1]:.2f}, {state[2]:.2f}, ...]")
        print(f"  Decision: {action_name}")


def interactive_mode(agent):
    """Interactive mode - let user input scenarios."""
    print("\n" + "=" * 70)
    print("INTERACTIVE MODE - Create Your Own Scenarios")
    print("=" * 70)
    
    print("\nEnter scenario parameters (or 'q' to quit):")
    
    while True:
        print("\n" + "-" * 70)
        try:
            velocity = input("Velocity (m/s, e.g., 25): ").strip()
            if velocity.lower() == 'q':
                break
            velocity = float(velocity)
            
            passengers = int(input("Number of passengers (0-5): "))
            peds_ahead = int(input("Pedestrians ahead (0-5): "))
            peds_left = int(input("Pedestrians left (0-5): "))
            peds_right = int(input("Pedestrians right (0-5): "))
            
            # Build state vector
            state = [
                velocity,
                float(passengers),
                0.0,  # lane_position
                0.0,  # velocity_delta
                float(peds_ahead),
                float(peds_left),
                float(peds_right),
            ] + [0.0] * 33
            
            # Get decision
            analysis = agent.get_action_with_analysis(state)
            
            print(f"\n→ DECISION: {analysis['action_name'].upper()}")
            print(f"  Confidence: {analysis['confidence']:.1%}")
            print(f"  Top 3 actions:")
            for action, prob in sorted(analysis['action_probabilities'].items(), 
                                       key=lambda x: x[1], 
                                       reverse=True)[:3]:
                print(f"    {action}: {prob:.1%}")
            
        except ValueError as e:
            print(f"Invalid input: {e}")
        except KeyboardInterrupt:
            break
    
    print("\nExiting interactive mode...")


def main():
    """Main demo function."""
    print("\n" + "=" * 70)
    print("ETHICAL AGENT DEMO")
    print("=" * 70)
    
    # Check for model
    model_path = "checkpoints/test_agent_checkpoint.pt"
    if not Path(model_path).exists():
        print(f"\n❌ No model found at {model_path}")
        print("Run tests first: python tests/test_agent.py")
        return
    
    # Load agent
    print(f"\nLoading model from {model_path}...")
    agent = EthicalAgent(model_path)
    
    # Get scenarios
    scenarios = create_test_scenarios()
    
    # Run demos
    demo_basic_usage(agent, scenarios)
    demo_detailed_analysis(agent, scenarios['pedestrian_ahead'])
    demo_compare_actions(agent, scenarios['multiple_pedestrians'])
    demo_human_feedback(agent)
    demo_random_states(agent, num_tests=3)
    
    # Interactive mode
    print("\n" + "=" * 70)
    response = input("\nWould you like to try interactive mode? (y/n): ").strip().lower()
    if response == 'y':
        interactive_mode(agent)
    
    print("\n" + "=" * 70)
    print("✓ DEMO COMPLETE!")
    print("=" * 70)
    print("\nNext steps:")
    print("  - Train a full model: python scripts/train.py")
    print("  - Integrate with CARLA")
    print("  - Collect human preferences")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()