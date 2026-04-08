"""
Test the trained model on all 30 edge cases.
See what it learned!
"""
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


import argparse
from ethical_agent import EthicalAgent
import numpy as np

# Import scenarios from creation script
import json

def load_split_scenarios(split='test'):
    """Load scenarios/decisions from the saved train/test split."""
    with open('data/splits/train_test_split.json', 'r') as f:
        data = json.load(f)

    scenarios = data[split]['scenarios']
    decisions = data[split]['decisions']
    return scenarios, decisions

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--theory', type=str, default='utilitarian',
                       choices=['utilitarian', 'kantian', 'mixed'])
    
    args = parser.parse_args()
    
    # Load agent
    print(f"Loading model from {args.checkpoint}...")
    agent = EthicalAgent(args.checkpoint)
    
    # Test on all 30 scenarios
    print("\n" + "=" * 70)
    print(f"TESTING ON 30 EDGE CASES (Expected: {args.theory.upper()})")
    print("=" * 70)
    
    correct = 0
    total = 0
    
    scenarios, decisions = load_split_scenarios('test')

    print(f"\nTesting on {len(scenarios)} HELD-OUT scenarios...\n")
    print("Scenario IDs in test set:", list(scenarios.keys()))

    for scenario_id, state in scenarios.items():
        expected_action = decisions[scenario_id][args.theory]
        predicted_action = agent.get_action(state)
        
        is_correct = predicted_action == expected_action
        if is_correct:
            correct += 1
        total += 1
        
        status = "✓" if is_correct else "✗"
        print(f"{status} {scenario_id}: Predicted={agent.get_action_name(predicted_action)}, "
              f"Expected={agent.get_action_name(expected_action)}")
    
    accuracy = correct / total
    print("\n" + "=" * 70)
    print(f"ACCURACY: {correct}/{total} = {accuracy:.1%}")
    print("=" * 70)
    
    if accuracy > 0.8:
        print("✓ Model learned the ethical framework well!")
    elif accuracy > 0.6:
        print("⚠ Model partially learned the framework")
    else:
        print("✗ Model needs more training")


if __name__ == "__main__":
    main()