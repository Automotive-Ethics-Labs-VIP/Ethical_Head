"""
Split scenarios into train/test sets for proper evaluation.
"""

import json
import random
from pathlib import Path

def split_scenarios(scenarios, decisions, train_ratio=0.8, seed=42):
    """
    Split scenarios into train and test sets.
    
    Args:
        scenarios: Dict of scenario_id -> state vector
        decisions: Dict of scenario_id -> {utilitarian, kantian}
        train_ratio: Fraction for training (0.8 = 80% train, 20% test)
        seed: Random seed for reproducibility
    
    Returns:
        (train_scenarios, test_scenarios, train_decisions, test_decisions)
    """
    random.seed(seed)
    
    scenario_ids = list(scenarios.keys())
    random.shuffle(scenario_ids)
    
    split_idx = int(len(scenario_ids) * train_ratio)
    
    train_ids = scenario_ids[:split_idx]
    test_ids = scenario_ids[split_idx:]
    
    train_scenarios = {sid: scenarios[sid] for sid in train_ids}
    test_scenarios = {sid: scenarios[sid] for sid in test_ids}
    
    train_decisions = {sid: decisions[sid] for sid in train_ids}
    test_decisions = {sid: decisions[sid] for sid in test_ids}
    
    return train_scenarios, test_scenarios, train_decisions, test_decisions


def save_split(train_scenarios, test_scenarios, train_decisions, test_decisions, output_dir='data/splits'):
    """Save train/test split to files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save splits
    split_data = {
        'train': {
            'scenarios': train_scenarios,
            'decisions': train_decisions
        },
        'test': {
            'scenarios': test_scenarios,
            'decisions': test_decisions
        }
    }
    
    with open(output_dir / 'train_test_split.json', 'w') as f:
        json.dump(split_data, f, indent=2)
    
    print(f"Split saved to {output_dir}/train_test_split.json")
    print(f"  Train: {len(train_scenarios)} scenarios")
    print(f"  Test: {len(test_scenarios)} scenarios")


if __name__ == "__main__":
    from create_edge_case_preferences import scenarios, decisions
    
    # For 30 scenarios: 80/20 split = 24 train, 6 test
    train_scen, test_scen, train_dec, test_dec = split_scenarios(
        scenarios, decisions, train_ratio=0.8
    )
    
    save_split(train_scen, test_scen, train_dec, test_dec)
    
    print("\nTrain scenarios:", list(train_scen.keys()))
    print("\nTest scenarios:", list(test_scen.keys()))