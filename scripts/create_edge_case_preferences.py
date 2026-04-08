"""
Convert your 30 edge case scenarios into preference pairs for training.

This creates comparisons between different ethical approaches:
- Utilitarian vs Kantian
- Utilitarian vs maintain course
- Kantian vs swerve options
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from data.preference_dataset import PreferenceDataset

# Your 30 edge case scenarios (40-dim vectors)
scenarios = {
    'CATA_S01': [10,1,0,0,1,2,0,  1,0,0,0,0,0,0,1,1,0,0,  1,0,0,0,0,0,0,0,0,0,0,  0,0,0,0,0,1,0,0,0,0,0],
    'CATA_S02': [10,1,0,0,1,2,0,  1,1,0,0,0,0,0,0,1,0,0,  1,0,0,0,0,0,0,0,0,1,0,  0,0,0,0,0,1,0,0,0,0,0],
    'CATA_S03': [10,2,0,0,2,1,0,  0,0,0,0,1,0,0,0,0,0,0,  1,0,0,0,0,0,0,0,0,0,1,  0,0,0,0,0,1,0,0,0,0,0],
    'CATA_S04': [10,1,0,0,2,2,0,  0,0,0,0,1,0,0,0,0,0,0,  1,0,0,0,0,0,0,1,0,0,0,  0,0,0,0,0,1,0,0,0,0,0],
    'CATA_S05': [10,3,0,0,3,1,0,  1,0,0,0,0,0,0,0,0,1,0,  1,1,0,0,0,0,0,0,1,0,0,  0,0,0,0,0,1,0,0,0,0,0],
    'CATA_S06': [10,1,0,0,3,2,0,  1,0,0,0,1,0,0,0,0,0,0,  1,1,0,0,0,0,0,0,1,0,0,  0,0,0,0,0,1,0,0,0,0,0],
    'CATA_S07': [10,2,0,0,4,1,0,  1,0,0,0,0,0,0,0,1,0,0,  1,1,0,0,0,0,0,1,0,0,0,  0,0,0,0,0,1,0,0,0,0,0],
    'CATA_S08': [10,2,0,0,2,2,0,  0,0,0,0,1,0,0,0,0,0,0,  1,0,0,0,0,0,0,0,0,0,1,  0,0,0,0,0,1,0,0,0,0,0],
    'CATA_S09': [10,1,0,0,1,3,0,  0,0,1,0,0,0,0,0,0,0,0,  1,0,0,0,0,0,0,1,0,0,0,  0,0,0,0,0,1,0,0,0,0,0],
    'CATA_S10': [10,3,0,0,3,1,0,  0,0,0,0,1,0,0,0,0,0,0,  1,0,0,0,0,0,0,0,1,0,1,  0,0,0,0,0,1,0,0,0,0,0],
    'CATA_S11': [10,2,0,0,2,2,0,  1,1,0,0,0,0,0,0,0,0,1,  1,0,0,0,0,0,0,0,1,0,0,  0,0,0,0,0,1,0,0,0,0,0],
    'CATA_S12': [10,2,0,0,1,2,0,  1,1,0,0,0,0,0,0,0,0,0,  1,0,0,0,0,0,0,1,0,0,0,  0,0,0,0,0,1,0,0,0,0,0],
    'CATA_S13': [10,3,0,0,2,1,0,  0,0,0,0,1,0,0,0,0,0,0,  1,0,0,0,0,0,0,0,1,1,0,  0,0,0,0,0,1,0,0,0,0,0],
    'CATA_S14': [10,2,0,0,3,2,0,  0,0,0,0,1,0,0,0,0,0,0,  1,1,0,0,0,0,0,0,0,0,1,  0,0,0,0,0,1,0,0,0,0,0],
    'CATA_S15': [10,1,0,0,1,0,0,  0,0,0,0,0,0,1,0,0,0,0,  1,0,0,0,0,0,0,0,0,0,0,  0,0,0,0,0,1,0,0,0,0,0],
    'CATA_S16': [10,2,0,0,3,2,0,  0,0,1,0,0,0,0,0,0,0,0,  1,1,0,0,0,0,0,0,1,0,0,  0,0,0,0,0,1,0,0,0,0,0],
    'CATA_S17': [10,3,0,0,4,1,0,  0,0,0,0,1,0,0,0,0,0,0,  1,1,0,0,0,0,0,0,0,0,1,  0,0,0,0,0,1,0,0,0,0,0],
    'CATA_S18': [10,1,0,0,2,2,0,  0,0,0,1,0,0,0,0,0,0,0,  1,0,0,0,0,0,0,1,0,0,0,  0,0,0,0,0,1,0,0,0,0,0],
    'CATA_S19': [10,2,0,0,1,3,0,  1,1,0,0,1,0,0,0,0,0,0,  1,0,0,0,0,0,0,0,1,0,0,  0,0,0,0,0,1,0,0,0,0,0],
    'CATA_S20': [10,3,0,0,2,0,0,  0,0,0,0,0,0,1,0,0,0,0,  1,0,0,0,0,0,0,0,0,0,1,  0,0,0,0,0,1,0,0,0,0,0],
    'CATA_S21': [10,2,0,0,3,3,0,  1,0,0,0,1,0,0,0,0,0,1,  1,1,0,0,0,0,0,0,1,0,0,  0,0,0,0,0,1,0,0,0,0,0],
    'CATA_S22': [10,2,0,0,2,4,0,  1,0,0,0,1,0,0,0,1,0,0,  1,1,0,0,0,0,0,0,0,0,0,  0,0,0,0,0,1,0,0,0,0,0],
    'CATA_S23': [10,3,0,0,1,3,0,  0,0,1,0,0,0,0,0,0,0,0,  1,0,0,0,0,0,0,1,0,0,0,  0,0,0,0,0,1,0,0,0,0,0],
    'CATA_S24': [10,4,0,0,4,2,0,  0,0,0,0,1,0,0,0,0,0,0,  1,1,0,0,0,0,0,0,1,0,0,  0,0,0,0,0,1,0,0,0,0,0],
    'CATA_S25': [10,2,0,0,3,3,0,  0,0,1,0,0,0,0,0,0,0,0,  1,1,0,0,0,0,0,0,0,0,1,  0,0,0,0,0,1,0,0,0,0,0],
    'CATA_S26': [10,3,0,0,2,2,0,  0,1,0,0,1,0,0,0,0,0,0,  1,0,0,0,0,0,0,0,1,0,0,  0,0,0,0,0,1,0,0,0,0,0],
    'CATA_S27': [10,2,0,0,5,2,0,  1,0,0,0,1,0,0,0,0,0,0,  1,1,0,0,0,0,0,1,1,0,0,  0,0,0,0,0,1,0,0,0,0,0],
    'CATA_S28': [10,1,0,0,2,5,0,  1,0,0,0,1,0,0,0,1,0,1,  1,1,0,0,0,0,0,0,0,0,0,  0,0,0,0,0,1,0,0,0,0,0],
    'CATA_S29': [10,4,0,0,3,1,0,  1,0,0,0,0,0,0,1,0,0,0,  1,1,0,0,0,0,0,0,1,0,0,  0,0,0,0,0,1,0,0,0,0,0],
    'CATA_S30': [10,1,0,0,4,4,0,  1,0,0,0,1,0,0,0,0,0,1,  1,1,0,0,0,0,0,0,1,0,0,  0,0,0,0,0,1,0,0,0,0,0],
}

# Ethical theory decisions for each scenario
# Actions: 0=maintain, 1=brake, 2=swerve_left, 3=swerve_right, 4=accelerate
decisions = {
    'CATA_S01': {'utilitarian': 3, 'kantian': 0},  # swerve_right vs maintain
    'CATA_S02': {'utilitarian': 0, 'kantian': 0},
    'CATA_S03': {'utilitarian': 2, 'kantian': 0},  # swerve_left vs maintain
    'CATA_S04': {'utilitarian': 3, 'kantian': 0},
    'CATA_S05': {'utilitarian': 2, 'kantian': 0},
    'CATA_S06': {'utilitarian': 3, 'kantian': 0},
    'CATA_S07': {'utilitarian': 2, 'kantian': 0},
    'CATA_S08': {'utilitarian': 0, 'kantian': 0},
    'CATA_S09': {'utilitarian': 0, 'kantian': 0},
    'CATA_S10': {'utilitarian': 2, 'kantian': 0},
    'CATA_S11': {'utilitarian': 0, 'kantian': 0},
    'CATA_S12': {'utilitarian': 0, 'kantian': 0},
    'CATA_S13': {'utilitarian': 2, 'kantian': 0},
    'CATA_S14': {'utilitarian': 2, 'kantian': 0},
    'CATA_S15': {'utilitarian': 2, 'kantian': 0},
    'CATA_S16': {'utilitarian': 2, 'kantian': 0},
    'CATA_S17': {'utilitarian': 2, 'kantian': 0},
    'CATA_S18': {'utilitarian': 3, 'kantian': 0},
    'CATA_S19': {'utilitarian': 0, 'kantian': 0},
    'CATA_S20': {'utilitarian': 2, 'kantian': 0},
    'CATA_S21': {'utilitarian': 3, 'kantian': 0},
    'CATA_S22': {'utilitarian': 0, 'kantian': 0},
    'CATA_S23': {'utilitarian': 0, 'kantian': 0},
    'CATA_S24': {'utilitarian': 2, 'kantian': 0},
    'CATA_S25': {'utilitarian': 3, 'kantian': 0},
    'CATA_S26': {'utilitarian': 0, 'kantian': 0},
    'CATA_S27': {'utilitarian': 2, 'kantian': 0},
    'CATA_S28': {'utilitarian': 3, 'kantian': 0},
    'CATA_S29': {'utilitarian': 2, 'kantian': 0},
    'CATA_S30': {'utilitarian': 3, 'kantian': 0},
}


def create_utilitarian_preferences():
    """
    Create preference dataset where Utilitarian choice is preferred.
    This teaches the model utilitarian ethics.
    """
    dataset = PreferenceDataset()
    
    for scenario_id, state in scenarios.items():
        util_action = decisions[scenario_id]['utilitarian']
        kant_action = decisions[scenario_id]['kantian']
        
        # Only create preference if theories disagree
        if util_action != kant_action:
            # Trajectory A: Utilitarian choice
            states_a = np.array([state])
            actions_a = np.array([util_action])
            
            # Trajectory B: Kantian choice
            states_b = np.array([state])
            actions_b = np.array([kant_action])
            
            # Preference: 0 = A is better (Utilitarian preferred)
            dataset.add_preference(
                (states_a, actions_a),
                (states_b, actions_b),
                preference=0,
                annotator_id='ethical_theory_util'
            )
    
    return dataset


def create_kantian_preferences():
    """
    Create preference dataset where Kantian choice is preferred.
    This teaches the model Kantian ethics.
    """
    dataset = PreferenceDataset()
    
    for scenario_id, state in scenarios.items():
        util_action = decisions[scenario_id]['utilitarian']
        kant_action = decisions[scenario_id]['kantian']
        
        # Only create preference if theories disagree
        if util_action != kant_action:
            # Trajectory A: Kantian choice
            states_a = np.array([state])
            actions_a = np.array([kant_action])
            
            # Trajectory B: Utilitarian choice
            states_b = np.array([state])
            actions_b = np.array([util_action])
            
            # Preference: 0 = A is better (Kantian preferred)
            dataset.add_preference(
                (states_a, actions_a),
                (states_b, actions_b),
                preference=0,
                annotator_id='ethical_theory_kant'
            )
    
    return dataset


def create_mixed_preferences(util_weight=0.5):
    """
    Create mixed preference dataset.
    util_weight: 0.0 = pure Kantian, 1.0 = pure Utilitarian, 0.5 = balanced
    """
    dataset = PreferenceDataset()
    
    for scenario_id, state in scenarios.items():
        util_action = decisions[scenario_id]['utilitarian']
        kant_action = decisions[scenario_id]['kantian']
        
        # Only create preference if theories disagree
        if util_action != kant_action:
            states_a = np.array([state])
            actions_a = np.array([util_action])
            
            states_b = np.array([state])
            actions_b = np.array([kant_action])
            
            # Randomly choose based on weight
            import random
            if random.random() < util_weight:
                # Prefer Utilitarian
                preference = 0  # A (util) better than B (kant)
                theory = 'utilitarian'
            else:
                # Prefer Kantian
                preference = 1  # B (kant) better than A (util)
                theory = 'kantian'
            
            dataset.add_preference(
                (states_a, actions_a),
                (states_b, actions_b),
                preference=preference,
                annotator_id=f'mixed_{util_weight}'
            )
    
    return dataset


if __name__ == "__main__":
    print("Creating preference datasets from 30 edge cases...")
    print("=" * 70)
    
    # Option 1: Pure Utilitarian
    util_dataset = create_utilitarian_preferences()
    util_dataset.save('data/preferences/edge_cases_utilitarian.json')
    print(f"✓ Utilitarian preferences: {len(util_dataset)} pairs")
    print(f"  Stats: {util_dataset.get_statistics()}")
    
    # Option 2: Pure Kantian
    kant_dataset = create_kantian_preferences()
    kant_dataset.save('data/preferences/edge_cases_kantian.json')
    print(f"✓ Kantian preferences: {len(kant_dataset)} pairs")
    print(f"  Stats: {kant_dataset.get_statistics()}")
    
    # Option 3: Mixed (50/50)
    mixed_dataset = create_mixed_preferences(util_weight=0.5)
    mixed_dataset.save('data/preferences/edge_cases_mixed_50_50.json')
    print(f"✓ Mixed (50/50) preferences: {len(mixed_dataset)} pairs")
    print(f"  Stats: {mixed_dataset.get_statistics()}")
    
    # Option 4: Utilitarian-leaning (70/30)
    util_lean_dataset = create_mixed_preferences(util_weight=0.7)
    util_lean_dataset.save('data/preferences/edge_cases_util_lean_70_30.json')
    print(f"✓ Utilitarian-leaning (70/30) preferences: {len(util_lean_dataset)} pairs")
    
    print("\n" + "=" * 70)
    print("✓ All preference datasets created!")
    print("\nNext step: Train the model with one of these datasets")
    print("=" * 70)