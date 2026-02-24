"""
Trajectory Collection Script
============================
This script handles the "Generation" phase of the RLHF pipeline. 

It loads the current Policy Network (the autonomous agent's "brain") and allows it 
to interact with the environment to generate driving trajectories. 

Because the RLHF process requires comparing *pairs* of trajectories, this script
runs two separate rollouts for each scenario and saves them together as an "unannotated pair".
These pairs are exported to a JSON file so that a human annotator can later review
them and provide feedback.

Usage:
    python scripts/collect_trajectories.py --num_pairs 10
    
Outputs:
    Saves generated trajectory pairs to `data/unannotated_pairs.json` by default.
"""

import sys
from pathlib import Path
import json
import torch
import random
import os

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from networks.policy import PolicyNetwork
from data.preference_dataset import PreferenceDataset

class DummyEnvironment:
    """
    A temporary mock environment to simulate state generation
    until CARLA integration is complete.
    """
    def __init__(self):
        self.state_dim = 40
        self.max_steps = 100
        self.current_step = 0
        self.last_state = None

    def reset(self):
        self.current_step = 0
        # Start state: some random features + standard ego velocity
        state = torch.randn(self.state_dim)
        state[0] = 30.0 # e.g. 30 km/h velocity
        self.last_state = state
        return state
        
    def step(self, action):
        self.current_step += 1
        
        # Simulate world reacting to action (add slight noise to previous state)
        # In a real setup, CARLA determines the next state based on physics
        next_state = self.last_state + torch.randn(self.state_dim) * 0.1
        
        # Determine done flag based on step count or a random "crash" event
        done = self.current_step >= self.max_steps or random.random() < 0.05
        
        self.last_state = next_state
        return next_state, done

def run_rollout(policy_net, env, max_steps=100):
    """
    Runs a single trajectory rollout using the policy network.
    """
    states = []
    actions = []
    
    state = env.reset()
    
    with torch.no_grad():
        for _ in range(max_steps):
            # Convert state to batch format for network
            state_batch = state.unsqueeze(0)
            
            # Sample action
            action, _ = policy_net.sample_action(state_batch)
            
            # Store
            states.append(state.tolist())
            actions.append(action) # Action is an int returned by sample_action
            
            # Take step
            next_state, done = env.step(action)
            state = next_state
            
            if done:
                break
                
    return (states, actions)

def collect_trajectory_pairs(num_pairs=10, output_file="data/unannotated_pairs.json", max_steps=50):
    """
    Collects pairs of trajectories and saves them to a temporary JSON file.
    """
    print(f"Collecting {num_pairs} trajectory pairs...")
    
    # 1. Initialize Network and Environment
    policy = PolicyNetwork()
    
    # Optional: Load checkpoint if one exists
    checkpoint_path = project_root / "checkpoints" / "policy_latest.pt"
    if checkpoint_path.exists():
        policy.load_state_dict(torch.load(checkpoint_path))
        print(f"Loaded policy checkpoint from {checkpoint_path}")
    else:
        print("No checkpoint found. Using randomly initialized policy.")
        
    policy.eval() # Set to evaluation mode
    env = DummyEnvironment()
    
    pairs_data = []
    
    # 2. Generate Pairs
    for i in range(num_pairs):
        # Generate two rollouts (representing two different ways an episode might unfold)
        # Note: Since the dummy env is random, the same policy will yield different trajectories.
        # In a real setup, we might force different initial actions or sample from a stochastic policy
        # to ensure the trajectories are noticeably different for human comparison.
        traj_a = run_rollout(policy, env, max_steps=max_steps)
        traj_b = run_rollout(policy, env, max_steps=max_steps)
        
        pair_record = {
            "id": f"pair_{i:04d}",
            "trajectory_a": {
                "states": traj_a[0],
                "actions": traj_a[1]
            },
            "trajectory_b": {
                "states": traj_b[0],
                "actions": traj_b[1]
            }
        }
        pairs_data.append(pair_record)
        
    # 3. Save to file
    out_path = project_root / output_file
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(out_path, 'w') as f:
        json.dump(pairs_data, f, indent=4)
        
    print(f"Successfully saved {num_pairs} unannotated pairs to {output_file}.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Collect trajectory pairs for RLHF annotation.")
    parser.add_argument("--num_pairs", type=int, default=10, help="Number of pairs to generate")
    parser.add_argument("--output", type=str, default="data/unannotated_pairs.json", help="Output JSON file path")
    parser.add_argument("--max_steps", type=int, default=50, help="Max steps per trajectory")
    
    args = parser.parse_args()
    
    collect_trajectory_pairs(
        num_pairs=args.num_pairs,
        output_file=args.output,
        max_steps=args.max_steps
    )
