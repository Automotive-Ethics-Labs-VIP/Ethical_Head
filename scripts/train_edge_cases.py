"""
Train the ethical model on edge case preferences.
"""
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
from training.trainer import RLTrainer
from training.config import Config
from data.preference_dataset import PreferenceDataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--theory', type=str, default='utilitarian',
                       choices=['utilitarian', 'kantian', 'mixed', 'util_lean'],
                       help='Which ethical framework to train')
    parser.add_argument('--timesteps', type=int, default=500_000,
                       help='Total training timesteps')
    parser.add_argument('--device', type=str, default='cpu')
    
    args = parser.parse_args()
    
    # Map theory to dataset file
    dataset_map = {
        'utilitarian': 'data/preferences/edge_cases_utilitarian.json',
        'kantian': 'data/preferences/edge_cases_kantian.json',
        'mixed': 'data/preferences/edge_cases_mixed_50_50.json',
        'util_lean': 'data/preferences/edge_cases_util_lean_70_30.json',
    }
    
    # Load preferences
    print(f"\nLoading {args.theory} preferences...")
    pref_dataset = PreferenceDataset()
    pref_dataset.load(dataset_map[args.theory])
    preferences = pref_dataset.get_training_format()
    
    print(f"Loaded {len(preferences)} preference pairs")
    print(f"Stats: {pref_dataset.get_statistics()}\n")
    
    # Setup config
    config = Config()
    config.total_timesteps = args.timesteps
    config.device = args.device
    config.checkpoint_dir = f"checkpoints/{args.theory}"
    config.log_dir = f"logs/{args.theory}"
    
    # Initialize trainer
    print("Initializing trainer...")
    trainer = RLTrainer(config, device=args.device)
    
    # Train
    print(f"\nTraining {args.theory} model for {args.timesteps:,} timesteps...")
    print("=" * 70)
    
    trainer.train(
        total_timesteps=args.timesteps,
        preference_dataset=preferences
    )
    
    print("\n" + "=" * 70)
    print("✓ Training complete!")
    print(f"Model saved to: {config.checkpoint_dir}/final_checkpoint.pt")
    print("=" * 70)


if __name__ == "__main__":
    main()