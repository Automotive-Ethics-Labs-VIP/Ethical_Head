"""
Train the ethical RLHF model on 200 scenarios.

Usage:
    python scripts/train_edge_cases_200.py --theory utilitarian --timesteps 500000
    python scripts/train_edge_cases_200.py --theory kantian     --timesteps 200000
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
from training.trainer_patched import RLTrainer
from training.config import Config
from data.preference_dataset import PreferenceDataset


DATASET_MAP = {
    "utilitarian": "data/preferences/edge_cases_utilitarian.json",
    "kantian":     "data/preferences/edge_cases_kantian.json",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--theory",    type=str, default="utilitarian",
                        choices=list(DATASET_MAP.keys()))
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--device",    type=str, default="cpu")
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--direct-supervision", action="store_true",
                        help="Bypass preference-based reward model and use ground-truth direct labels.")
    args = parser.parse_args()

    # Load preferences
    print(f"\nLoading {args.theory} preferences...")
    pref_dataset = PreferenceDataset()
    pref_dataset.load(DATASET_MAP[args.theory])
    preferences = pref_dataset.get_training_format()
    print(f"Loaded {len(preferences)} preference pairs")
    print(f"Stats: {pref_dataset.get_statistics()}\n")

    # Config
    config = Config()
    config.total_timesteps = args.timesteps
    config.device          = args.device
    config.checkpoint_dir  = args.checkpoint_dir or f"checkpoints/{args.theory}"
    config.log_dir         = f"logs/{args.theory}"
    config.alpha = 0.0
    config.beta  = 1.0

    # Train
    trainer = RLTrainer(
        config,
        device=args.device,
        theory=args.theory,
        split_path="data/splits/train_test_split.json",
        direct_supervision=args.direct_supervision,
    )

    trainer.train(total_timesteps=args.timesteps, preference_dataset=preferences)

    ckpt = Path(config.checkpoint_dir) / "final_checkpoint.pt"
    print(f"\nTest with:")
    print(f"  python scripts/test_edge_cases_200.py --checkpoint {ckpt} --theory {args.theory}")


if __name__ == "__main__":
    main()