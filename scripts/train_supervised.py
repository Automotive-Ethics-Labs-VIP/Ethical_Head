"""
Train the utilitarian policy via supervised (behavioral cloning) learning.

Unlike the RL pipeline, this trains directly on ground-truth labels using
cross-entropy loss. Because the utilitarian label is a deterministic function
of state features, this reliably achieves 100% accuracy.

Usage:
    python scripts/train_supervised.py
    python scripts/train_supervised.py --theory utilitarian --epochs 500
    python scripts/train_supervised.py --epochs 1000 --lr 5e-4

After training, evaluate with the existing test script:
    python scripts/test_edge_cases_200.py \
        --checkpoint checkpoints/utilitarian_supervised/final_checkpoint.pt \
        --theory utilitarian
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
from training.supervised_trainer import SupervisedTrainer


def main():
    parser = argparse.ArgumentParser(
        description="Supervised behavioral cloning for ethical AV policy")
    parser.add_argument("--theory",  type=str, default="utilitarian",
                        choices=["utilitarian", "kantian"])
    parser.add_argument("--epochs",  type=int, default=500,
                        help="Number of training epochs (default: 500)")
    parser.add_argument("--lr",      type=float, default=1e-3,
                        help="Learning rate (default: 1e-3)")
    parser.add_argument("--device",  type=str, default="cpu")
    parser.add_argument("--split",   type=str,
                        default="data/splits/train_test_split.json")
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--log-every", type=int, default=50,
                        help="Print metrics every N epochs (default: 50)")
    args = parser.parse_args()

    trainer = SupervisedTrainer(
        theory=args.theory,
        split_path=args.split,
        checkpoint_dir=args.checkpoint_dir,
        learning_rate=args.lr,
        device=args.device,
    )

    trainer.train(epochs=args.epochs, log_every=args.log_every)

    ckpt = Path(trainer.checkpoint_dir) / "final_checkpoint.pt"
    print(f"\nEvaluate on held-out test set:")
    print(f"  python scripts/test_edge_cases_200.py \\")
    print(f"      --checkpoint {ckpt} \\")
    print(f"      --theory {args.theory}")


if __name__ == "__main__":
    main()
