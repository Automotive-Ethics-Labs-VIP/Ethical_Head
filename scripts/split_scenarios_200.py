"""
Split 200 scenarios into stratified train/test sets.

Stratified on utilitarian label (0/1/2) so the test set gets a
representative mix rather than a random clump of one class.

Usage:
    python scripts/split_scenarios_200.py
    python scripts/split_scenarios_200.py --train-ratio 0.85 --seed 99
"""
import json
import random
import argparse
from pathlib import Path
from collections import defaultdict


def stratified_split(scenarios: dict, decisions: dict,
                     theory: str = "utilitarian",
                     train_ratio: float = 0.80,
                     seed: int = 42):
    """
    Stratified split by `theory` label so each class is proportionally
    represented in both train and test.
    """
    random.seed(seed)

    # Group scenario IDs by their label
    buckets = defaultdict(list)
    for sid in scenarios:
        label = decisions[sid][theory]
        buckets[label].append(sid)

    train_ids, test_ids = [], []
    for label, ids in buckets.items():
        random.shuffle(ids)
        cut = max(1, int(len(ids) * train_ratio))   # keep at least 1 in train
        train_ids.extend(ids[:cut])
        test_ids.extend(ids[cut:])

    # Shuffle final lists for good measure
    random.shuffle(train_ids)
    random.shuffle(test_ids)

    train_scenarios  = {s: scenarios[s]  for s in train_ids}
    test_scenarios   = {s: scenarios[s]  for s in test_ids}
    train_decisions  = {s: decisions[s]  for s in train_ids}
    test_decisions   = {s: decisions[s]  for s in test_ids}

    return train_scenarios, test_scenarios, train_decisions, test_decisions


def save_split(train_scen, test_scen, train_dec, test_dec,
               output_dir: str = "data/splits"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    split_data = {
        "train": {"scenarios": train_scen, "decisions": train_dec},
        "test":  {"scenarios": test_scen,  "decisions": test_dec},
    }
    path = output_dir / "train_test_split.json"
    with open(path, "w") as f:
        json.dump(split_data, f, indent=2)

    print(f"Split saved → {path}")
    print(f"  Train : {len(train_scen):3d} scenarios")
    print(f"  Test  : {len(test_scen):3d} scenarios")
    return path


def label_dist(decisions, theory):
    from collections import Counter
    names = {0: "maintain", 1: "swerve_left", 2: "swerve_right"}
    c = Counter(v[theory] for v in decisions.values())
    return "  |  ".join(f"{names[a]}:{c[a]}" for a in sorted(c))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-ratio", type=float, default=0.80)
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--theory",      type=str,   default="utilitarian",
                        help="Which theory to stratify on (utilitarian|kantian)")
    args = parser.parse_args()

    # Load master data
    with open("data/scenarios_200.json") as f:
        scenarios = json.load(f)
    with open("data/decisions_200.json") as f:
        decisions = json.load(f)

    print(f"\nStratified split  (ratio={args.train_ratio}, seed={args.seed}, "
          f"stratify_on={args.theory})\n")

    train_scen, test_scen, train_dec, test_dec = stratified_split(
        scenarios, decisions,
        theory=args.theory,
        train_ratio=args.train_ratio,
        seed=args.seed,
    )

    # Print class distributions
    for split_name, dec in [("Train", train_dec), ("Test", test_dec)]:
        for theory in ["utilitarian", "kantian"]:
            print(f"  {split_name} {theory:12s}: {label_dist(dec, theory)}")
        print()

    save_split(train_scen, test_scen, train_dec, test_dec)

    # Also rebuild theory-specific preference files from TRAIN only
    # (avoids leaking test labels into training signal)
    pref_dir = Path("data/preferences")
    pref_dir.mkdir(exist_ok=True)

    from parse_200_scenarios import build_preference_pairs  # reuse helper
    for theory in ["utilitarian", "kantian"]:
        pairs = build_preference_pairs(train_scen, train_dec, theory)
        path = pref_dir / f"edge_cases_{theory}.json"
        with open(path, "w") as f:
            json.dump(pairs, f, indent=2)
        print(f"✓ Rebuilt {theory} preferences from train set "
              f"({len(pairs)} pairs) → {path}")

    print("\nNext: python scripts/train_edge_cases_200.py --theory utilitarian")


if __name__ == "__main__":
    main()
