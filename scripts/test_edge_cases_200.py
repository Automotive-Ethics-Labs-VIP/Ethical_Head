"""
Test a trained ethical model on the 200-scenario held-out test set.

Usage:
    python scripts/test_edge_cases_200.py \
        --checkpoint checkpoints/utilitarian/final_checkpoint.pt \
        --theory utilitarian

    # Evaluate on the FULL 200 (train+test) to see overall fit:
    python scripts/test_edge_cases_200.py \
        --checkpoint checkpoints/utilitarian/final_checkpoint.pt \
        --theory utilitarian \
        --split all
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import json
from collections import defaultdict, Counter

from ethical_agent import EthicalAgent


# ── Our 3-action label space ──────────────────────────────────────────────────
ACTION_NAMES = {0: "maintain", 1: "swerve_left", 2: "swerve_right"}
ACTION_SHORT = {0: "MAINT", 1: "LEFT ", 2: "RIGHT"}

# EthicalAgent has a 5-action space:
#   0: maintain_course  → our 0 (maintain)
#   1: brake_hard       → our 0 (maintain — non-swerve conservative action)
#   2: swerve_left      → our 1 (swerve_left)
#   3: swerve_right     → our 2 (swerve_right)
#   4: accelerate       → our 0 (maintain — non-swerve)
AGENT_TO_LABEL = {0: 0, 1: 0, 2: 1, 3: 2, 4: 0}

def remap(agent_action: int) -> int:
    """Collapse agent's 5-action output → our 3-action label space."""
    return AGENT_TO_LABEL.get(agent_action, 0)


def load_split(split: str = "test"):
    """Load scenarios and decisions from the saved 200-scenario split."""
    with open("data/splits/train_test_split.json") as f:
        data = json.load(f)

    if split == "all":
        scenarios, decisions = {}, {}
        for part in ("train", "test"):
            scenarios.update(data[part]["scenarios"])
            decisions.update(data[part]["decisions"])
    else:
        scenarios = data[split]["scenarios"]
        decisions = data[split]["decisions"]

    return scenarios, decisions


def confusion_matrix_str(matrix: dict, labels: list) -> str:
    """Pretty-print a 3x3 confusion matrix."""
    header  = f"{'':14s}" + "".join(f"Pred {ACTION_SHORT[a]:6s}" for a in labels)
    divider = "-" * (14 + 12 * len(labels))
    rows = [header, divider]
    for true_a in labels:
        row = f"True {ACTION_SHORT[true_a]:9s}"
        for pred_a in labels:
            row += f"{matrix[true_a][pred_a]:12d}"
        rows.append(row)
    return "\n".join(rows)


def main():
    parser = argparse.ArgumentParser(description="Test ethical AV model on 200 scenarios")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--theory",     type=str, default="utilitarian",
                        choices=["utilitarian", "kantian", "mixed"])
    parser.add_argument("--split",      type=str, default="test",
                        choices=["train", "test", "all"],
                        help="Which split to evaluate on")
    parser.add_argument("--verbose",    action="store_true",
                        help="Print every scenario result (not just wrong ones)")
    args = parser.parse_args()

    # Load model
    print(f"\nLoading model from {args.checkpoint} ...")
    agent = EthicalAgent(args.checkpoint)

    # Load data
    scenarios, decisions = load_split(args.split)
    total = len(scenarios)

    print(f"\n{'='*70}")
    print(f"  EVALUATION  |  Theory: {args.theory.upper():12s}  "
          f"|  Split: {args.split.upper()}  ({total} scenarios)")
    print(f"{'='*70}\n")

    # Evaluate
    correct        = 0
    per_class_ok   = defaultdict(int)
    per_class_tot  = defaultdict(int)
    conf_matrix    = {a: {b: 0 for b in range(3)} for a in range(3)}
    wrong_cases    = []
    raw_action_dist = Counter()

    for sid in sorted(scenarios.keys(), key=lambda x: int(x.split("_S")[1])):
        state      = scenarios[sid]
        expected   = decisions[sid][args.theory]   # 0/1/2
        raw_action = agent.get_action(state)       # 0-4
        predicted  = remap(raw_action)             # 0/1/2

        raw_action_dist[raw_action] += 1
        is_correct = (predicted == expected)

        if is_correct:
            correct += 1
            per_class_ok[expected] += 1
        else:
            wrong_cases.append((sid, expected, predicted, raw_action))

        per_class_tot[expected] += 1
        conf_matrix[expected][predicted] += 1

        if args.verbose or not is_correct:
            mark = "X" if not is_correct else "v"
            raw_name = agent.ACTIONS.get(raw_action, str(raw_action))
            print(f"  [{mark}] {sid:12s}  "
                  f"Expected={ACTION_NAMES[expected]:12s}  "
                  f"Predicted={ACTION_NAMES[predicted]:12s}  "
                  f"(raw: {raw_name})")

    # Per-class accuracy
    print(f"\n{'-'*70}")
    print("  PER-CLASS ACCURACY")
    print(f"{'-'*70}")
    for action in sorted(per_class_tot.keys()):
        n   = per_class_tot[action]
        ok  = per_class_ok[action]
        pct = ok / n if n else 0
        bar = "#" * int(pct * 20)
        print(f"  {ACTION_NAMES[action]:15s}  {ok:3d}/{n:3d}  ({pct:5.1%})  {bar}")

    # Confusion matrix
    print(f"\n{'-'*70}")
    print("  CONFUSION MATRIX  (rows=True label, cols=Predicted label)")
    print(f"{'-'*70}")
    print(confusion_matrix_str(conf_matrix, sorted(per_class_tot.keys())))

    # Raw agent output distribution
    agent_action_names = getattr(agent, 'ACTIONS',
                                 {0:'maintain',1:'brake_hard',2:'swerve_left',
                                  3:'swerve_right',4:'accelerate'})
    print(f"\n{'-'*70}")
    print("  RAW AGENT ACTION DISTRIBUTION (before remapping)")
    print(f"{'-'*70}")
    for a in sorted(raw_action_dist.keys()):
        mapped = remap(a)
        print(f"  agent={a} ({agent_action_names.get(a,a):16s})"
              f"  -> label={mapped} ({ACTION_NAMES[mapped]:12s})"
              f"  count={raw_action_dist[a]}")

    # Overall accuracy
    accuracy = correct / total
    print(f"\n{'='*70}")
    print(f"  OVERALL ACCURACY: {correct}/{total} = {accuracy:.1%}")
    print(f"{'='*70}")

    if accuracy > 0.80:
        verdict = "Model learned the ethical framework well!"
    elif accuracy > 0.60:
        verdict = "Model partially learned the framework"
    else:
        verdict = "Model needs more training"
    print(f"  {verdict}\n")

    # Misclassified summary
    if wrong_cases:
        print(f"  MISCLASSIFIED ({len(wrong_cases)} scenarios):")
        for sid, exp, pred, raw in wrong_cases:
            raw_name = agent_action_names.get(raw, str(raw))
            print(f"    {sid:12s}  true={ACTION_NAMES[exp]:12s}  "
                  f"pred={ACTION_NAMES[pred]:12s}  (raw={raw_name})")

    # Label distribution
    label_dist = Counter(decisions[s][args.theory] for s in decisions)
    print(f"\n  Label distribution in this split ({args.theory}):")
    for a in sorted(label_dist):
        print(f"    {ACTION_NAMES[a]:15s}: {label_dist[a]:3d}  "
              f"({label_dist[a]/total:.0%})")
    print()

    # Warn about any unexpected raw action codes
    unmapped = [a for a in raw_action_dist if a not in AGENT_TO_LABEL]
    if unmapped:
        print(f"  WARNING: agent returned unexpected action codes {unmapped}.")
        print(f"  Update AGENT_TO_LABEL in this script to handle them.\n")


if __name__ == "__main__":
    main()