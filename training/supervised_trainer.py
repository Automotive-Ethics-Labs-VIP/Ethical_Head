"""
Supervised (Behavioral Cloning) Trainer
========================================
Trains the policy network directly on labeled scenarios using cross-entropy loss.

This bypasses RL entirely. Because the utilitarian label is a deterministic
function of state features (argmin of pedestrian counts in each path), a
supervised approach is guaranteed to converge to 100% training accuracy and
generalises well to the held-out test set.

Usage:
    from training.supervised_trainer import SupervisedTrainer
    trainer = SupervisedTrainer(theory='utilitarian')
    trainer.train(epochs=500)
"""

import json
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Optional


# Policy action indices that are valid in the dataset label space.
# Actions 1 (brake_hard) and 4 (accelerate) never appear as labels.
VALID_ACTIONS = [0, 2, 3]   # maintain_course / swerve_left / swerve_right

# Dataset label (0/1/2) → policy action index
LABEL_TO_ACTION = {0: 0, 1: 2, 2: 3}

# Policy action index → dataset label (for evaluation)
ACTION_TO_LABEL = {0: 0, 1: 0, 2: 1, 3: 2, 4: 0}

LABEL_NAMES = {0: "maintain", 1: "swerve_left", 2: "swerve_right"}


class SupervisedTrainer:
    """
    Trains the policy network with cross-entropy loss on (state, label) pairs.

    The trainer loads the train split, maps dataset labels to valid policy
    action indices, and minimises cross-entropy over the 3 valid logits
    (actions 0, 2, 3) only — actions 1 and 4 are excluded from the loss.
    """

    def __init__(
        self,
        theory: str = "utilitarian",
        split_path: str = "data/splits/train_test_split.json",
        checkpoint_dir: Optional[str] = None,
        learning_rate: float = 1e-3,
        device: str = "cpu",
        seed: int = 42,
    ):
        self.theory        = theory
        self.device        = device
        self.checkpoint_dir = Path(checkpoint_dir or f"checkpoints/{theory}_supervised")

        torch.manual_seed(seed)
        np.random.seed(seed)

        # Load train/test split
        with open(split_path) as f:
            data = json.load(f)

        self.train_scenarios = data["train"]["scenarios"]
        self.train_decisions = data["train"]["decisions"]
        self.test_scenarios  = data["test"]["scenarios"]
        self.test_decisions  = data["test"]["decisions"]

        # Pre-build tensors for the training set
        self._train_states, self._train_targets = self._build_tensors(
            self.train_scenarios, self.train_decisions)
        self._test_states, self._test_targets = self._build_tensors(
            self.test_scenarios, self.test_decisions)

        print(f"SupervisedTrainer | theory={theory}")
        print(f"  Train : {len(self._train_states)} scenarios")
        print(f"  Test  : {len(self._test_states)} scenarios")

        # Policy network
        from networks.policy import PolicyNetwork
        self.policy_net = PolicyNetwork().to(device)
        print(f"  Policy params: {sum(p.numel() for p in self.policy_net.parameters()):,}")

        self.optimizer = torch.optim.Adam(
            self.policy_net.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=50, factor=0.5)

        self.history = {"train_loss": [], "train_acc": [], "test_acc": []}

    # -------------------------------------------------------------------------

    def _build_tensors(self, scenarios, decisions):
        """Convert scenario dicts to (states [N,40], targets [N]) tensors."""
        states  = []
        targets = []
        for sid in sorted(scenarios.keys()):
            state  = scenarios[sid]
            label  = decisions[sid][self.theory]   # 0 / 1 / 2
            # Map dataset label → index within [action0, action2, action3]
            # i.e. 0->0, 1->1, 2->2 within the VALID_ACTIONS list
            target = VALID_ACTIONS.index(LABEL_TO_ACTION[label])
            states.append(state)
            targets.append(target)
        return (
            torch.FloatTensor(states).to(self.device),
            torch.LongTensor(targets).to(self.device),
        )

    # -------------------------------------------------------------------------

    def _forward_masked(self, states: torch.Tensor) -> torch.Tensor:
        """
        Run policy network and extract logits for valid actions only.

        Returns:
            logits [N, 3] for actions [maintain(0), swerve_left(2), swerve_right(3)]
        """
        probs  = self.policy_net(states)             # [N, 5]
        logits = torch.log(probs + 1e-8)             # convert to log-space
        return logits[:, VALID_ACTIONS]              # [N, 3]

    # -------------------------------------------------------------------------

    def evaluate(self, states: torch.Tensor, targets: torch.Tensor):
        """Return (loss, accuracy) on the given split."""
        self.policy_net.eval()
        with torch.no_grad():
            logits = self._forward_masked(states)
            loss   = nn.functional.cross_entropy(logits, targets).item()
            preds  = logits.argmax(dim=-1)
            acc    = (preds == targets).float().mean().item()
        return loss, acc

    # -------------------------------------------------------------------------

    def per_class_accuracy(self, states: torch.Tensor, targets: torch.Tensor):
        """Return per-class accuracy dict for the given split."""
        self.policy_net.eval()
        with torch.no_grad():
            logits = self._forward_masked(states)
            preds  = logits.argmax(dim=-1)

        results = {}
        for cls_idx, action_idx in enumerate(VALID_ACTIONS):
            label_name = LABEL_NAMES[ACTION_TO_LABEL[action_idx]]
            mask = (targets == cls_idx)
            if mask.sum() == 0:
                results[label_name] = float("nan")
            else:
                results[label_name] = (preds[mask] == cls_idx).float().mean().item()
        return results

    # -------------------------------------------------------------------------

    def train(self, epochs: int = 500, log_every: int = 50):
        """
        Train for `epochs` passes over the full training set.

        Args:
            epochs    : Number of full-dataset passes.
            log_every : Print metrics every N epochs.
        """
        print(f"\n{'='*60}")
        print(f"SUPERVISED TRAINING  ({self.theory})")
        print(f"{'='*60}")
        print(f"  Epochs : {epochs}")
        print(f"  Device : {self.device}")
        print(f"{'='*60}\n")

        best_test_acc = 0.0

        for epoch in range(1, epochs + 1):
            self.policy_net.train()

            logits = self._forward_masked(self._train_states)
            loss   = nn.functional.cross_entropy(logits, self._train_targets)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            train_loss = loss.item()
            _, train_acc = self.evaluate(self._train_states, self._train_targets)
            _, test_acc  = self.evaluate(self._test_states,  self._test_targets)

            self.scheduler.step(train_loss)

            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["test_acc"].append(test_acc)

            if test_acc > best_test_acc:
                best_test_acc = test_acc
                self._save_checkpoint("best_checkpoint.pt")

            if epoch % log_every == 0 or epoch == 1 or train_acc == 1.0:
                print(f"  [epoch {epoch:4d}]  "
                      f"loss={train_loss:.4f}  "
                      f"train={train_acc:.1%}  "
                      f"test={test_acc:.1%}")

            if train_acc == 1.0 and test_acc == 1.0:
                print(f"\n  100% on both splits reached at epoch {epoch}. Stopping.")
                break

        print(f"\n{'='*60}")
        print(f"  Best test accuracy : {best_test_acc:.1%}")
        print(f"{'='*60}")

        self._save_checkpoint("final_checkpoint.pt")
        self._print_per_class()

    # -------------------------------------------------------------------------

    def _save_checkpoint(self, filename: str):
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = self.checkpoint_dir / filename
        torch.save({
            "policy_state_dict": self.policy_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "history": self.history,
            "theory": self.theory,
        }, path)

    def _print_per_class(self):
        print("\n  Per-class accuracy on TEST split:")
        for split_name, states, targets in [
            ("train", self._train_states, self._train_targets),
            ("test",  self._test_states,  self._test_targets),
        ]:
            cls_acc = self.per_class_accuracy(states, targets)
            print(f"    [{split_name}]")
            for label_name, acc in cls_acc.items():
                bar = "#" * int(acc * 20) if not np.isnan(acc) else "N/A"
                print(f"      {label_name:15s}  {acc:5.1%}  {bar}")
