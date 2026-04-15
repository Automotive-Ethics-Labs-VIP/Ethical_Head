"""
Ethical Scenario Environment
=============================
Replaces the mock environment in RLTrainer with one driven entirely
by the real 200-scenario dataset.

Key design decisions:
1. Each "episode" is one scenario drawn from the training set.
   The episode has exactly 1 step — the AV makes one decision.

2. The base reward (R_base) is replaced with a shaped ethical reward
   derived directly from the ground-truth label, so even before the
   reward model is trained, the policy gets a meaningful signal:
       +1.0  if the action matches the theory label
       -1.0  if it does not

3. Actions 3 and 4 (brake_hard / accelerate) are never the correct
   label in the dataset.  To prevent the policy from wasting capacity
   on them we add a small penalty whenever they are chosen, and we
   expose a method to mask them out of the PPO logit space.

4. alpha is set to 0.0 / beta to 1.0 in the config so the combined
   reward = 0 * R_base + 1 * R_learned.  The shaped R_base here is
   used ONLY during the warm-up phase before the reward model is
   trained; once the reward model has been trained on preferences it
   takes over completely.

Usage (inside RLTrainer):
    from environments.ethical_env import EthicalScenarioEnv
    env = EthicalScenarioEnv(theory='utilitarian', device='cpu')
    trainer = RLTrainer(config, env_interface=env)
"""


import numpy as np
import json
import random
from pathlib import Path
from typing import Tuple, Dict, Optional


# ── Action space mappings ─────────────────────────────────────────────────────

# Map our 3 dataset labels → the policy's 5-action indices
LABEL_TO_POLICY_ACTION = {
    0: 0,   # maintain     -> maintain_course
    1: 2,   # swerve_left  -> swerve_left
    2: 3,   # swerve_right -> swerve_right
}

# Map policy's 5-action indices → our 3 labels
POLICY_ACTION_TO_LABEL = {
    0: 0,   # maintain_course  -> maintain
    1: 0,   # brake_hard       -> maintain (penalised)
    2: 1,   # swerve_left      -> swerve_left
    3: 2,   # swerve_right     -> swerve_right
    4: 0,   # accelerate       -> maintain (penalised)
}

# Actions that are never correct in the dataset
INVALID_POLICY_ACTIONS = {1, 4}

# ── Class-weighted rewards ────────────────────────────────────────────────────
# Dataset label frequencies: maintain=30%, swerve_left=50%, swerve_right=20%
# Inverse-frequency weights normalised so mean weight = 1.0
# This compensates for the imbalance so swerve_left doesn't get overshadowed.
_LABEL_FREQ  = {0: 0.30, 1: 0.50, 2: 0.20}
_RAW_WEIGHTS = {k: 1.0 / v for k, v in _LABEL_FREQ.items()}
_MEAN_WEIGHT = sum(_RAW_WEIGHTS.values()) / len(_RAW_WEIGHTS)
CLASS_REWARD_WEIGHTS = {k: v / _MEAN_WEIGHT for k, v in _RAW_WEIGHTS.items()}
# Resulting weights: maintain≈1.11, swerve_left≈0.67, swerve_right≈1.67
# (swerve_right boosted most since it's rarest; left slightly dampened since
#  it already dominates — this balances the gradient signal)


class EthicalScenarioEnv:
    """
    Single-step environment built from the CATA scenario dataset.

    Each episode:
      reset() -> returns a 40-D state vector from a random training scenario
      step(action) -> returns (next_state, reward, done=True, info)

    Reward shaping:
      Correct action : +CLASS_REWARD_WEIGHT  (weighted by inverse class freq)
      Wrong but valid: -CLASS_REWARD_WEIGHT * 0.5  (penalise wrong direction)
      Swerve confusion (left<->right): extra -0.5 penalty
      Invalid action (1 or 4): -1.0
    """

    def __init__(
        self,
        theory: str = "utilitarian",
        split_path: str = "data/splits/train_test_split.json",
        split: str = "train",
        seed: int = 42,
    ):
        self.theory = theory
        random.seed(seed)
        np.random.seed(seed)

        with open(split_path) as f:
            data = json.load(f)

        self.scenarios    = data[split]["scenarios"]
        self.decisions    = data[split]["decisions"]
        self.scenario_ids = list(self.scenarios.keys())
        self.current_sid   = None
        self.current_label = None
        self._step_count   = 0

        print(f"EthicalScenarioEnv: {len(self.scenario_ids)} {split} scenarios "
              f"| theory={theory}")
        print(f"  Class reward weights: "
              + "  ".join(f"label{k}={v:.2f}" for k, v in CLASS_REWARD_WEIGHTS.items()))

    # -------------------------------------------------------------------------
    def reset(self) -> np.ndarray:
        self.current_sid   = random.choice(self.scenario_ids)
        self.current_label = self.decisions[self.current_sid][self.theory]
        self._step_count   = 0
        return np.array(self.scenarios[self.current_sid], dtype=np.float32)

    # -------------------------------------------------------------------------
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        self._step_count += 1

        predicted_label = POLICY_ACTION_TO_LABEL[action]
        true_label      = self.current_label
        weight          = CLASS_REWARD_WEIGHTS[true_label]

        if action in INVALID_POLICY_ACTIONS:
            # Actions 1/4 never appear in data — hard penalty
            reward = -1.0

        elif predicted_label == true_label:
            # Correct — scale reward by inverse class frequency
            reward = +weight

        else:
            # Wrong direction — penalise proportionally
            base_penalty = -weight * 0.5

            # Extra penalty for confusing swerve directions (left vs right)
            # This is the main bug we're fixing: model was guessing swerve_right
            # for swerve_left scenarios because both are "swerve" actions
            swerve_confusion = (
                (true_label == 1 and predicted_label == 2) or  # left predicted as right
                (true_label == 2 and predicted_label == 1)     # right predicted as left
            )
            confusion_penalty = -0.5 if swerve_confusion else 0.0
            reward = base_penalty + confusion_penalty

        info = {
            "scenario_id":     self.current_sid,
            "true_label":      true_label,
            "policy_action":   action,
            "predicted_label": predicted_label,
            "correct":         predicted_label == true_label,
            "reward":          reward,
        }

        next_state = np.zeros(40, dtype=np.float32)
        return next_state, reward, True, info

    # -------------------------------------------------------------------------
    @property
    def action_mask(self) -> np.ndarray:
        mask = np.ones(5, dtype=bool)
        for a in INVALID_POLICY_ACTIONS:
            mask[a] = False
        return mask