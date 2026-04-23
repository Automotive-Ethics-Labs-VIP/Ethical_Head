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

# Label index → pedestrian-count dimension in the 40-D state vector
# used by the death-count shaped reward.
LABEL_TO_PED_DIM = {0: 4, 1: 5, 2: 6}  # maintain->dim4, left->dim5, right->dim6


class EthicalScenarioEnv:
    """
    Single-step environment built from the CATA scenario dataset.

    Each episode:
      reset() -> returns a 40-D state vector from a random training scenario
      step(action) -> returns (next_state, reward, done=True, info)

    Reward shaping (death-count based):
      Correct action : +1.0
      Wrong action   : -(deaths_chosen - deaths_optimal)  e.g. -3 if 4 peds chosen vs 1 optimal
      Invalid action (1 or 4): -5.0
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
        print(f"  Reward: death-count shaped  "
              f"(correct=+1, wrong=-(deaths_chosen - deaths_optimal), invalid=-5)")

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

        # Pedestrian counts for each path from the current state vector
        state = np.array(self.scenarios[self.current_sid], dtype=np.float32)
        peds  = [state[4], state[5], state[6]]  # [maintain, swerve_left, swerve_right]

        if action in INVALID_POLICY_ACTIONS:
            # Actions 1/4 never appear in data — hard penalty
            reward = -5.0

        elif predicted_label == true_label:
            reward = +1.0

        else:
            # Graded penalty: how many extra deaths did this choice cause?
            deaths_chosen  = peds[predicted_label]
            deaths_optimal = peds[true_label]
            reward = -(deaths_chosen - deaths_optimal)

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