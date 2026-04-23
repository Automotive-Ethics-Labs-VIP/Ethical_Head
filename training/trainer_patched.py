"""
Ethical RLHF Trainer (patched v2)
===================================
Fixes vs v1:
  1. Reward normalization — learned rewards are z-score normalised before
     being passed to PPO so the value network doesn't blow up.
  2. Reward model is frozen after warm-up and only updated every 50 policy
     updates (was every 10). This stops the reward model from overfitting
     and sending adversarial signals to the policy.
  3. Best-checkpoint saving based on env_accuracy — the final checkpoint
     is the best-performing one, not the last one.
  4. Mixed reward throughout — even after warm-up we keep 30% shaped env
     reward so the policy always has a stable grounding signal.
  5. Warm-up epochs reduced to 1× (was 3×) to avoid early overfitting.
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Optional, Callable, List, Tuple
import time
import numpy as np

from networks.policy import PolicyNetwork
from networks.value import ValueNetwork
from networks.reward import RewardModel
from data.replay_buffer import ReplayBuffer
from algorithms.ppo import update as ppo_update
from training.reward_trainer import RewardModelTrainer
from training.config import Config
from environments.ethical_env import EthicalScenarioEnv


INVALID_ACTIONS = {1, 4}
VALID_ACTIONS   = [0, 2, 3]


class RLTrainer:

    def __init__(
        self,
        config: Config,
        env_interface=None,
        device: str = "cpu",
        theory: str = "utilitarian",
        split_path: str = "data/splits/train_test_split.json",
        direct_supervision: bool = False,
    ):
        self.config = config
        self.device = device
        self.theory = theory
        self.direct_supervision = direct_supervision

        torch.manual_seed(config.seed)
        np.random.seed(config.seed)

        # Networks
        print("Initializing networks...")
        self.policy_net   = PolicyNetwork().to(device)
        self.value_net    = ValueNetwork().to(device)
        self.reward_model = RewardModel().to(device)

        print(f"  Policy params : {sum(p.numel() for p in self.policy_net.parameters()):,}")
        print(f"  Value  params : {sum(p.numel() for p in self.value_net.parameters()):,}")
        print(f"  Reward params : {sum(p.numel() for p in self.reward_model.parameters()):,}")

        # Optimisers
        self.policy_optimizer = torch.optim.Adam(
            self.policy_net.parameters(), lr=config.learning_rate)
        self.value_optimizer  = torch.optim.Adam(
            self.value_net.parameters(),  lr=config.learning_rate)

        # Replay buffer
        self.buffer = ReplayBuffer(
            capacity=config.buffer_size, state_dim=config.state_dim)

        # Reward model trainer
        self.reward_trainer = RewardModelTrainer(
            self.reward_model, config, device=device)

        # Ethical environment
        self.env = EthicalScenarioEnv(
            theory=theory, split_path=split_path, split="train")

        # Action mask: large negative bias on invalid actions
        mask = torch.zeros(5)
        for a in INVALID_ACTIONS:
            mask[a] = -1e9
        self.action_logit_mask = mask.to(device)

        # Training state
        self.total_timesteps = 0
        self.num_updates     = 0
        self.training_history = {
            "policy_loss": [], "value_loss": [],
            "entropy": [], "episode_rewards": [],
            "timesteps": [], "env_accuracy": [],
        }
        self._reward_model_warmed_up = False
        self._best_env_accuracy = 0.0   # for best-checkpoint saving

        print(f"\nRLTrainer (ethical v2) on {device} | theory={theory}")
        print(f"  Env: {len(self.env.scenario_ids)} train scenarios")

    # =========================================================================
    # Preference balancing
    # =========================================================================

    def _balance_preferences(self, preference_dataset: List) -> List:
        from environments.ethical_env import POLICY_ACTION_TO_LABEL
        import random as _random

        left_pairs  = []
        other_pairs = []
        for traj_a, traj_b, pref in preference_dataset:
            preferred_action = int(traj_a[1][0].item())
            preferred_label  = POLICY_ACTION_TO_LABEL.get(preferred_action, 0)
            if preferred_label == 1:
                left_pairs.append((traj_a, traj_b, pref))
            else:
                other_pairs.append((traj_a, traj_b, pref))

        if not left_pairs:
            return preference_dataset

        target = len(other_pairs)
        if len(left_pairs) < target:
            oversampled = left_pairs * (target // len(left_pairs) + 1)
            left_pairs  = oversampled[:target]

        balanced = left_pairs + other_pairs
        _random.shuffle(balanced)
        return balanced

    # =========================================================================
    # Masked action sampling
    # =========================================================================

    def _masked_action(self, state_tensor: torch.Tensor) -> Tuple[int, float]:
        with torch.no_grad():
            probs        = self.policy_net(state_tensor)
            log_p        = torch.log(probs + 1e-8)
            masked_logits = log_p + self.action_logit_mask.unsqueeze(0)
            masked_probs  = torch.softmax(masked_logits, dim=-1)
            dist   = torch.distributions.Categorical(masked_probs)
            action = dist.sample()
            logp   = dist.log_prob(action)
        return int(action.item()), float(logp.item())

    # =========================================================================
    # Rollout collection
    # =========================================================================

    def collect_rollout(self, n_steps: Optional[int] = None,
                        render: bool = False) -> Dict[str, float]:
        n_steps = n_steps or self.config.buffer_size
        self.policy_net.eval()
        self.value_net.eval()

        episode_rewards   = []
        current_ep_reward = 0.0
        episodes_done     = 0
        correct_steps     = 0

        state = self.env.reset()

        for _ in range(n_steps):
            state_tensor = torch.FloatTensor(state).to(self.device)
            action, log_prob = self._masked_action(state_tensor)

            with torch.no_grad():
                value = self.value_net(state_tensor).squeeze().item()

            next_state, reward, done, info = self.env.step(action)

            self.buffer.store(
                state=state_tensor, action=action, reward=reward,
                value=value, log_prob=log_prob, done=float(done),
            )

            current_ep_reward += reward
            self.total_timesteps += 1
            if info.get("correct"):
                correct_steps += 1

            if done:
                episode_rewards.append(current_ep_reward)
                current_ep_reward = 0.0
                episodes_done += 1
                state = self.env.reset()
            else:
                state = next_state

        return {
            "steps_collected":     n_steps,
            "episodes_completed":  episodes_done,
            "mean_episode_reward": np.mean(episode_rewards) if episode_rewards else 0.0,
            "env_accuracy":        correct_steps / n_steps,
        }

    # =========================================================================
    # Combined reward — with normalization (KEY FIX)
    # =========================================================================

    def compute_combined_rewards(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        base_rewards: torch.Tensor,
    ) -> torch.Tensor:

        if self.direct_supervision:
            return base_rewards

        with torch.no_grad():
            learned_rewards = self.reward_model(states, actions)

        # ── Normalize learned rewards to prevent value-network explosion ──────
        # z-score within the batch, then clip to [-3, 3]
        r_mean = learned_rewards.mean()
        r_std  = learned_rewards.std().clamp(min=1e-6)
        learned_norm = ((learned_rewards - r_mean) / r_std).clamp(-3.0, 3.0)

        if self._reward_model_warmed_up:
            # 70% learned (normalised) + 30% shaped env reward
            # Keeping 30% env reward gives the policy a stable grounding
            # signal even when the reward model is highly confident
            return 0.7 * learned_norm + 0.3 * base_rewards
        else:
            return 0.3 * base_rewards + 0.7 * learned_norm

    # =========================================================================
    # Policy update
    # =========================================================================

    def update_policy(self) -> Dict[str, float]:
        states, actions, rewards, values, log_probs, dones = self.buffer.get_batch()

        states    = states.to(self.device)
        actions   = actions.to(self.device)
        rewards   = rewards.to(self.device)
        log_probs = log_probs.to(self.device)
        dones     = dones.to(self.device)

        combined_rewards = self.compute_combined_rewards(states, actions, rewards)

        with torch.no_grad():
            last_value = 0.0 if dones[-1] == 1.0 else \
                self.value_net(states[-1]).squeeze().item()

        advantages, returns = self.buffer.compute_advantages_and_returns(
            last_value=last_value,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
            rewards=combined_rewards,
        )

        advantages = advantages.to(self.device)
        returns    = returns.to(self.device)

        total_pl = total_vl = total_ent = 0.0
        for _ in range(self.config.ppo_epochs):
            metrics = ppo_update(
                policy_net=self.policy_net,
                value_net=self.value_net,
                policy_optimizer=self.policy_optimizer,
                value_optimizer=self.value_optimizer,
                states=states,
                actions=actions,
                old_log_probs=log_probs,
                returns=returns,
                advantages=advantages,
                epsilon=self.config.clip_epsilon,
                entropy_coef=self.config.entropy_coef,
                max_grad_norm=self.config.max_grad_norm,
            )
            total_pl  += metrics["policy_loss"]
            total_vl  += metrics["value_loss"]
            total_ent += metrics["entropy"]

        self.num_updates += 1

        return {
            "policy_loss":    total_pl  / self.config.ppo_epochs,
            "value_loss":     total_vl  / self.config.ppo_epochs,
            "entropy":        total_ent / self.config.ppo_epochs,
            "mean_advantage": advantages.mean().item(),
            "mean_return":    returns.mean().item(),
        }

    # =========================================================================
    # Main training loop
    # =========================================================================

    def train(
        self,
        total_timesteps: Optional[int] = None,
        preference_dataset: Optional[List] = None,
        log_callback: Optional[Callable] = None,
    ):
        total_timesteps = total_timesteps or self.config.total_timesteps

        print("\n" + "=" * 60)
        print("STARTING ETHICAL RLHF TRAINING (v2)")
        print("=" * 60)
        print(f"  Total timesteps : {total_timesteps:,}")
        print(f"  Theory          : {self.theory}")
        print(f"  Buffer size     : {self.config.buffer_size}")
        print(f"  PPO epochs      : {self.config.ppo_epochs}")
        print("=" * 60 + "\n")

        start_time = time.time()
        checkpoint_dir = Path(self.config.checkpoint_dir)

        # ── Reward model warm-up (1× epochs, not 3×) ─────────────────────────
        if not self.direct_supervision and preference_dataset and len(preference_dataset) > 0:
            print("Pre-training reward model (warm-up)...")
            balanced = self._balance_preferences(preference_dataset)
            warmup_metrics = self.reward_trainer.train_on_preferences(
                balanced,
                epochs=self.config.reward_model_epochs,   # just 1× — avoid overfit
                verbose=True,
            )
            print(f"  Warm-up accuracy: {warmup_metrics['accuracy']:.2%}")
            if warmup_metrics["accuracy"] > 0.55:
                self._reward_model_warmed_up = True
                print("  Reward model ready.")

        # ── Main loop ─────────────────────────────────────────────────────────
        # Reward model update frequency: every 50 policy updates (was 10)
        RM_UPDATE_FREQ = 50

        while self.total_timesteps < total_timesteps:

            rollout_stats  = self.collect_rollout()
            update_metrics = self.update_policy()
            env_acc        = rollout_stats["env_accuracy"]

            # Reward model update (less frequent to prevent overfitting)
            if (not self.direct_supervision and 
                self.num_updates % RM_UPDATE_FREQ == 0
                    and preference_dataset
                    and len(preference_dataset) > 0):

                balanced   = self._balance_preferences(preference_dataset)
                rm_metrics = self.reward_trainer.train_on_preferences(
                    balanced, epochs=2, verbose=False)   # just 2 epochs per update

                if rm_metrics["accuracy"] > 0.55:
                    self._reward_model_warmed_up = True

                if self.num_updates % self.config.log_interval == 0:
                    print(f"  [RM] accuracy={rm_metrics['accuracy']:.2%}")

            # ── Save best checkpoint whenever env_accuracy improves ───────────
            if env_acc > self._best_env_accuracy:
                self._best_env_accuracy = env_acc
                self.save_checkpoint(checkpoint_dir / "best_checkpoint.pt")
                print(f"  *** New best env_accuracy: {env_acc:.1%} — saved best_checkpoint.pt ***")

            # Logging
            if self.num_updates % self.config.log_interval == 0:
                elapsed = time.time() - start_time
                fps     = self.total_timesteps / elapsed if elapsed > 0 else 0

                print(f"\n[Update {self.num_updates}] "
                      f"steps={self.total_timesteps:,}/{total_timesteps:,}  "
                      f"fps={fps:.0f}")
                print(f"  env_accuracy  : {env_acc:.1%}  "
                      f"(best={self._best_env_accuracy:.1%})")
                print(f"  mean_reward   : {rollout_stats['mean_episode_reward']:.3f}")
                print(f"  policy_loss   : {update_metrics['policy_loss']:.4f}")
                print(f"  value_loss    : {update_metrics['value_loss']:.4f}")
                print(f"  entropy       : {update_metrics['entropy']:.4f}")

                self.training_history["policy_loss"].append(update_metrics["policy_loss"])
                self.training_history["value_loss"].append(update_metrics["value_loss"])
                self.training_history["entropy"].append(update_metrics["entropy"])
                self.training_history["episode_rewards"].append(
                    rollout_stats["mean_episode_reward"])
                self.training_history["env_accuracy"].append(env_acc)
                self.training_history["timesteps"].append(self.total_timesteps)

                if log_callback:
                    log_callback({"timesteps": self.total_timesteps,
                                  **rollout_stats, **update_metrics})

            # Periodic checkpoint
            if self.num_updates % self.config.save_interval == 0:
                self.save_checkpoint(
                    checkpoint_dir / f"checkpoint_{self.num_updates}.pt")

            self.buffer.clear()

        print("\n" + "=" * 60)
        print("TRAINING COMPLETE")
        print(f"  Total timesteps    : {self.total_timesteps:,}")
        print(f"  Total updates      : {self.num_updates}")
        print(f"  Best env_accuracy  : {self._best_env_accuracy:.1%}")
        print(f"  Time               : {time.time() - start_time:.1f}s")
        print("=" * 60)

        # Save final checkpoint AND copy best as final for test script
        self.save_checkpoint(checkpoint_dir / "final_checkpoint.pt")

        best_path = checkpoint_dir / "best_checkpoint.pt"
        if best_path.exists():
            import shutil
            shutil.copy(best_path, checkpoint_dir / "final_checkpoint.pt")
            print(f"  (best_checkpoint copied to final_checkpoint.pt)")

    # =========================================================================
    # Checkpoint helpers
    # =========================================================================

    def save_checkpoint(self, path: Path, metadata: Optional[Dict] = None):
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "policy_state_dict":           self.policy_net.state_dict(),
            "value_state_dict":            self.value_net.state_dict(),
            "reward_state_dict":           self.reward_model.state_dict(),
            "policy_optimizer_state_dict": self.policy_optimizer.state_dict(),
            "value_optimizer_state_dict":  self.value_optimizer.state_dict(),
            "config":                      self.config.to_dict(),
            "total_timesteps":             self.total_timesteps,
            "num_updates":                 self.num_updates,
            "training_history":            self.training_history,
            "best_env_accuracy":           self._best_env_accuracy,
        }
        if metadata:
            checkpoint["metadata"] = metadata
        torch.save(checkpoint, path)
        print(f"  Checkpoint saved → {path}")

    def load_checkpoint(self, path: Path):
        ckpt = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(ckpt["policy_state_dict"])
        self.value_net.load_state_dict(ckpt["value_state_dict"])
        self.reward_model.load_state_dict(ckpt["reward_state_dict"])
        self.policy_optimizer.load_state_dict(ckpt["policy_optimizer_state_dict"])
        self.value_optimizer.load_state_dict(ckpt["value_optimizer_state_dict"])
        self.total_timesteps     = ckpt["total_timesteps"]
        self.num_updates         = ckpt["num_updates"]
        self.training_history    = ckpt["training_history"]
        self._best_env_accuracy  = ckpt.get("best_env_accuracy", 0.0)
        print(f"Checkpoint loaded from {path}")