"""
Ethical RLHF Trainer (patched)
================================
Drop-in replacement for training/trainer.py that fixes three bugs:

  1. Uses EthicalScenarioEnv instead of the random mock environment,
     so rollouts come from real 40-D scenario states.

  2. Sets alpha=0 after reward-model warm-up so R_total = R_learned only
     (the mock base reward added noise; the shaped env reward handles warm-up).

  3. Masks policy actions 1 and 4 (never correct in the dataset) so the
     policy can only pick from {0, 2, 3} matching the dataset's label space.

Everything else — PPO, GAE, Bradley-Terry, checkpointing — is unchanged.
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


# Actions the policy should never pick (not in dataset label space)
INVALID_ACTIONS = {1, 4}
VALID_ACTIONS   = [0, 2, 3]   # maintain_course / swerve_left / swerve_right


class RLTrainer:
    """
    Ethical RLHF trainer — patched to use the scenario environment.
    API is identical to the original RLTrainer.
    """

    def __init__(
        self,
        config: Config,
        env_interface=None,   # kept for API compatibility; overridden internally
        device: str = "cpu",
        theory: str = "utilitarian",
        split_path: str = "data/splits/train_test_split.json",
    ):
        self.config  = config
        self.device  = device
        self.theory  = theory

        torch.manual_seed(config.seed)
        np.random.seed(config.seed)

        # ── Networks ──────────────────────────────────────────────────────────
        print("Initializing networks...")
        self.policy_net  = PolicyNetwork().to(device)
        self.value_net   = ValueNetwork().to(device)
        self.reward_model = RewardModel().to(device)

        print(f"  Policy params : {sum(p.numel() for p in self.policy_net.parameters()):,}")
        print(f"  Value  params : {sum(p.numel() for p in self.value_net.parameters()):,}")
        print(f"  Reward params : {sum(p.numel() for p in self.reward_model.parameters()):,}")

        # ── Optimisers ────────────────────────────────────────────────────────
        self.policy_optimizer = torch.optim.Adam(
            self.policy_net.parameters(), lr=config.learning_rate)
        self.value_optimizer  = torch.optim.Adam(
            self.value_net.parameters(),  lr=config.learning_rate)

        # ── Replay buffer ─────────────────────────────────────────────────────
        self.buffer = ReplayBuffer(
            capacity=config.buffer_size, state_dim=config.state_dim)

        # ── Reward model trainer ──────────────────────────────────────────────
        self.reward_trainer = RewardModelTrainer(
            self.reward_model, config, device=device)

        # ── Ethical environment (replaces mock) ───────────────────────────────
        self.env = EthicalScenarioEnv(
            theory=theory, split_path=split_path, split="train")

        # ── Action mask tensor (for logit masking in policy) ──────────────────
        # Shape [5]: 1.0 for valid actions, -1e9 for invalid
        mask = torch.zeros(5)
        for a in INVALID_ACTIONS:
            mask[a] = -1e9
        self.action_logit_mask = mask.to(device)

        # ── Training state ────────────────────────────────────────────────────
        self.total_timesteps = 0
        self.num_updates     = 0
        self.training_history = {
            "policy_loss": [], "value_loss": [],
            "entropy": [], "episode_rewards": [],
            "timesteps": [], "env_accuracy": [],
        }

        # Track reward-model warm-up: use shaped base reward for first N updates,
        # then switch to pure learned reward (alpha=0, beta=1)
        self._reward_model_warmed_up = False

        print(f"\nRLTrainer (ethical) initialised on {device}")
        print(f"  Theory  : {theory}")
        print(f"  Env     : {len(self.env.scenario_ids)} train scenarios")

    # =========================================================================
    # Masked policy action sampling
    # =========================================================================

    def _masked_action(self, state_tensor: torch.Tensor) -> Tuple[int, float]:
        """
        Sample an action from the policy, masking out invalid actions (1, 4).

        Returns:
            action (int), log_prob (float)
        """
        with torch.no_grad():
            probs = self.policy_net(state_tensor)          # [1, 5]
            # Add large negative bias to invalid action logits BEFORE softmax
            # Equivalent to zeroing their probability mass
            log_probs_raw = torch.log(probs + 1e-8)       # [1, 5]
            masked_logits = log_probs_raw + self.action_logit_mask.unsqueeze(0)

            # Re-normalise via softmax over masked logits
            masked_probs = torch.softmax(masked_logits, dim=-1)

            dist   = torch.distributions.Categorical(masked_probs)
            action = dist.sample()
            logp   = dist.log_prob(action)

        return int(action.item()), float(logp.item())


    # =========================================================================
    # Preference balancing — oversample minority classes for reward model
    # =========================================================================

    def _balance_preferences(self, preference_dataset: List) -> List:
        """Pass through the preference dataset unmodified.

        Inverse-frequency oversampling was removed because the death-count
        shaped reward in EthicalScenarioEnv already gives the policy graded
        signal proportional to how wrong each choice is. Artificially
        reweighting the reward model training data on top of that added bias
        against swerve_left (the majority class).
        """
        return preference_dataset

    # =========================================================================
    # Rollout collection (uses ethical env)
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

            # Masked action sampling
            action, log_prob = self._masked_action(state_tensor)

            with torch.no_grad():
                value = self.value_net(state_tensor).squeeze().item()

            # Step environment
            next_state, reward, done, info = self.env.step(action)

            self.buffer.store(
                state=state_tensor,
                action=action,
                reward=reward,
                value=value,
                log_prob=log_prob,
                done=float(done),
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

        env_accuracy = correct_steps / n_steps

        return {
            "steps_collected":    n_steps,
            "episodes_completed": episodes_done,
            "mean_episode_reward": np.mean(episode_rewards) if episode_rewards else 0.0,
            "env_accuracy":       env_accuracy,
        }

    # =========================================================================
    # Combined reward (patched: alpha=0 once reward model is warmed up)
    # =========================================================================

    def compute_combined_rewards(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        base_rewards: torch.Tensor,
        use_explicit_ethics: bool = False,
    ) -> torch.Tensor:

        with torch.no_grad():
            learned_rewards = self.reward_model(states, actions)

        if self._reward_model_warmed_up:
            # Pure learned reward — base reward was shaped noise-free env signal
            # but at this point the reward model carries all the ethical signal
            return learned_rewards
        else:
            # Warm-up: mix shaped base reward + (whatever the reward model says)
            alpha = self.config.alpha
            beta  = self.config.beta
            return alpha * base_rewards + beta * learned_rewards

    # =========================================================================
    # Policy update (unchanged from original)
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
        print("STARTING ETHICAL RLHF TRAINING")
        print("=" * 60)
        print(f"  Total timesteps : {total_timesteps:,}")
        print(f"  Theory          : {self.theory}")
        print(f"  Buffer size     : {self.config.buffer_size}")
        print(f"  PPO epochs      : {self.config.ppo_epochs}")
        print(f"  Reward model update every {self.config.reward_model_update_freq} updates")
        print("=" * 60 + "\n")

        start_time = time.time()

        # ── Initial reward model training (warm-up on full preference set) ────
        if preference_dataset and len(preference_dataset) > 0:
            print("Pre-training reward model on preferences (warm-up)...")
            balanced = self._balance_preferences(preference_dataset)
            warmup_metrics = self.reward_trainer.train_on_preferences(
                balanced,
                epochs=self.config.reward_model_epochs * 3,   # 3× for warm-up
                verbose=True,
            )
            print(f"  Warm-up reward model accuracy: {warmup_metrics['accuracy']:.2%}")
            if warmup_metrics["accuracy"] > 0.60:
                self._reward_model_warmed_up = True
                print("  Reward model warm-up complete — switching to pure learned reward.")

        # ── Main loop ─────────────────────────────────────────────────────────
        while self.total_timesteps < total_timesteps:

            rollout_stats = self.collect_rollout()
            update_metrics = self.update_policy()

            # Periodic reward model update
            if (self.num_updates % self.config.reward_model_update_freq == 0
                    and preference_dataset
                    and len(preference_dataset) > 0):

                balanced = self._balance_preferences(preference_dataset)
                rm_metrics = self.reward_trainer.train_on_preferences(
                    balanced, verbose=False)

                if rm_metrics["accuracy"] > 0.60:
                    self._reward_model_warmed_up = True

                if self.num_updates % self.config.log_interval == 0:
                    print(f"  [RM] accuracy={rm_metrics['accuracy']:.2%}  "
                          f"warmed_up={self._reward_model_warmed_up}")

            # Logging
            if self.num_updates % self.config.log_interval == 0:
                elapsed = time.time() - start_time
                fps = self.total_timesteps / elapsed if elapsed > 0 else 0

                print(f"\n[Update {self.num_updates}] "
                      f"steps={self.total_timesteps:,}/{total_timesteps:,}  "
                      f"fps={fps:.0f}")
                print(f"  env_accuracy  : {rollout_stats['env_accuracy']:.1%}  "
                      f"(policy correct actions on scenarios)")
                print(f"  mean_reward   : {rollout_stats['mean_episode_reward']:.3f}")
                print(f"  policy_loss   : {update_metrics['policy_loss']:.4f}")
                print(f"  value_loss    : {update_metrics['value_loss']:.4f}")
                print(f"  entropy       : {update_metrics['entropy']:.4f}")

                self.training_history["policy_loss"].append(update_metrics["policy_loss"])
                self.training_history["value_loss"].append(update_metrics["value_loss"])
                self.training_history["entropy"].append(update_metrics["entropy"])
                self.training_history["episode_rewards"].append(
                    rollout_stats["mean_episode_reward"])
                self.training_history["env_accuracy"].append(
                    rollout_stats["env_accuracy"])
                self.training_history["timesteps"].append(self.total_timesteps)

                if log_callback:
                    log_callback({
                        "timesteps": self.total_timesteps,
                        **rollout_stats, **update_metrics,
                    })

            # Checkpoint
            if self.num_updates % self.config.save_interval == 0:
                self.save_checkpoint(
                    Path(self.config.checkpoint_dir) / f"checkpoint_{self.num_updates}.pt")

            self.buffer.clear()

        print("\n" + "=" * 60)
        print("TRAINING COMPLETE")
        print(f"  Total timesteps : {self.total_timesteps:,}")
        print(f"  Total updates   : {self.num_updates}")
        print(f"  Time            : {time.time() - start_time:.1f}s")
        print("=" * 60)

        self.save_checkpoint(
            Path(self.config.checkpoint_dir) / "final_checkpoint.pt")

    # =========================================================================
    # Checkpoint helpers (unchanged)
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
        self.total_timesteps = ckpt["total_timesteps"]
        self.num_updates     = ckpt["num_updates"]
        self.training_history = ckpt["training_history"]
        print(f"Checkpoint loaded from {path}")