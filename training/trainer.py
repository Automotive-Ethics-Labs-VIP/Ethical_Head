"""
Main RL Trainer
===============
This module implements the main training loop that orchestrates the entire
ethical self-driving RL system with human feedback.

Training Flow:
1. Collect rollouts (2048 steps) using current policy
2. Compute advantages using GAE
3. Update policy using PPO
4. Periodically train reward model on human preferences
5. Log metrics and save checkpoints

This integrates:
- Policy Network (action selection)
- Value Network (advantage estimation)
- Reward Model (ethical scoring)
- GAE (advantage calculation)
- PPO (policy optimization)
- Replay Buffer (experience storage)
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


class RLTrainer:
    """
    Main trainer for ethical self-driving RL with human feedback.
    
    This class orchestrates the entire training pipeline, managing:
    - Rollout collection
    - Advantage computation
    - Policy updates
    - Reward model training
    - Logging and checkpointing
    """
    
    def __init__(
        self,
        config: Config,
        env_interface: Optional[Callable] = None,
        device: str = 'cpu'
    ):
        """
        Initialize the RL trainer.
        
        Args:
            config: Configuration object with all hyperparameters
            env_interface: Optional environment interface (mock if None)
            device: Device to train on ('cpu' or 'cuda')
        """
        self.config = config
        self.device = device
        self.env_interface = env_interface
        
        # Set random seed for reproducibility
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        
        # Initialize networks
        print("Initializing networks...")
        self.policy_net = PolicyNetwork().to(device)
        self.value_net = ValueNetwork().to(device)
        self.reward_model = RewardModel().to(device)
        
        print(f"Policy parameters: {sum(p.numel() for p in self.policy_net.parameters()):,}")
        print(f"Value parameters: {sum(p.numel() for p in self.value_net.parameters()):,}")
        print(f"Reward parameters: {sum(p.numel() for p in self.reward_model.parameters()):,}")
        
        # Initialize optimizers
        self.policy_optimizer = torch.optim.Adam(
            self.policy_net.parameters(),
            lr=config.learning_rate
        )
        self.value_optimizer = torch.optim.Adam(
            self.value_net.parameters(),
            lr=config.learning_rate
        )
        
        # Initialize replay buffer
        self.buffer = ReplayBuffer(
            capacity=config.buffer_size,
            state_dim=config.state_dim
        )
        
        # Initialize reward model trainer
        self.reward_trainer = RewardModelTrainer(
            self.reward_model,
            config,
            device=device
        )
        
        # Training state
        self.total_timesteps = 0
        self.num_updates = 0
        self.training_history = {
            'policy_loss': [],
            'value_loss': [],
            'entropy': [],
            'episode_rewards': [],
            'timesteps': []
        }
        
        print(f"RLTrainer initialized on {device}")
        print(f"Buffer size: {config.buffer_size}")
        print(f"Total training timesteps: {config.total_timesteps:,}")
    
    
    def collect_rollout(
        self,
        n_steps: Optional[int] = None,
        render: bool = False
    ) -> Dict[str, float]:
        """
        Collect a rollout of experience using the current policy.
        
        Args:
            n_steps: Number of steps to collect (default: buffer_size)
            render: Whether to render environment (if available)
        
        Returns:
            Dictionary with rollout statistics
        """
        if n_steps is None:
            n_steps = self.config.buffer_size
        
        self.policy_net.eval()
        self.value_net.eval()
        
        # Rollout statistics
        episode_rewards = []
        current_episode_reward = 0.0
        episodes_completed = 0
        
        # Get initial state (mock or from environment)
        state = self._get_initial_state()
        
        for step in range(n_steps):
            # Convert state to tensor
            state_tensor = torch.FloatTensor(state).to(self.device)
            
            # Policy selects action
            with torch.no_grad():
                action, log_prob = self.policy_net.sample_action(state_tensor)
                value = self.value_net(state_tensor).squeeze()
            
            # Convert to Python types for storage
            if isinstance(action, torch.Tensor):
                action = action.item()
            if isinstance(log_prob, torch.Tensor):
                log_prob = log_prob.item()
            if isinstance(value, torch.Tensor):
                value = value.item()
            
            # Execute action in environment
            next_state, reward, done, info = self._step_environment(state, action)
            
            # Store transition
            self.buffer.store(
                state=state_tensor,
                action=action,
                reward=reward,
                value=value,
                log_prob=log_prob,
                done=float(done)
            )
            
            # Update tracking
            current_episode_reward += reward
            self.total_timesteps += 1
            
            # Handle episode termination
            if done:
                episode_rewards.append(current_episode_reward)
                current_episode_reward = 0.0
                episodes_completed += 1
                state = self._get_initial_state()
            else:
                state = next_state
        
        # Compute statistics
        stats = {
            'steps_collected': n_steps,
            'episodes_completed': episodes_completed,
            'mean_episode_reward': np.mean(episode_rewards) if episode_rewards else 0.0,
            'min_episode_reward': np.min(episode_rewards) if episode_rewards else 0.0,
            'max_episode_reward': np.max(episode_rewards) if episode_rewards else 0.0,
        }
        
        return stats
    
    
    def compute_combined_rewards(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        base_rewards: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute combined reward: R_total = α * R_base + β * R_learned
        
        Args:
            states: Batch of states [T, 40]
            actions: Batch of actions [T]
            base_rewards: Base safety rewards [T]
        
        Returns:
            Combined rewards [T]
        """
        # Get learned ethical rewards from reward model
        with torch.no_grad():
            learned_rewards = self.reward_model(states, actions)
        
        # Combine with weights from config
        combined = (
            self.config.alpha * base_rewards +
            self.config.beta * learned_rewards
        )
        
        return combined
    
    
    def update_policy(self) -> Dict[str, float]:
        """
        Update policy and value networks using PPO.
        
        Returns:
            Dictionary with training metrics
        """
        # Get data from buffer
        states, actions, rewards, values, log_probs, dones = self.buffer.get_batch()
        
        # Move to device
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        log_probs = log_probs.to(self.device)
        dones = dones.to(self.device)
        
        # Compute combined rewards (base + learned ethical)
        combined_rewards = self.compute_combined_rewards(states, actions, rewards)
        
        # Get bootstrap value for last state
        with torch.no_grad():
            # If last step was terminal, next value is 0
            if dones[-1] == 1.0:
                last_value = 0.0
            else:
                # Otherwise, estimate value of next state
                last_state = states[-1]
                last_value = self.value_net(last_state).squeeze().item()
        
        # Compute advantages and returns using GAE
        advantages, returns = self.buffer.compute_advantages_and_returns(
            last_value=last_value,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda
        )
        
        advantages = advantages.to(self.device)
        returns = returns.to(self.device)
        
        # PPO update for multiple epochs
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        
        for epoch in range(self.config.ppo_epochs):
            # Single update (could be extended to minibatches)
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
                max_grad_norm=self.config.max_grad_norm
            )
            
            total_policy_loss += metrics['policy_loss']
            total_value_loss += metrics['value_loss']
            total_entropy += metrics['entropy']
        
        # Average metrics over epochs
        avg_metrics = {
            'policy_loss': total_policy_loss / self.config.ppo_epochs,
            'value_loss': total_value_loss / self.config.ppo_epochs,
            'entropy': total_entropy / self.config.ppo_epochs,
            'mean_advantage': advantages.mean().item(),
            'mean_return': returns.mean().item(),
        }
        
        self.num_updates += 1
        
        return avg_metrics
    
    
    def train(
        self,
        total_timesteps: Optional[int] = None,
        preference_dataset: Optional[List] = None,
        log_callback: Optional[Callable] = None
    ):
        """
        Main training loop.
        
        Args:
            total_timesteps: Total timesteps to train (default: from config)
            preference_dataset: Human preference data for reward model
            log_callback: Optional callback for logging (e.g., to wandb)
        """
        if total_timesteps is None:
            total_timesteps = self.config.total_timesteps
        
        print("\n" + "=" * 60)
        print("STARTING TRAINING")
        print("=" * 60)
        print(f"Total timesteps: {total_timesteps:,}")
        print(f"Buffer size: {self.config.buffer_size}")
        print(f"Updates per rollout: {self.config.ppo_epochs}")
        print("=" * 60 + "\n")
        
        start_time = time.time()
        
        while self.total_timesteps < total_timesteps:
            # === Collect Rollout ===
            rollout_stats = self.collect_rollout()
            
            # === Update Policy ===
            update_metrics = self.update_policy()
            
            # === Train Reward Model (periodically) ===
            if (self.num_updates % self.config.reward_model_update_freq == 0 
                and preference_dataset is not None 
                and len(preference_dataset) > 0):
                
                print(f"\n[Update {self.num_updates}] Training reward model...")
                reward_metrics = self.reward_trainer.train_on_preferences(
                    preference_dataset,
                    verbose=False
                )
                print(f"  Reward model accuracy: {reward_metrics['accuracy']:.2%}")
            
            # === Logging ===
            if self.num_updates % self.config.log_interval == 0:
                elapsed = time.time() - start_time
                fps = self.total_timesteps / elapsed if elapsed > 0 else 0
                
                print(f"\n[Update {self.num_updates}] Timesteps: {self.total_timesteps:,} / {total_timesteps:,}")
                print(f"  FPS: {fps:.1f}")
                print(f"  Episodes completed: {rollout_stats['episodes_completed']}")
                print(f"  Mean episode reward: {rollout_stats['mean_episode_reward']:.2f}")
                print(f"  Policy loss: {update_metrics['policy_loss']:.4f}")
                print(f"  Value loss: {update_metrics['value_loss']:.4f}")
                print(f"  Entropy: {update_metrics['entropy']:.4f}")
                
                # Store history
                self.training_history['policy_loss'].append(update_metrics['policy_loss'])
                self.training_history['value_loss'].append(update_metrics['value_loss'])
                self.training_history['entropy'].append(update_metrics['entropy'])
                self.training_history['episode_rewards'].append(rollout_stats['mean_episode_reward'])
                self.training_history['timesteps'].append(self.total_timesteps)
                
                # Optional callback
                if log_callback is not None:
                    log_callback({
                        'timesteps': self.total_timesteps,
                        'updates': self.num_updates,
                        **rollout_stats,
                        **update_metrics
                    })
            
            # === Save Checkpoint ===
            if self.num_updates % self.config.save_interval == 0:
                self.save_checkpoint(
                    Path(self.config.checkpoint_dir) / f"checkpoint_{self.num_updates}.pt"
                )
            
            # === Clear Buffer ===
            self.buffer.clear()
        
        print("\n" + "=" * 60)
        print("TRAINING COMPLETE")
        print("=" * 60)
        print(f"Total timesteps: {self.total_timesteps:,}")
        print(f"Total updates: {self.num_updates}")
        print(f"Training time: {time.time() - start_time:.1f}s")
        
        # Save final checkpoint
        self.save_checkpoint(
            Path(self.config.checkpoint_dir) / "final_checkpoint.pt"
        )
    
    
    def save_checkpoint(self, path: Path, metadata: Optional[Dict] = None):
        """
        Save training checkpoint.
        
        Args:
            path: Path to save checkpoint
            metadata: Optional additional metadata
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'policy_state_dict': self.policy_net.state_dict(),
            'value_state_dict': self.value_net.state_dict(),
            'reward_state_dict': self.reward_model.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'value_optimizer_state_dict': self.value_optimizer.state_dict(),
            'config': self.config.to_dict(),
            'total_timesteps': self.total_timesteps,
            'num_updates': self.num_updates,
            'training_history': self.training_history,
        }
        
        if metadata is not None:
            checkpoint['metadata'] = metadata
        
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")
    
    
    def load_checkpoint(self, path: Path):
        """
        Load training checkpoint.
        
        Args:
            path: Path to checkpoint file
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        self.policy_net.load_state_dict(checkpoint['policy_state_dict'])
        self.value_net.load_state_dict(checkpoint['value_state_dict'])
        self.reward_model.load_state_dict(checkpoint['reward_state_dict'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        self.value_optimizer.load_state_dict(checkpoint['value_optimizer_state_dict'])
        
        self.total_timesteps = checkpoint['total_timesteps']
        self.num_updates = checkpoint['num_updates']
        self.training_history = checkpoint['training_history']
        
        print(f"Checkpoint loaded from {path}")
        print(f"  Timesteps: {self.total_timesteps:,}")
        print(f"  Updates: {self.num_updates}")
    
    
    # =========================================================================
    # Environment Interface (Mock Implementation)
    # =========================================================================
    
    def _get_initial_state(self) -> np.ndarray:
        """
        Get initial state from environment.
        
        If env_interface is None, returns mock state.
        """
        if self.env_interface is not None:
            return self.env_interface.reset()
        else:
            # Mock: random 40-dim state
            return np.random.randn(40).astype(np.float32)
    
    
    def _step_environment(
        self,
        state: np.ndarray,
        action: int
    ) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Take a step in the environment.
        
        If env_interface is None, returns mock transition.
        
        Args:
            state: Current state
            action: Action to take
        
        Returns:
            (next_state, reward, done, info)
        """
        if self.env_interface is not None:
            return self.env_interface.step(action)
        else:
            # Mock environment
            next_state = np.random.randn(40).astype(np.float32)
            reward = np.random.randn()  # Random reward
            done = np.random.random() < 0.01  # 1% chance of episode end
            info = {}
            return next_state, reward, done, info