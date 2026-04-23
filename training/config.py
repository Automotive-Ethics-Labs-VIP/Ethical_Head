"""
Training Configuration
======================
Centralized configuration for all training hyperparameters.

This file defines all the hyperparameters used across the ethical self-driving
RL system. Using a config class ensures consistency and makes experimentation easy.

Usage:
    from training.config import Config
    
    config = Config()
    # Or load from file:
    config = Config.from_yaml('configs/experiment.yaml')
"""

from dataclasses import dataclass, asdict
from typing import Tuple
import yaml
from pathlib import Path


@dataclass
class Config:
    """
    Configuration for ethical self-driving RL training.
    
    All hyperparameters from your design document are defined here.
    """
    
    # =========================================================================
    # Network Architecture (matches your design)
    # =========================================================================
    state_dim: int = 40
    action_dim: int = 5
    policy_hidden_dims: Tuple[int, ...] = (256, 128)
    value_hidden_dims: Tuple[int, ...] = (256, 128)
    reward_hidden_dims: Tuple[int, ...] = (256, 128)
    
    # =========================================================================
    # PPO Hyperparameters
    # =========================================================================
    gamma: float = 0.99              # Discount factor for future rewards
    gae_lambda: float = 0.95         # GAE smoothing parameter (λ)
    clip_epsilon: float = 0.2        # PPO clipping parameter (ε)
    entropy_coef: float = 0.03       # Elevated entropy to prevent early collapse on tabular features
    value_loss_coef: float = 0.5     # Weight for value loss in total loss
    max_grad_norm: float = 0.5       # Gradient clipping threshold
    
    # =========================================================================
    # Training Loop
    # =========================================================================
    buffer_size: int = 2048          # Rollout length (from your design)
    ppo_epochs: int = 4              # Number of epochs per PPO update
    batch_size: int = 64             # Minibatch size for PPO updates
    learning_rate: float = 3e-4      # Learning rate for policy and value
    total_timesteps: int = 1_000_000 # Total training steps
    
    # =========================================================================
    # Reward Composition (α and β from your design)
    # =========================================================================
    alpha: float = 0.3               # Weight for R_base (safety rewards)
    beta: float = 0.7                # Weight for R_learned (ethical rewards)
    
    # Optional: Explicit ethical theory weights (if use_explicit_ethics=True)
    use_explicit_ethics: bool = False  # Enable explicit ethical scoring
    explicit_weight: float = 0.0               # Weight for explicit ethics (if enabled)
    # Note: If use_explicit_ethics=True, weights become:
    #   R_total = 0.3*R_base + 0.4*R_learned + 0.3*R_explicit
    
    # =========================================================================
    # Reward Model Training (Bradley-Terry)
    # =========================================================================
    reward_model_lr: float = 1e-3           # Learning rate for reward model
    reward_model_epochs: int = 5            # Epochs per reward model update
    preference_batch_size: int = 32         # Batch size for preference learning
    reward_model_update_freq: int = 10      # Update reward model every N policy updates
    
    # =========================================================================
    # Logging and Checkpointing
    # =========================================================================
    log_interval: int = 10           # Log metrics every N updates
    save_interval: int = 100         # Save checkpoint every N updates
    eval_interval: int = 50          # Evaluate policy every N updates
    eval_episodes: int = 10          # Number of episodes for evaluation
    
    # =========================================================================
    # Paths
    # =========================================================================
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    preference_data_dir: str = "data/preferences"
    
    # =========================================================================
    # Device
    # =========================================================================
    device: str = "cpu"              # 'cpu' or 'cuda'
    
    # =========================================================================
    # Random Seed
    # =========================================================================
    seed: int = 42
    
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        # Validate ranges
        assert 0.0 <= self.gamma <= 1.0, "gamma must be in [0, 1]"
        assert 0.0 <= self.gae_lambda <= 1.0, "gae_lambda must be in [0, 1]"
        assert self.clip_epsilon > 0, "clip_epsilon must be positive"
        assert self.alpha + self.beta == 1.0, f"alpha + beta must equal 1.0, got {self.alpha + self.beta}"
        
        # Validate dimensions
        assert self.state_dim == 40, "State dimension must be 40 (from design)"
        assert self.action_dim == 5, "Action dimension must be 5 (from design)"
        assert self.buffer_size == 2048, "Buffer size must be 2048 (from design)"
        
        # Create directories if they don't exist
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)
        Path(self.preference_data_dir).mkdir(parents=True, exist_ok=True)
    
    
    def to_dict(self):
        """Convert config to dictionary (with tuple->list conversion for YAML)."""
        config_dict = asdict(self)
        
        # Convert tuples to lists for YAML compatibility
        for key, value in config_dict.items():
            if isinstance(value, tuple):
                config_dict[key] = list(value)
        
        return config_dict
    
    
    def save_yaml(self, path: str):
        """Save configuration to YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
        
        print(f"Config saved to {path}")
    
    
    @classmethod
    def from_yaml(cls, path: str):
        """Load configuration from YAML file."""
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Convert lists back to tuples for hidden dims
        for key in ['policy_hidden_dims', 'value_hidden_dims', 'reward_hidden_dims']:
            if key in config_dict and isinstance(config_dict[key], list):
                config_dict[key] = tuple(config_dict[key])
        
        return cls(**config_dict)
    
    
    def __repr__(self):
        """Pretty print configuration."""
        lines = ["=" * 60, "Training Configuration", "=" * 60]
        
        sections = {
            "Network Architecture": [
                "state_dim", "action_dim", "policy_hidden_dims", 
                "value_hidden_dims", "reward_hidden_dims"
            ],
            "PPO Hyperparameters": [
                "gamma", "gae_lambda", "clip_epsilon", "entropy_coef",
                "value_loss_coef", "max_grad_norm"
            ],
            "Training Loop": [
                "buffer_size", "ppo_epochs", "batch_size", 
                "learning_rate", "total_timesteps"
            ],
            "Reward Composition": [
                "alpha", "beta"
            ],
            "Reward Model": [
                "reward_model_lr", "reward_model_epochs", 
                "preference_batch_size", "reward_model_update_freq"
            ],
            "Logging": [
                "log_interval", "save_interval", "eval_interval"
            ],
        }
        
        for section_name, keys in sections.items():
            lines.append(f"\n{section_name}:")
            lines.append("-" * 60)
            for key in keys:
                value = getattr(self, key)
                lines.append(f"  {key:.<40} {value}")
        
        lines.append("=" * 60)
        return "\n".join(lines)


# =========================================================================
# Convenience Functions
# =========================================================================

def get_default_config() -> Config:
    """Get default configuration matching the design document."""
    return Config()


def create_experiment_config(
    name: str,
    learning_rate: float = None,
    gamma: float = None,
    clip_epsilon: float = None,
    **kwargs
) -> Config:
    """
    Create a config for a specific experiment.
    
    Args:
        name: Experiment name (for logging/saving)
        learning_rate: Override default learning rate
        gamma: Override default discount factor
        clip_epsilon: Override default PPO clipping
        **kwargs: Any other config parameters to override
    
    Returns:
        Config object with overrides applied
    
    Example:
        config = create_experiment_config(
            name="high_lr_experiment",
            learning_rate=1e-3,
            gamma=0.95
        )
    """
    # Start with defaults
    config_dict = asdict(Config())
    
    # Apply overrides
    if learning_rate is not None:
        config_dict['learning_rate'] = learning_rate
    if gamma is not None:
        config_dict['gamma'] = gamma
    if clip_epsilon is not None:
        config_dict['clip_epsilon'] = clip_epsilon
    
    # Apply any additional kwargs
    config_dict.update(kwargs)
    
    # Update paths to include experiment name
    config_dict['checkpoint_dir'] = f"checkpoints/{name}"
    config_dict['log_dir'] = f"logs/{name}"
    
    return Config(**config_dict)


# =========================================================================
# Example Usage
# =========================================================================

if __name__ == "__main__":
    # Create default config
    config = get_default_config()
    print(config)
    print()
    
    # Save to YAML
    config.save_yaml("configs/default_config.yaml")
    
    # Load from YAML
    loaded_config = Config.from_yaml("configs/default_config.yaml")
    print("✓ Config successfully saved and loaded")
    print()
    
    # Create experiment config
    exp_config = create_experiment_config(
        name="test_experiment",
        learning_rate=1e-3,
        gamma=0.95
    )
    print("Experiment Config:")
    print(f"  Learning Rate: {exp_config.learning_rate}")
    print(f"  Gamma: {exp_config.gamma}")
    print(f"  Checkpoint Dir: {exp_config.checkpoint_dir}")