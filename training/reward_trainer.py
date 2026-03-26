"""
Reward Model Trainer
====================
This module trains the reward model using Bradley-Terry preference learning.

The reward model learns to predict ethical rewards from human preferences over
trajectory pairs. This is the "Human Feedback" component of RLHF.

Key Concept:
Given two trajectories A and B, and a human preference (A > B or B > A),
the model learns to assign higher cumulative reward to the preferred trajectory.

Bradley-Terry Model:
    P(A > B) = exp(R_A) / (exp(R_A) + exp(R_B)) = sigmoid(R_A - R_B)

Loss:
    Binary cross-entropy between predicted preference and actual preference
"""

import sys

import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from networks.reward import RewardModel
from algorithms.BT import bradley_terry_loss, trajectory_reward
from training.config import Config


class RewardModelTrainer:
    """
    Trainer for the reward model using Bradley-Terry preference learning.
    
    Attributes:
        reward_model: Neural network that scores (state, action) pairs
        optimizer: Optimizer for reward model parameters
        config: Training configuration
        device: Device for training ('cpu' or 'cuda')
    """
    
    def __init__(
        self,
        reward_model: RewardModel,
        config: Config,
        device: str = 'cpu'
    ):
        """
        Initialize the reward model trainer.
        
        Args:
            reward_model: RewardModel instance to train
            config: Configuration object with hyperparameters
            device: Device to train on ('cpu' or 'cuda')
        """
        self.reward_model = reward_model.to(device)
        self.config = config
        self.device = device
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(
            self.reward_model.parameters(),
            lr=config.reward_model_lr
        )
        
        # Training metrics
        self.training_history = {
            'losses': [],
            'accuracies': [],
            'epochs': []
        }
        
        print(f"RewardModelTrainer initialized on {device}")
        print(f"Learning rate: {config.reward_model_lr}")
        print(f"Model parameters: {sum(p.numel() for p in reward_model.parameters()):,}")
    
    
    def train_on_preferences(
        self,
        preference_dataset: List[Tuple],
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        verbose: bool = True
    ) -> Dict[str, float]:
        """
        Train reward model on preference dataset.
        
        Args:
            preference_dataset: List of (traj_A, traj_B, preference) tuples
                - traj_A: (states [T_A, 40], actions [T_A])
                - traj_B: (states [T_B, 40], actions [T_B])
                - preference: 1.0 if A preferred, 0.0 if B preferred
            epochs: Number of training epochs (default: from config)
            batch_size: Batch size (default: from config)
            verbose: Print training progress
        
        Returns:
            Dictionary with training metrics (loss, accuracy)
        """
        if epochs is None:
            epochs = self.config.reward_model_epochs
        if batch_size is None:
            batch_size = self.config.preference_batch_size
        
        self.reward_model.train()
        
        total_metrics = {
            'total_loss': 0.0,
            'total_correct': 0,
            'total_samples': 0
        }
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_correct = 0
            num_batches = 0
            
            # Shuffle dataset
            indices = np.random.permutation(len(preference_dataset))
            
            # Process in batches
            for i in range(0, len(preference_dataset), batch_size):
                batch_indices = indices[i:i + batch_size]
                batch_loss = 0.0
                batch_correct = 0
                
                for idx in batch_indices:
                    traj_A, traj_B, preference = preference_dataset[idx]
                    
                    # Move to device
                    states_A, actions_A = traj_A
                    states_B, actions_B = traj_B
                    
                    states_A = states_A.to(self.device)
                    actions_A = actions_A.to(self.device)
                    states_B = states_B.to(self.device)
                    actions_B = actions_B.to(self.device)
                    
                    # Compute loss
                    traj_A_device = (states_A, actions_A)
                    traj_B_device = (states_B, actions_B)
                    
                    loss = bradley_terry_loss(
                        self.reward_model,
                        traj_A_device,
                        traj_B_device,
                        preference
                    )
                    
                    batch_loss += loss
                    
                    # Compute accuracy (for monitoring)
                    with torch.no_grad():
                        R_A = trajectory_reward(self.reward_model, states_A, actions_A)
                        R_B = trajectory_reward(self.reward_model, states_B, actions_B)
                        predicted_preference = 1.0 if R_A > R_B else 0.0
                        if predicted_preference == preference:
                            batch_correct += 1
                
                # Backward pass on batch
                self.optimizer.zero_grad()
                batch_loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.reward_model.parameters(),
                    max_norm=1.0
                )
                
                self.optimizer.step()
                
                # Accumulate metrics
                epoch_loss += batch_loss.item()
                epoch_correct += batch_correct
                num_batches += 1
            
            # Average metrics for epoch
            avg_loss = epoch_loss / len(preference_dataset)
            accuracy = epoch_correct / len(preference_dataset)
            
            # Store metrics
            self.training_history['losses'].append(avg_loss)
            self.training_history['accuracies'].append(accuracy)
            self.training_history['epochs'].append(epoch + 1)
            
            total_metrics['total_loss'] += avg_loss
            total_metrics['total_correct'] += epoch_correct
            total_metrics['total_samples'] = len(preference_dataset)
            
            if verbose:
                print(f"Epoch {epoch + 1}/{epochs} | "
                      f"Loss: {avg_loss:.4f} | "
                      f"Accuracy: {accuracy:.2%}")
        
        # Average across all epochs
        final_metrics = {
            'loss': total_metrics['total_loss'] / epochs,
            'accuracy': total_metrics['total_correct'] / (epochs * total_metrics['total_samples']),
            'epochs': epochs
        }
        
        return final_metrics
    
    
    def evaluate(
        self,
        test_preferences: List[Tuple],
        verbose: bool = True
    ) -> Dict[str, float]:
        """
        Evaluate reward model on test preferences.
        
        Args:
            test_preferences: List of (traj_A, traj_B, preference) tuples
            verbose: Print evaluation results
        
        Returns:
            Dictionary with evaluation metrics
        """
        self.reward_model.eval()
        
        total_loss = 0.0
        total_correct = 0
        
        with torch.no_grad():
            for traj_A, traj_B, preference in test_preferences:
                # Move to device
                states_A, actions_A = traj_A
                states_B, actions_B = traj_B
                
                states_A = states_A.to(self.device)
                actions_A = actions_A.to(self.device)
                states_B = states_B.to(self.device)
                actions_B = actions_B.to(self.device)
                
                traj_A_device = (states_A, actions_A)
                traj_B_device = (states_B, actions_B)
                
                # Compute loss
                loss = bradley_terry_loss(
                    self.reward_model,
                    traj_A_device,
                    traj_B_device,
                    preference
                )
                total_loss += loss.item()
                
                # Compute accuracy
                R_A = trajectory_reward(self.reward_model, states_A, actions_A)
                R_B = trajectory_reward(self.reward_model, states_B, actions_B)
                predicted_preference = 1.0 if R_A > R_B else 0.0
                
                if predicted_preference == preference:
                    total_correct += 1
        
        metrics = {
            'test_loss': total_loss / len(test_preferences),
            'test_accuracy': total_correct / len(test_preferences)
        }
        
        if verbose:
            print(f"\nEvaluation Results:")
            print(f"  Loss: {metrics['test_loss']:.4f}")
            print(f"  Accuracy: {metrics['test_accuracy']:.2%}")
        
        return metrics
    
    
    def predict_preference(
        self,
        traj_A: Tuple[torch.Tensor, torch.Tensor],
        traj_B: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[float, float, float]:
        """
        Predict which trajectory the model prefers.
        
        Args:
            traj_A: (states [T_A, 40], actions [T_A])
            traj_B: (states [T_B, 40], actions [T_B])
        
        Returns:
            Tuple of (R_A, R_B, preference_prob)
            - R_A: Cumulative reward for trajectory A
            - R_B: Cumulative reward for trajectory B
            - preference_prob: P(A > B)
        """
        self.reward_model.eval()
        
        with torch.no_grad():
            states_A, actions_A = traj_A
            states_B, actions_B = traj_B
            
            states_A = states_A.to(self.device)
            actions_A = actions_A.to(self.device)
            states_B = states_B.to(self.device)
            actions_B = actions_B.to(self.device)
            
            R_A = trajectory_reward(self.reward_model, states_A, actions_A)
            R_B = trajectory_reward(self.reward_model, states_B, actions_B)
            
            # P(A > B) = sigmoid(R_A - R_B)
            preference_prob = torch.sigmoid(R_A - R_B)
            
            return R_A.item(), R_B.item(), preference_prob.item()
    
    
    def save_checkpoint(
        self,
        path: str,
        epoch: Optional[int] = None,
        metadata: Optional[Dict] = None
    ):
        """
        Save reward model checkpoint.
        
        Args:
            path: Path to save checkpoint
            epoch: Current epoch (optional)
            metadata: Additional metadata to save
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'reward_state_dict': self.reward_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_history': self.training_history,
            'config': self.config.to_dict(),
        }
        
        if epoch is not None:
            checkpoint['epoch'] = epoch
        
        if metadata is not None:
            checkpoint['metadata'] = metadata
        
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")
    
    
    def load_checkpoint(self, path: str):
        """
        Load reward model checkpoint.
        
        Args:
            path: Path to checkpoint file
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        self.reward_model.load_state_dict(checkpoint['reward_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'training_history' in checkpoint:
            self.training_history = checkpoint['training_history']
        
        print(f"Checkpoint loaded from {path}")
        
        if 'epoch' in checkpoint:
            print(f"  Epoch: {checkpoint['epoch']}")
        if 'metadata' in checkpoint:
            print(f"  Metadata: {checkpoint['metadata']}")
    
    
    def get_training_history(self) -> Dict[str, List]:
        """Get training history for plotting."""
        return self.training_history


# =========================================================================
# Utility Functions
# =========================================================================

def create_synthetic_preferences(
    num_pairs: int = 100,
    traj_length_range: Tuple[int, int] = (5, 15),
    state_dim: int = 40
) -> List[Tuple]:
    """
    Create synthetic preference data for testing.
    
    Args:
        num_pairs: Number of preference pairs to generate
        traj_length_range: (min_length, max_length) for trajectories
        state_dim: State dimension
    
    Returns:
        List of (traj_A, traj_B, preference) tuples
    """
    preferences = []
    
    for _ in range(num_pairs):
        # Random trajectory lengths
        len_A = np.random.randint(*traj_length_range)
        len_B = np.random.randint(*traj_length_range)
        
        # Generate random trajectories
        states_A = torch.randn(len_A, state_dim)
        actions_A = torch.randint(0, 5, (len_A,))
        
        states_B = torch.randn(len_B, state_dim)
        actions_B = torch.randint(0, 5, (len_B,))
        
        # Random preference
        preference = float(np.random.randint(0, 2))
        
        traj_A = (states_A, actions_A)
        traj_B = (states_B, actions_B)
        
        preferences.append((traj_A, traj_B, preference))
    
    return preferences


def split_preferences(
    preferences: List[Tuple],
    train_ratio: float = 0.8
) -> Tuple[List[Tuple], List[Tuple]]:
    """
    Split preferences into train and test sets.
    
    Args:
        preferences: List of preference pairs
        train_ratio: Fraction for training set
    
    Returns:
        (train_preferences, test_preferences)
    """
    n = len(preferences)
    n_train = int(n * train_ratio)
    
    # Shuffle
    indices = np.random.permutation(n)
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]
    
    train_prefs = [preferences[i] for i in train_indices]
    test_prefs = [preferences[i] for i in test_indices]
    
    return train_prefs, test_prefs