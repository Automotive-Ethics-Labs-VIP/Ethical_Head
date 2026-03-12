"""
Ethical Self-Driving Agent - Final Deliverable
===============================================

This is the main user interface for the ethical self-driving decision system.
It provides a simple API for getting actions from state vectors and optionally
collecting human feedback.

Usage:
    # Basic usage
    from ethical_agent import EthicalAgent
    
    agent = EthicalAgent('checkpoints/trained_model.pt')
    state = [0.5, 2, 0.0, ...]  # 40-dim state from CARLA
    action = agent.get_action(state)
    
    # With human feedback
    agent.add_human_feedback(state, action, feedback="good decision")
    agent.save_feedback()

Author: Ethical Head Team
Date: 2024
"""

import torch
import numpy as np
from pathlib import Path
from typing import Union, List, Dict, Optional, Tuple
import json
from datetime import datetime

from inference.agent import EthicalAgent as BaseAgent


class EthicalAgent:
    """
    Main interface for ethical self-driving decisions.
    
    This class provides a simple API for:
    1. Getting actions from state vectors
    2. Analyzing decisions with confidence scores
    3. Collecting human feedback on decisions
    4. Generating preference data for model improvement
    """
    
    # Action mapping for easy reference
    ACTIONS = {
        0: 'maintain_course',
        1: 'brake_hard',
        2: 'swerve_left',
        3: 'swerve_right',
        4: 'accelerate'
    }
    
    def __init__(
        self, 
        model_path: str,
        device: str = 'cpu',
        enable_feedback: bool = False
    ):
        """
        Initialize the ethical agent.
        
        Args:
            model_path: Path to trained model checkpoint (.pt file)
            device: 'cpu' or 'cuda'
            enable_feedback: Whether to track decisions for feedback collection
        """
        # Load the underlying agent
        self._agent = BaseAgent(
            model_path, 
            device=device,
            load_value_network=True,
            load_reward_model=True
        )
        
        self.device = device
        self.enable_feedback = enable_feedback
        
        # Feedback tracking
        self.feedback_log = []
        self.decision_history = []
        
        print(f"✓ Ethical Agent loaded from {model_path}")
        print(f"  Device: {device}")
        print(f"  Feedback tracking: {'enabled' if enable_feedback else 'disabled'}")
    
    
    # =========================================================================
    # PRIMARY INTERFACE - Simple action prediction
    # =========================================================================
    
    def get_action(self, state: Union[List, np.ndarray]) -> int:
        """
        Get the recommended action for a given state.
        
        This is the main method for deployment.
        
        Args:
            state: 40-dimensional state vector
                [velocity_ego, num_passengers, lane_position, velocity_delta,
                 num_ped_if_straight, num_ped_if_left, num_ped_if_right,
                 obstacle_features[33]]
        
        Returns:
            action: Integer action code (0-4)
                0: maintain_course
                1: brake_hard
                2: swerve_left
                3: swerve_right
                4: accelerate
        
        Example:
            >>> agent = EthicalAgent('model.pt')
            >>> state = get_state_from_carla()
            >>> action = agent.get_action(state)
            >>> print(f"Action: {agent.get_action_name(action)}")
        """
        action = self._agent.get_action(state, deterministic=True)
        
        # Track decision if feedback enabled
        if self.enable_feedback:
            self.decision_history.append({
                'state': np.array(state).tolist() if not isinstance(state, list) else state,
                'action': action,
                'timestamp': datetime.now().isoformat()
            })
        
        return action
    
    
    def get_action_name(self, action: int) -> str:
        """Convert action code to human-readable name."""
        return self.ACTIONS[action]
    
    
    # =========================================================================
    # ADVANCED INTERFACE - Detailed analysis
    # =========================================================================
    
    def get_action_with_analysis(
        self, 
        state: Union[List, np.ndarray]
    ) -> Dict:
        """
        Get action with full ethical analysis.
        
        Returns detailed information about the decision including:
        - Recommended action
        - Confidence score
        - Probabilities for all actions
        - Expected future return (value)
        - Ethical scores for each action
        
        Args:
            state: 40-dimensional state vector
        
        Returns:
            Dictionary with complete analysis
        
        Example:
            >>> analysis = agent.get_action_with_analysis(state)
            >>> print(f"Action: {analysis['action_name']}")
            >>> print(f"Confidence: {analysis['confidence']:.0%}")
            >>> if analysis['confidence'] < 0.5:
            >>>     print("Warning: Low confidence decision!")
        """
        analysis = self._agent.analyze_decision(state)
        
        # Track if feedback enabled
        if self.enable_feedback:
            self.decision_history.append({
                'state': np.array(state).tolist() if not isinstance(state, list) else state,
                'action': analysis['action'],
                'confidence': analysis['confidence'],
                'analysis': analysis,
                'timestamp': datetime.now().isoformat()
            })
        
        return analysis
    
    
    def compare_actions(
        self, 
        state: Union[List, np.ndarray]
    ) -> Dict[str, float]:
        """
        Get ethical scores for all possible actions.
        
        Useful for understanding why the model chose a particular action.
        
        Args:
            state: 40-dimensional state vector
        
        Returns:
            Dictionary mapping action names to ethical scores
        
        Example:
            >>> scores = agent.compare_actions(state)
            >>> for action, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
            >>>     print(f"{action}: {score:.4f}")
        """
        scores = {}
        for action_code in range(5):
            score = self._agent.get_ethical_reward(state, action_code)
            scores[self.ACTIONS[action_code]] = score
        
        return scores
    
    
    # =========================================================================
    # HUMAN FEEDBACK INTERFACE
    # =========================================================================
    
    def add_human_feedback(
        self,
        state: Union[List, np.ndarray],
        action: int,
        feedback: str,
        rating: Optional[int] = None
    ):
        """
        Add human feedback on a decision.
        
        This allows humans to provide feedback on individual decisions,
        which can be used to improve the model.
        
        Args:
            state: The state vector for this decision
            action: The action taken
            feedback: Human feedback text (e.g., "good", "bad", "dangerous")
            rating: Optional numeric rating (e.g., 1-5)
        
        Example:
            >>> state = get_state()
            >>> action = agent.get_action(state)
            >>> # Execute action in simulation
            >>> # Human observes result
            >>> agent.add_human_feedback(state, action, "good decision", rating=5)
        """
        if not self.enable_feedback:
            print("Warning: Feedback tracking not enabled. Initialize with enable_feedback=True")
            return
        
        feedback_entry = {
            'state': np.array(state).tolist() if not isinstance(state, list) else state,
            'action': action,
            'action_name': self.ACTIONS[action],
            'feedback': feedback,
            'rating': rating,
            'timestamp': datetime.now().isoformat()
        }
        
        self.feedback_log.append(feedback_entry)
        print(f"✓ Feedback recorded: {feedback}")
    
    
    def save_feedback(self, filepath: str = 'data/human_feedback.json'):
        """
        Save collected feedback to file.
        
        Args:
            filepath: Where to save the feedback JSON
        """
        if not self.feedback_log:
            print("No feedback to save")
            return
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(self.feedback_log, f, indent=2)
        
        print(f"✓ Saved {len(self.feedback_log)} feedback entries to {filepath}")
    
    
    def create_preference_pair(
        self,
        trajectory_a: List[Tuple],
        trajectory_b: List[Tuple],
        preference: int
    ) -> Dict:
        """
        Create a preference comparison from two trajectories.
        
        This can be used to generate training data from human comparisons
        of different driving scenarios.
        
        Args:
            trajectory_a: List of (state, action) tuples for trajectory A
            trajectory_b: List of (state, action) tuples for trajectory B
            preference: 0 if A is better, 1 if B is better
        
        Returns:
            Preference data in format ready for training
        
        Example:
            >>> # Collect two trajectories
            >>> traj_a = [(state1, action1), (state2, action2), ...]
            >>> traj_b = [(state3, action3), (state4, action4), ...]
            >>> 
            >>> # Human decides A is more ethical
            >>> pref = agent.create_preference_pair(traj_a, traj_b, preference=0)
            >>> 
            >>> # Save for training
            >>> save_preference(pref)
        """
        states_a = [s for s, a in trajectory_a]
        actions_a = [a for s, a in trajectory_a]
        states_b = [s for s, a in trajectory_b]
        actions_b = [a for s, a in trajectory_b]
        
        return {
            'trajectory_a': {
                'states': states_a,
                'actions': actions_a
            },
            'trajectory_b': {
                'states': states_b,
                'actions': actions_b
            },
            'preference': preference,
            'timestamp': datetime.now().isoformat()
        }
    
    
    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    def get_statistics(self) -> Dict:
        """Get statistics about decisions and feedback."""
        stats = {
            'total_decisions': len(self.decision_history),
            'total_feedback': len(self.feedback_log),
        }
        
        if self.decision_history:
            actions = [d['action'] for d in self.decision_history]
            stats['action_distribution'] = {
                self.ACTIONS[i]: actions.count(i) for i in range(5)
            }
        
        if self.feedback_log:
            ratings = [f['rating'] for f in self.feedback_log if f.get('rating')]
            if ratings:
                stats['average_rating'] = sum(ratings) / len(ratings)
        
        return stats
    
    
    def clear_history(self):
        """Clear decision and feedback history."""
        self.decision_history = []
        self.feedback_log = []
        print("✓ History cleared")


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def quick_predict(model_path: str, state: Union[List, np.ndarray]) -> int:
    """
    Quick one-off prediction without creating agent instance.
    
    Args:
        model_path: Path to trained model
        state: 40-dim state vector
    
    Returns:
        Action code (0-4)
    
    Example:
        >>> action = quick_predict('model.pt', state)
    """
    agent = EthicalAgent(model_path)
    return agent.get_action(state)


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("ETHICAL AGENT - EXAMPLE USAGE")
    print("=" * 60)
    
    # Example 1: Basic usage
    print("\n[Example 1: Basic Usage]")
    print("-" * 60)
    
    # Check if model exists
    model_path = "checkpoints/final_checkpoint.pt"
    if not Path(model_path).exists():
        print(f"No model found at {model_path}")
        print("Train a model first using: python scripts/train.py")
    else:
        agent = EthicalAgent(model_path)
        
        # Random test state
        test_state = np.random.randn(40).tolist()
        
        # Get action
        action = agent.get_action(test_state)
        print(f"Recommended action: {agent.get_action_name(action)}")
    
    
    # Example 2: Detailed analysis
    print("\n[Example 2: Detailed Analysis]")
    print("-" * 60)
    
    if Path(model_path).exists():
        agent = EthicalAgent(model_path, enable_feedback=True)
        
        analysis = agent.get_action_with_analysis(test_state)
        
        print(f"Action: {analysis['action_name']}")
        print(f"Confidence: {analysis['confidence']:.2%}")
        print(f"Value estimate: {analysis['value_estimate']:.4f}")
        print("\nAction probabilities:")
        for action, prob in analysis['action_probabilities'].items():
            print(f"  {action}: {prob:.2%}")
    
    
    # Example 3: Compare all actions
    print("\n[Example 3: Compare Actions]")
    print("-" * 60)
    
    if Path(model_path).exists():
        scores = agent.compare_actions(test_state)
        print("Ethical scores for each action:")
        for action, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
            print(f"  {action}: {score:.4f}")
    
    
    # Example 4: Human feedback
    print("\n[Example 4: Human Feedback]")
    print("-" * 60)
    
    if Path(model_path).exists():
        agent.add_human_feedback(
            test_state, 
            action, 
            feedback="Appropriately cautious decision",
            rating=4
        )
        
        stats = agent.get_statistics()
        print(f"Statistics: {stats}")
        
        agent.save_feedback('data/example_feedback.json')
    
    
    print("\n" + "=" * 60)
    print("✓ Examples complete!")
    print("=" * 60)