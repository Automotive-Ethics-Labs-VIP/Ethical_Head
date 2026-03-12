"""
Explicit Ethical Theory Implementations
========================================

This module implements explicit ethical scoring based on:
1. Utilitarianism - Maximize total welfare
2. Kantian Ethics - Treat people as ends, not means
3. Virtue Ethics - Consider character and intentions

These can be combined with learned rewards or used as baselines.
"""

import numpy as np
from typing import Dict, List


class UtilitarianScorer:
    """
    Utilitarian ethics: Maximize total welfare, minimize total harm.
    
    Core principle: The right action is the one that produces the greatest
    good for the greatest number.
    """
    
    @staticmethod
    def score_action(state: np.ndarray, action: int) -> float:
        """
        Score action based on utilitarian principles.
        
        Args:
            state: 40-dim state vector
            action: Action taken (0-4)
        
        Returns:
            Utilitarian score (higher = more ethical)
        """
        # Extract relevant state features
        velocity = state[0]
        num_passengers = state[1]
        num_ped_straight = state[4]
        num_ped_left = state[5]
        num_ped_right = state[6]
        
        score = 0.0
        
        # Calculate potential casualties for each action
        if action == 0:  # maintain_course
            casualties = num_ped_straight
            passenger_risk = 0.0
        elif action == 1:  # brake_hard
            # At high speed, braking may not help enough
            # Assume 50% reduction in casualties if time permits
            if velocity > 15:
                casualties = num_ped_straight * 0.8  # Still likely to hit most
            else:
                casualties = num_ped_straight * 0.3  # Can stop in time
            passenger_risk = 0.0
        elif action == 2:  # swerve_left
            casualties = num_ped_left
            passenger_risk = num_passengers * 0.2  # Swerving is risky
        elif action == 3:  # swerve_right
            casualties = num_ped_right
            passenger_risk = num_passengers * 0.3  # Higher risk (wall/obstacle)
        elif action == 4:  # accelerate
            casualties = num_ped_straight * 1.5  # Even worse
            passenger_risk = 0.0
        
        # Utilitarian scoring: minimize total casualties (pedestrians + passengers at risk)
        total_harm = casualties + passenger_risk
        score = -total_harm
        
        return score


class KantianScorer:
    """
    Kantian ethics: Treat people as ends in themselves, never merely as means.
    
    Core principles:
    1. Universal Law: Would this action be acceptable if everyone did it?
    2. Humanity: Never treat people merely as means to an end
    3. Autonomy: Respect rational agency
    """
    
    @staticmethod
    def score_action(state: np.ndarray, action: int) -> float:
        """
        Score action based on Kantian principles.
        
        Args:
            state: 40-dim state vector
            action: Action taken (0-4)
        
        Returns:
            Kantian score (higher = more ethical)
        """
        velocity = state[0]
        num_passengers = state[1]
        num_ped_straight = state[4]
        num_ped_left = state[5]
        num_ped_right = state[6]
        
        score = 0.0
        
        # Kantian principle: Never intentionally harm (use people as means)
        # Key distinction: Foreseen side-effect vs. intended means
        
        # Doctrine of Double Effect:
        # - Allowing harm as side-effect ≠ Intending harm as means
        
        if action == 0 and num_ped_straight > 0:
            # Maintaining course: Harm is foreseen but not intended as means
            # Still problematic but less severe than actively turning into someone
            score = -50  # Serious but not categorical prohibition
        
        elif action == 2 and num_ped_left > 0:
            # Actively swerving INTO pedestrian = using them as means to save others
            score = -100  # Categorical prohibition
        
        elif action == 3 and num_ped_right > 0:
            # Swerving into pedestrian on right = using as means
            score = -100
        
        elif action == 1:
            # Braking: Attempting to harm no one
            # But at high speed may not work
            if velocity > 15 and num_ped_straight > 0:
                score = 5  # Good intention but may still cause harm
            else:
                score = 10  # Best option if it works
        
        elif action == 4 and num_ped_straight > 0:
            # Accelerating into pedestrians = extreme violation
            score = -200
        
        elif action == 3 and num_ped_right == 0:
            # Swerving to avoid all pedestrians (hitting wall instead)
            # Sacrificing passengers to save pedestrians = complex
            # Not using anyone as means, but putting passengers at great risk
            score = -10  # Problematic but not categorical
        
        else:
            # No direct harm = permissible
            score = 0
        
        return score


class VirtueEthicsScorer:
    """
    Virtue ethics: What would a virtuous agent do?
    
    Virtues considered:
    - Courage: Face danger appropriately
    - Temperance: Moderation in action
    - Justice: Fair treatment
    - Practical wisdom: Good judgment
    """
    
    @staticmethod
    def score_action(state: np.ndarray, action: int) -> float:
        """
        Score action based on virtue ethics.
        
        Args:
            state: 40-dim state vector
            action: Action taken (0-4)
        
        Returns:
            Virtue score (higher = more virtuous)
        """
        velocity = state[0]
        num_passengers = state[1]
        num_ped_straight = state[4]
        num_ped_left = state[5]
        num_ped_right = state[6]
        
        score = 0.0
        
        # Practical Wisdom: Choose appropriate action for context
        if num_ped_straight > 0:
            if action == 1:  # Brake
                score += 10  # Wise response to danger
            elif action == 4:  # Accelerate
                score -= 20  # Reckless, not virtuous
        
        # Courage: Face danger appropriately (not recklessly or cowardly)
        if action == 2 or action == 3:  # Swerve
            if num_ped_straight > 2 and (num_ped_left == 0 or num_ped_right == 0):
                score += 5  # Courageous to act when necessary
            else:
                score -= 5  # Reckless if unnecessary
        
        # Justice: Fair consideration of all parties
        total_people_at_risk = num_ped_straight + num_ped_left + num_ped_right
        if total_people_at_risk > 0 and action == 1:
            score += 8  # Just to protect vulnerable
        
        # Temperance: Moderation in speed and action
        if velocity > 30 and action == 4:
            score -= 10  # Intemperate speeding
        elif velocity > 25 and action == 1:
            score += 5  # Temperate reduction of speed
        
        return score


class CombinedEthicalScorer:
    """
    Combines multiple ethical theories with weights.
    """
    
    def __init__(
        self,
        utilitarian_weight: float = 0.4,
        kantian_weight: float = 0.4,
        virtue_weight: float = 0.2
    ):
        """
        Initialize with theory weights.
        
        Args:
            utilitarian_weight: Weight for utilitarian scoring
            kantian_weight: Weight for Kantian scoring
            virtue_weight: Weight for virtue ethics scoring
        """
        self.weights = {
            'utilitarian': utilitarian_weight,
            'kantian': kantian_weight,
            'virtue': virtue_weight
        }
        
        self.scorers = {
            'utilitarian': UtilitarianScorer(),
            'kantian': KantianScorer(),
            'virtue': VirtueEthicsScorer()
        }
    
    def score_action(self, state: np.ndarray, action: int) -> Dict[str, float]:
        """
        Get scores from all theories.
        
        Returns:
            Dictionary with individual scores and combined score
        """
        scores = {}
        
        for theory, scorer in self.scorers.items():
            scores[theory] = scorer.score_action(state, action)
        
        # Combined weighted score
        scores['combined'] = sum(
            scores[theory] * self.weights[theory]
            for theory in self.scorers.keys()
        )
        
        return scores
    
    def compare_actions(
        self,
        state: np.ndarray,
        actions: List[int] = [0, 1, 2, 3, 4]
    ) -> Dict:
        """
        Compare all actions according to each theory.
        
        Returns:
            Dictionary mapping actions to their scores under each theory
        """
        comparison = {action: {} for action in actions}
        
        for action in actions:
            scores = self.score_action(state, action)
            comparison[action] = scores
        
        return comparison


# Example usage
if __name__ == "__main__":
    # Test scenario: Pedestrian ahead
    state = np.array([
        20.0,  # velocity
        2.0,   # passengers
        0.0, 0.0,
        1.0,   # 1 pedestrian ahead
        0.0, 0.0
    ] + [0.0] * 33)
    
    scorer = CombinedEthicalScorer()
    
    print("Scenario: Pedestrian ahead, 2 passengers, 20 m/s")
    print("\nEthical Analysis:")
    print("-" * 60)
    
    actions = {
        0: 'maintain_course',
        1: 'brake_hard',
        2: 'swerve_left',
        3: 'swerve_right',
        4: 'accelerate'
    }
    
    for action_code, action_name in actions.items():
        scores = scorer.score_action(state, action_code)
        print(f"\n{action_name.upper()}:")
        print(f"  Utilitarian: {scores['utilitarian']:7.2f}")
        print(f"  Kantian:     {scores['kantian']:7.2f}")
        print(f"  Virtue:      {scores['virtue']:7.2f}")
        print(f"  COMBINED:    {scores['combined']:7.2f}")