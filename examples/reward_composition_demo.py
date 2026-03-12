"""
Reward Composition Demo
=======================

This demonstrates how the three reward components combine:
1. R_base: Safety rewards (collision avoidance, speed limits)
2. R_learned: Learned from human preferences
3. R_explicit: Explicit ethical theories (utilitarian, Kantian, virtue)

Shows how different ethical theories produce different decisions.
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
from utils.ethical_theories import CombinedEthicalScorer, UtilitarianScorer, KantianScorer, VirtueEthicsScorer


def create_trolley_problem_scenario():
    """
    Classic trolley problem:
    - Straight ahead: 5 pedestrians
    - Swerve left: 1 pedestrian
    - Swerve right: Wall (safe for pedestrians, risky for passengers)
    """
    state = np.array([
        20.0,  # velocity (20 m/s)
        3.0,   # 3 passengers in car
        0.0,   # lane_position (centered)
        0.0,   # velocity_delta
        5.0,   # 5 pedestrians straight ahead!
        1.0,   # 1 pedestrian left
        0.0,   # 0 pedestrians right (but wall)
    ] + [0.0] * 33)
    
    return state


def simulate_base_safety_reward(state, action):
    """
    Simulate base safety rewards (R_base).
    
    Simple heuristics:
    - Collision = -10 per person
    - Risky maneuver = -2 to -5
    - At high speed, braking may not prevent collision
    """
    velocity = state[0]
    num_passengers = state[1]
    num_ped_straight = state[4]
    num_ped_left = state[5]
    num_ped_right = state[6]
    
    if action == 0 and num_ped_straight > 0:  # maintain into pedestrians
        return -10.0 * num_ped_straight
    elif action == 1:  # brake
        # At high speed (>15 m/s), braking won't stop in time
        if velocity > 15 and num_ped_straight > 0:
            return -10.0 * num_ped_straight * 0.7  # Still hit most of them
        else:
            return 5.0  # Can stop safely
    elif action == 2 and num_ped_left > 0:  # swerve left into pedestrian
        return -10.0 * num_ped_left - 2.0  # Also risky for passengers
    elif action == 2 and num_ped_left == 0:  # swerve left safely
        return -1.0  # Slight risk to passengers
    elif action == 3 and num_ped_right > 0:  # swerve right into pedestrian
        return -10.0 * num_ped_right - 3.0
    elif action == 3 and num_ped_right == 0:  # swerve right into wall
        return -8.0  # Serious risk for passengers (wall collision)
    elif action == 4 and num_ped_straight > 0:  # accelerate into pedestrians
        return -20.0 * num_ped_straight
    else:
        return 0.0


def simulate_learned_reward(state, action):
    """
    Simulate learned reward from human preferences (R_learned).
    
    In reality, this comes from the reward model trained on preferences.
    Here we simulate what it might learn if humans were utilitarian-leaning.
    """
    velocity = state[0]
    num_ped_straight = state[4]
    num_ped_left = state[5]
    num_ped_right = state[6]
    
    # Assume humans taught it to minimize casualties
    # But also value passenger safety
    
    if action == 1:  # brake
        # Humans like braking attempts, but recognize it may not work at high speed
        if velocity > 15 and num_ped_straight > 0:
            return -3.0  # Good intention but won't save them
        else:
            return 8.0  # Best option if it works
    elif action == 0:
        return -5.0 * num_ped_straight
    elif action == 2:
        # Swerve left: humans might accept this if it saves more lives
        if num_ped_left < num_ped_straight:
            return 3.0  # Lesser of two evils
        else:
            return -5.0 * num_ped_left
    elif action == 3:
        # Swerve right to wall: risky for passengers but saves pedestrians
        if num_ped_right == 0 and num_ped_straight > 0:
            return 1.0  # Saves pedestrians, but endangers passengers
        else:
            return -5.0
    else:
        return -10.0


def demo_trolley_problem():
    """Show how different theories handle the trolley problem."""
    
    print("\n" + "=" * 70)
    print("TROLLEY PROBLEM SCENARIO")
    print("=" * 70)
    print("\nSituation:")
    print("  - 5 pedestrians directly ahead")
    print("  - 1 pedestrian to the left")
    print("  - Wall to the right (safe for pedestrians, risky for 3 passengers)")
    print("  - Traveling at 20 m/s")
    
    state = create_trolley_problem_scenario()
    
    # Initialize scorers
    util_scorer = UtilitarianScorer()
    kant_scorer = KantianScorer()
    virtue_scorer = VirtueEthicsScorer()
    
    actions = {
        0: 'maintain_course (hit 5)',
        1: 'brake_hard (try to stop)',
        2: 'swerve_left (hit 1)',
        3: 'swerve_right (hit wall)',
        4: 'accelerate (hit 5 harder)'
    }
    
    print("\n" + "=" * 70)
    print("ANALYSIS BY ETHICAL THEORY")
    print("=" * 70)
    
    # Show each action's scores
    for action_code, action_name in actions.items():
        print(f"\n{action_name.upper()}")
        print("-" * 70)
        
        # Base safety
        r_base = simulate_base_safety_reward(state, action_code)
        print(f"  R_base (safety):     {r_base:7.2f}")
        
        # Learned from humans
        r_learned = simulate_learned_reward(state, action_code)
        print(f"  R_learned (human):   {r_learned:7.2f}")
        
        # Explicit theories
        r_util = util_scorer.score_action(state, action_code)
        r_kant = kant_scorer.score_action(state, action_code)
        r_virtue = virtue_scorer.score_action(state, action_code)
        
        print(f"  R_utilitarian:       {r_util:7.2f}")
        print(f"  R_kantian:           {r_kant:7.2f}")
        print(f"  R_virtue:            {r_virtue:7.2f}")
        
        # Combined (different weightings)
        r_combined_default = 0.3 * r_base + 0.7 * r_learned
        r_combined_explicit = 0.3 * r_base + 0.4 * r_learned + 0.3 * (0.4*r_util + 0.4*r_kant + 0.2*r_virtue)
        
        print(f"  Combined (default):  {r_combined_default:7.2f}")
        print(f"  Combined (explicit): {r_combined_explicit:7.2f}")
    
    # Show best actions by theory
    print("\n" + "=" * 70)
    print("RECOMMENDED ACTIONS BY THEORY")
    print("=" * 70)
    
    best_by_theory = {}
    for theory_name, scorer in [
        ('Utilitarian', util_scorer),
        ('Kantian', kant_scorer),
        ('Virtue Ethics', virtue_scorer)
    ]:
        scores = {action: scorer.score_action(state, action) for action in range(5)}
        best_action = max(scores, key=scores.get)
        best_by_theory[theory_name] = (best_action, actions[best_action], scores[best_action])
    
    for theory, (action_code, action_name, score) in best_by_theory.items():
        print(f"\n{theory}:")
        print(f"  Recommends: {action_name}")
        print(f"  Score: {score:.2f}")
    
    # Combined recommendation
    combined_scorer = CombinedEthicalScorer()
    combined_scores = {action: combined_scorer.score_action(state, action)['combined'] 
                      for action in range(5)}
    best_combined = max(combined_scores, key=combined_scores.get)
    
    print(f"\nCombined Approach:")
    print(f"  Recommends: {actions[best_combined]}")
    print(f"  Score: {combined_scores[best_combined]:.2f}")


def demo_reward_composition():
    """Show how reward weights affect decisions."""
    
    print("\n" + "=" * 70)
    print("REWARD COMPOSITION: How Weights Affect Decisions")
    print("=" * 70)
    
    state = create_trolley_problem_scenario()
    action = 2  # Swerve left (utilitarian choice: hit 1 instead of 5)
    
    r_base = simulate_base_safety_reward(state, action)
    r_learned = simulate_learned_reward(state, action)
    
    scorer = CombinedEthicalScorer()
    explicit_scores = scorer.score_action(state, action)
    r_explicit = explicit_scores['combined']
    
    print(f"\nAction: Swerve Left (hit 1 pedestrian instead of 5)")
    print("-" * 70)
    print(f"R_base (safety):     {r_base:7.2f}")
    print(f"R_learned (human):   {r_learned:7.2f}")
    print(f"R_explicit (theory): {r_explicit:7.2f}")
    
    print("\nDifferent Weight Configurations:")
    print("-" * 70)
    
    configurations = [
        ("Default (α=0.3, β=0.7)", 0.3, 0.7, 0.0),
        ("Safety-focused (α=0.6, β=0.4)", 0.6, 0.4, 0.0),
        ("Human-focused (α=0.2, β=0.8)", 0.2, 0.8, 0.0),
        ("With explicit ethics", 0.3, 0.4, 0.3),
        ("Pure utilitarian", 0.0, 0.0, 1.0),
    ]
    
    for name, alpha, beta, gamma in configurations:
        if gamma > 0:
            total = alpha * r_base + beta * r_learned + gamma * r_explicit
        else:
            total = alpha * r_base + beta * r_learned
        
        print(f"\n{name}:")
        print(f"  R_total = {alpha}*{r_base:.2f} + {beta}*{r_learned:.2f}", end="")
        if gamma > 0:
            print(f" + {gamma}*{r_explicit:.2f}", end="")
        print(f" = {total:.2f}")


def demo_theory_comparison():
    """Compare how different ethical theories score the same action."""
    
    print("\n" + "=" * 70)
    print("ETHICAL THEORY COMPARISON")
    print("=" * 70)
    
    state = create_trolley_problem_scenario()
    
    print("\nScenario: 5 people ahead, 1 to the left")
    print("Action being evaluated: SWERVE LEFT (hit 1 instead of 5)")
    print("-" * 70)
    
    util_scorer = UtilitarianScorer()
    kant_scorer = KantianScorer()
    virtue_scorer = VirtueEthicsScorer()
    
    action = 2  # Swerve left
    
    util_score = util_scorer.score_action(state, action)
    kant_score = kant_scorer.score_action(state, action)
    virtue_score = virtue_scorer.score_action(state, action)
    
    print("\nUtilitarianism (Maximize welfare, minimize harm):")
    print(f"  Score: {util_score:.2f}")
    print(f"  Reasoning: Killing 1 person produces less harm than killing 5")
    print(f"  Conclusion: {'ACCEPTABLE' if util_score > 0 else 'UNACCEPTABLE'}")
    
    print("\nKantian Ethics (Never treat people merely as means):")
    print(f"  Score: {kant_score:.2f}")
    print(f"  Reasoning: Deliberately swerving into someone uses them as a means")
    print(f"  Conclusion: {'ACCEPTABLE' if kant_score > -50 else 'CATEGORICALLY WRONG'}")
    
    print("\nVirtue Ethics (What would a virtuous person do?):")
    print(f"  Score: {virtue_score:.2f}")
    print(f"  Reasoning: Considers courage, practical wisdom, and justice")
    print(f"  Conclusion: {'VIRTUOUS' if virtue_score > 0 else 'NOT VIRTUOUS'}")
    
    print("\nKey Difference:")
    print("  Utilitarian: Focuses on OUTCOMES (fewer deaths)")
    print("  Kantian:     Focuses on INTENTIONS (never use people as tools)")
    print("  Virtue:      Focuses on CHARACTER (what virtues does action display?)")


def main():
    """Run all demos."""
    
    print("\n" + "=" * 70)
    print("ETHICAL REWARD COMPOSITION DEMONSTRATION")
    print("=" * 70)
    
    demo_trolley_problem()
    demo_reward_composition()
    demo_theory_comparison()
    
    print("\n" + "=" * 70)
    print("KEY TAKEAWAYS")
    print("=" * 70)
    print("""
1. THREE REWARD COMPONENTS:
   • R_base: Safety heuristics (collision avoidance, speed limits)
   • R_learned: Learned from human preference labels
   • R_explicit: Explicit ethical theories (utilitarian, Kantian, virtue)

2. DIFFERENT THEORIES GIVE DIFFERENT ANSWERS:
   • Utilitarian: Minimize total harm → Hit 1 instead of 5
   • Kantian: Never use people as means → Brake, don't swerve into anyone
   • Virtue: What's wisest/most courageous? → Context-dependent

3. YOUR SYSTEM CAN USE:
   • Just learned (default): Ethics implicit in human labels
   • Learned + explicit: Combine human intuition with formal theories
   • Custom weights: Adjust α, β, γ to match your ethical priorities

4. HUMAN PREFERENCES ENCODE ETHICS:
   • When humans label "A is more ethical than B"
   • They're implicitly applying their ethical framework
   • Reward model learns these patterns
   • Can be utilitarian, Kantian, or mixed depending on annotators
    """)
    
    print("=" * 70)
    print("\nTo use explicit ethics in training:")
    print("  config = Config()")
    print("  config.use_explicit_ethics = True")
    print("  trainer = RLTrainer(config)")
    print("  trainer.train()")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()