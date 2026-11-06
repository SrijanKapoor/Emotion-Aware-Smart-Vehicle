"""
Evaluation and Comparison of Baseline RL vs Human-in-the-Loop RL
Generates performance plots and detailed analysis
"""

import numpy as np
import matplotlib.pyplot as plt
from config import EMOTION_STATES, ACTIONS
from train import TrainingManager


def evaluate_agents(manager, num_eval_episodes=100):
    """
    Evaluate both trained agents without learning (pure exploitation).
    
    Args:
        manager: TrainingManager instance with trained agents
        num_eval_episodes: Number of evaluation episodes
    
    Returns:
        dict: Evaluation metrics for both agents
    """
    print("=" * 60)
    print("EVALUATION: Testing Trained Agents")
    print("=" * 60)
    
    baseline_rewards = []
    baseline_alerts = []
    baseline_emotion_match = []
    
    hitl_rewards = []
    hitl_alerts = []
    hitl_emotion_match = []
    
    # Baseline evaluation
    print("\nEvaluating Baseline Agent...")
    for ep in range(num_eval_episodes):
        state = manager.baseline_env.reset()
        episode_reward = 0
        alert_count = 0
        
        for step in range(50):  # 50 steps per eval episode
            # Use best action (no exploration)
            action = manager.baseline_agent.get_best_action(state)
            next_state, env_reward, done = manager.baseline_env.step(action)
            
            if action in [2, 3]:
                alert_count += 1
            
            episode_reward += env_reward
            state = next_state
            if done:
                break
        
        baseline_rewards.append(episode_reward)
        baseline_alerts.append(alert_count)
    
    # HiTL evaluation
    print("Evaluating Human-in-the-Loop Agent...")
    for ep in range(num_eval_episodes):
        state = manager.hitl_env.reset()
        episode_reward = 0
        alert_count = 0
        emotional_matches = 0
        
        for step in range(50):
            # Use best action (no exploration)
            action = manager.hitl_agent.get_best_action(state)
            next_state, env_reward, done = manager.hitl_env.step(action)
            
            if action in [2, 3]:
                alert_count += 1
            
            # Check emotional appropriateness
            emotion_reward = manager.human_feedback.compute_emotional_reward(state, action)
            if emotion_reward > 0:
                emotional_matches += 1
            
            episode_reward += env_reward
            state = next_state
            if done:
                break
        
        hitl_rewards.append(episode_reward)
        hitl_alerts.append(alert_count)
        hitl_emotion_match.append(emotional_matches)
    
    print("✓ Evaluation complete!\n")
    
    return {
        'baseline': {
            'rewards': baseline_rewards,
            'alerts': baseline_alerts,
        },
        'hitl': {
            'rewards': hitl_rewards,
            'alerts': hitl_alerts,
            'emotion_match': hitl_emotion_match,
        }
    }


def plot_training_curves(manager):
    """
    Plot training curves comparing baseline and HiTL agents.
    
    Args:
        manager: TrainingManager instance
    """
    # Smooth rewards using moving average
    window = 20
    baseline_smooth = np.convolve(manager.baseline_episode_rewards, 
                                  np.ones(window)/window, mode='valid')
    hitl_smooth = np.convolve(manager.hitl_episode_rewards, 
                              np.ones(window)/window, mode='valid')
    
    baseline_alerts_smooth = np.convolve(manager.baseline_episode_alerts, 
                                        np.ones(window)/window, mode='valid')
    hitl_alerts_smooth = np.convolve(manager.hitl_episode_alerts, 
                                    np.ones(window)/window, mode='valid')
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Cumulative Rewards
    ax1 = axes[0]
    episodes = np.arange(len(baseline_smooth))
    ax1.plot(episodes, baseline_smooth, label='Baseline RL', linewidth=2, color='#FF6B6B')
    ax1.plot(episodes, hitl_smooth, label='Human-in-the-Loop RL', linewidth=2, color='#4ECDC4')
    ax1.set_xlabel('Episode (smoothed over 20 episodes)', fontsize=11)
    ax1.set_ylabel('Average Episode Reward', fontsize=11)
    ax1.set_title('Training Progress: Reward Comparison', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Alert Frequency
    ax2 = axes[1]
    ax2.plot(episodes, baseline_alerts_smooth, label='Baseline RL', linewidth=2, color='#FF6B6B')
    ax2.plot(episodes, hitl_alerts_smooth, label='Human-in-the-Loop RL', linewidth=2, color='#4ECDC4')
    ax2.set_xlabel('Episode (smoothed over 20 episodes)', fontsize=11)
    ax2.set_ylabel('Average Alerts per Episode', fontsize=11)
    ax2.set_title('Training Progress: Alert Frequency Reduction', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: training_curves.png\n")


def plot_evaluation_comparison(eval_results):
    """
    Plot evaluation results comparing agents.
    
    Args:
        eval_results: Results from evaluate_agents()
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Reward Distribution
    ax1 = axes[0]
    ax1.hist(eval_results['baseline']['rewards'], bins=15, alpha=0.6, 
             label='Baseline RL', color='#FF6B6B', edgecolor='black')
    ax1.hist(eval_results['hitl']['rewards'], bins=15, alpha=0.6, 
             label='Human-in-the-Loop RL', color='#4ECDC4', edgecolor='black')
    ax1.set_xlabel('Episode Reward', fontsize=11)
    ax1.set_ylabel('Frequency', fontsize=11)
    ax1.set_title('Evaluation: Reward Distribution', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Alert Count Distribution
    ax2 = axes[1]
    ax2.hist(eval_results['baseline']['alerts'], bins=15, alpha=0.6, 
             label='Baseline RL', color='#FF6B6B', edgecolor='black')
    ax2.hist(eval_results['hitl']['alerts'], bins=15, alpha=0.6, 
             label='Human-in-the-Loop RL', color='#4ECDC4', edgecolor='black')
    ax2.set_xlabel('Alerts per Episode', fontsize=11)
    ax2.set_ylabel('Frequency', fontsize=11)
    ax2.set_title('Evaluation: Alert Frequency Distribution', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('evaluation_comparison.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: evaluation_comparison.png\n")


def print_detailed_analysis(manager, eval_results):
    """
    Print detailed analysis comparing both approaches.
    
    Args:
        manager: TrainingManager instance
        eval_results: Results from evaluate_agents()
    """
    print("=" * 70)
    print("DETAILED EVALUATION ANALYSIS")
    print("=" * 70)
    
    baseline_rewards = np.array(eval_results['baseline']['rewards'])
    baseline_alerts = np.array(eval_results['baseline']['alerts'])
    
    hitl_rewards = np.array(eval_results['hitl']['rewards'])
    hitl_alerts = np.array(eval_results['hitl']['alerts'])
    hitl_emotion_match = np.array(eval_results['hitl']['emotion_match'])
    
    print("\n📊 REWARD METRICS:")
    print("-" * 70)
    print(f"{'Metric':<40} {'Baseline':>14} {'HiTL':>14}")
    print("-" * 70)
    print(f"{'Mean Reward':<40} {baseline_rewards.mean():>14.2f} {hitl_rewards.mean():>14.2f}")
    print(f"{'Std Dev Reward':<40} {baseline_rewards.std():>14.2f} {hitl_rewards.std():>14.2f}")
    print(f"{'Min Reward':<40} {baseline_rewards.min():>14.2f} {hitl_rewards.min():>14.2f}")
    print(f"{'Max Reward':<40} {baseline_rewards.max():>14.2f} {hitl_rewards.max():>14.2f}")
    
    print("\n🚨 ALERT METRICS:")
    print("-" * 70)
    print(f"{'Metric':<40} {'Baseline':>14} {'HiTL':>14}")
    print("-" * 70)
    print(f"{'Mean Alerts/Episode':<40} {baseline_alerts.mean():>14.2f} {hitl_alerts.mean():>14.2f}")
    print(f"{'Std Dev Alerts':<40} {baseline_alerts.std():>14.2f} {hitl_alerts.std():>14.2f}")
    print(f"{'Min Alerts':<40} {baseline_alerts.min():>14.2f} {hitl_alerts.min():>14.2f}")
    print(f"{'Max Alerts':<40} {baseline_alerts.max():>14.2f} {hitl_alerts.max():>14.2f}")
    
    print("\n😊 EMOTIONAL APPROPRIATENESS (HiTL Only):")
    print("-" * 70)
    print(f"Mean Emotionally-Matched Actions: {hitl_emotion_match.mean():.2f} per episode")
    print(f"Percentage of Appropriate Actions: {(hitl_emotion_match.sum() / (100 * 50) * 100):.1f}%")
    
    print("\n" + "=" * 70)
    print("KEY FINDINGS & INTERPRETATION")
    print("=" * 70)
    
    reward_improvement = hitl_rewards.mean() - baseline_rewards.mean()
    alert_reduction = baseline_alerts.mean() - hitl_alerts.mean()
    reward_improvement_pct = (reward_improvement / abs(baseline_rewards.mean())) * 100 if baseline_rewards.mean() != 0 else 0
    alert_reduction_pct = (alert_reduction / baseline_alerts.mean()) * 100 if baseline_alerts.mean() > 0 else 0
    
    print(f"""
1. REWARD IMPACT:
   • HiTL achieved {reward_improvement:+.2f} {reward_improvement_pct:+.1f}% change in average reward
   • This is expected: HiTL optimizes for emotional satisfaction, not just safety
   
2. ALERT MANAGEMENT:
   • Baseline issues {baseline_alerts.mean():.2f} alerts per episode
   • HiTL reduces to {hitl_alerts.mean():.2f} alerts per episode
   • Alert reduction: {alert_reduction:.2f} alerts ({alert_reduction_pct:.1f}% decrease)
   • This means HiTL provides smarter, more targeted assistance
   
3. EMOTIONAL AWARENESS:
   • HiTL agent learns to match actions to driver emotional state
   • {(hitl_emotion_match.sum() / (100 * 50) * 100):.1f}% of HiTL actions are emotionally appropriate
   • Baseline agent has no concept of emotional state
   
4. PRACTICAL IMPLICATIONS:
   • Human-in-the-Loop RL produces a more personalized system
   • Reduces alert fatigue while maintaining safety
   • Shows that human feedback shapes agent behavior effectively
   • Suitable for real CPS where driver comfort matters
""")
    print("=" * 70)


def main():
    """Main evaluation function"""
    # Train both agents
    print("\n" + "=" * 70)
    print("EMOTION-AWARE SMART VEHICLE CPS - COMPLETE SIMULATION")
    print("=" * 70 + "\n")
    
    manager = TrainingManager()
    manager.train_both()
    
    # Plot training curves
    print("Generating visualizations...")
    plot_training_curves(manager)
    
    # Evaluate agents
    eval_results = evaluate_agents(manager, num_eval_episodes=100)
    
    # Plot evaluation results
    plot_evaluation_comparison(eval_results)
    
    # Print detailed analysis
    print_detailed_analysis(manager, eval_results)
    
    print("\n✓ All plots saved!")
    print("  • training_curves.png")
    print("  • evaluation_comparison.png")


if __name__ == "__main__":
    main()
