"""
Simulation vs CARLA Comparison
Compare performance and analyze differences
"""

import numpy as np
import matplotlib.pyplot as plt
from env_hybrid import VehicleEnvironmentHybrid
from agent import QLearningAgent
from human import HumanFeedback
from config import *


class EnvironmentComparison:
    """Compare simulation vs CARLA environments"""
    
    def __init__(self):
        self.results = {}
    
    def run_single_episode(self, env, agent, human_feedback, num_steps=200):
        """Run single episode and collect detailed stats"""
        state = env.reset()
        rewards = []
        actions = []
        states = []
        
        for step in range(num_steps):
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            
            emotional_reward = human_feedback.compute_emotional_reward(
                state, action
            )
            total_reward = reward + emotional_reward
            
            rewards.append(total_reward)
            actions.append(action)
            states.append(state)
            
            state = next_state
            if done:
                break
        
        return {
            'total_reward': sum(rewards),
            'avg_reward': np.mean(rewards),
            'rewards': rewards,
            'actions': actions,
            'states': states,
            'steps': len(rewards)
        }
    
    def compare_environments(self):
        """Compare SIM vs CARLA"""
        print("\n" + "="*70)
        print("ENVIRONMENT COMPARISON: SIMULATION vs CARLA")
        print("="*70)
        
        agent = QLearningAgent()
        human_feedback = HumanFeedback()
        
        # Pre-train agents quickly
        print("\n[Pre-training agents...]")
        
        for mode in ['sim', 'carla']:
            print(f"\nTraining in {mode.upper()} mode (50 episodes)...")
            env = VehicleEnvironmentHybrid(mode=mode)
            
            for episode in range(50):
                state = env.reset()
                
                for step in range(100):
                    action = agent.select_action(state)
                    next_state, reward, done = env.step(action)
                    
                    emotional_reward = human_feedback.compute_emotional_reward(
                        state, action
                    )
                    total_reward = reward + emotional_reward
                    
                    agent.update_q_value(state, action, total_reward, next_state)
                    
                    state = next_state
                    if done:
                        break
                
                if (episode + 1) % 10 == 0:
                    print(f"  Episode {episode+1}/50 complete")
            
            env.close()
        
        # Test episodes
        print("\n[Testing in both environments...]")
        test_episodes = 10
        
        results = {}
        for mode in ['sim', 'carla']:
            print(f"\nTesting {mode.upper()} mode ({test_episodes} episodes)...")
            env = VehicleEnvironmentHybrid(mode=mode)
            
            episodes_data = []
            for episode in range(test_episodes):
                ep_data = self.run_single_episode(env, agent, human_feedback)
                episodes_data.append(ep_data)
                print(f"  Episode {episode+1:2d} | "
                      f"Total Reward: {ep_data['total_reward']:8.2f} | "
                      f"Steps: {ep_data['steps']}")
            
            results[mode] = episodes_data
            env.close()
        
        # Analysis
        print("\n" + "="*70)
        print("ANALYSIS")
        print("="*70)
        
        for mode in ['sim', 'carla']:
            rewards = [ep['total_reward'] for ep in results[mode]]
            steps = [ep['steps'] for ep in results[mode]]
            
            print(f"\n{mode.upper()}:")
            print(f"  Avg Total Reward:  {np.mean(rewards):8.2f} "
                  f"(±{np.std(rewards):6.2f})")
            print(f"  Avg Episode Length: {np.mean(steps):8.2f} "
                  f"(±{np.std(steps):6.2f})")
            print(f"  Min/Max Reward:    {np.min(rewards):8.2f} / "
                  f"{np.max(rewards):8.2f}")
        
        # Comparison
        sim_rewards = [ep['total_reward'] for ep in results['sim']]
        carla_rewards = [ep['total_reward'] for ep in results.get('carla', sim_rewards)]
        
        mean_diff = np.mean(carla_rewards) - np.mean(sim_rewards)
        pct_diff = (mean_diff / abs(np.mean(sim_rewards))) * 100
        
        print(f"\nCARLA vs SIMULATION:")
        print(f"  Reward Difference:  {mean_diff:+8.2f}")
        print(f"  Percentage Change:  {pct_diff:+8.2f}%")
        print(f"  Realism Factor:     {'CARLA is more challenging' if mean_diff < 0 else 'CARLA is easier'}")
        
        print("\n" + "="*70 + "\n")
        
        return results
    
    def generate_comparison_plot(self, results):
        """Generate comparison plots"""
        print("[Generating comparison plots...]")
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Simulation vs CARLA Comparison', fontsize=14, fontweight='bold')
        
        # 1. Reward distribution
        ax = axes[0, 0]
        for mode in ['sim', 'carla']:
            if mode in results:
                rewards = [ep['total_reward'] for ep in results[mode]]
                ax.hist(rewards, alpha=0.6, label=mode.upper(), bins=8)
        
        ax.set_xlabel('Total Reward')
        ax.set_ylabel('Frequency')
        ax.set_title('Reward Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Episode length comparison
        ax = axes[0, 1]
        for mode in ['sim', 'carla']:
            if mode in results:
                steps = [ep['steps'] for ep in results[mode]]
                ax.bar(mode.upper(), np.mean(steps), 
                       yerr=np.std(steps), capsize=5, alpha=0.7)
        
        ax.set_ylabel('Average Steps per Episode')
        ax.set_title('Episode Duration')
        ax.grid(True, alpha=0.3, axis='y')
        
        # 3. Statistics comparison
        ax = axes[1, 0]
        ax.axis('off')
        
        stats_text = "STATISTICS\n" + "="*40 + "\n\n"
        
        for mode in ['sim', 'carla']:
            if mode in results:
                rewards = [ep['total_reward'] for ep in results[mode]]
                stats_text += f"{mode.upper()}\n"
                stats_text += f"  Mean:    {np.mean(rewards):8.2f}\n"
                stats_text += f"  Std Dev: {np.std(rewards):8.2f}\n"
                stats_text += f"  Min:     {np.min(rewards):8.2f}\n"
                stats_text += f"  Max:     {np.max(rewards):8.2f}\n\n"
        
        ax.text(0.1, 0.9, stats_text, transform=ax.transAxes,
                fontfamily='monospace', verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 4. Episode progression
        ax = axes[1, 1]
        for mode in ['sim', 'carla']:
            if mode in results:
                rewards = [ep['total_reward'] for ep in results[mode]]
                ax.plot(range(1, len(rewards)+1), rewards, 'o-', 
                        label=mode.upper(), alpha=0.7)
        
        ax.set_xlabel('Episode')
        ax.set_ylabel('Total Reward')
        ax.set_title('Performance Over Episodes')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('comparison_sim_vs_carla.png', dpi=150, bbox_inches='tight')
        print("✓ Comparison plot saved: comparison_sim_vs_carla.png")
        plt.close()


def main():
    """Run complete comparison"""
    comparator = EnvironmentComparison()
    
    try:
        results = comparator.compare_environments()
        comparator.generate_comparison_plot(results)
        print("\n✓ Comparison complete!")
        
    except Exception as e:
        print(f"\n✗ Comparison failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
