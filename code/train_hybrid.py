"""
Hybrid Training: Train agents in SIMULATION or CARLA
Switch modes to compare performance
"""

import numpy as np
from env_hybrid import VehicleEnvironmentHybrid
from agent import QLearningAgent
from human import HumanFeedback
from config import *
import time


class HybridTrainingManager:
    """Train agents in either simulation or CARLA"""
    
    def __init__(self, mode='sim'):
        """
        Initialize training manager
        
        Args:
            mode: 'sim' or 'carla'
        """
        self.mode = mode
        self.env = VehicleEnvironmentHybrid(mode=mode)
        self.baseline_agent = QLearningAgent()
        self.hitl_agent = QLearningAgent()
        self.human_feedback = HumanFeedback()
        
        print(f"\n✓ Training manager initialized ({mode.upper()} mode)")
    
    def train_baseline(self, episodes=100):
        """Train baseline agent"""
        print(f"\n{'='*60}")
        print(f"BASELINE TRAINING ({self.mode.upper()}) - {episodes} episodes")
        print(f"{'='*60}")
        
        episode_rewards = []
        episode_alerts = []
        
        for episode in range(episodes):
            state = self.env.reset()
            ep_reward = 0
            ep_alerts = 0
            steps = 0
            
            while steps < 200:
                action = self.baseline_agent.select_action(state)
                next_state, reward, done = self.env.step(action)
                
                self.baseline_agent.update_q_value(
                    state, action, reward, next_state
                )
                
                ep_reward += reward
                if reward < 0:
                    ep_alerts += 1
                
                state = next_state
                steps += 1
                if done:
                    break
            
            self.baseline_agent.decay_epsilon()
            episode_rewards.append(ep_reward)
            episode_alerts.append(ep_alerts)
            
            if (episode + 1) % 20 == 0:
                avg_rew = np.mean(episode_rewards[-20:])
                print(f"Episode {episode+1:3d}/{episodes} | "
                      f"Avg Reward: {avg_rew:7.2f} | "
                      f"ε: {self.baseline_agent.epsilon:.3f}")
        
        print(f"✓ Baseline training complete")
        return episode_rewards, episode_alerts
    
    def train_hitl(self, episodes=100):
        """Train HiTL agent with emotional feedback"""
        print(f"\n{'='*60}")
        print(f"HiTL TRAINING ({self.mode.upper()}) - {episodes} episodes")
        print(f"{'='*60}")
        
        episode_rewards = []
        episode_alerts = []
        
        for episode in range(episodes):
            state = self.env.reset()
            ep_reward = 0
            ep_alerts = 0
            steps = 0
            
            while steps < 200:
                action = self.hitl_agent.select_action(state)
                next_state, reward, done = self.env.step(action)
                
                # Add emotional feedback
                emotional_reward = self.human_feedback.compute_emotional_reward(
                    state, action
                )
                total_reward = reward + emotional_reward
                
                self.hitl_agent.update_q_value(
                    state, action, total_reward, next_state
                )
                
                ep_reward += total_reward
                if total_reward < 0:
                    ep_alerts += 1
                
                state = next_state
                steps += 1
                if done:
                    break
            
            self.hitl_agent.decay_epsilon()
            episode_rewards.append(ep_reward)
            episode_alerts.append(ep_alerts)
            
            if (episode + 1) % 20 == 0:
                avg_rew = np.mean(episode_rewards[-20:])
                print(f"Episode {episode+1:3d}/{episodes} | "
                      f"Avg Reward: {avg_rew:7.2f} | "
                      f"ε: {self.hitl_agent.epsilon:.3f}")
        
        print(f"✓ HiTL training complete")
        return episode_rewards, episode_alerts
    
    def evaluate(self, episodes=50):
        """Evaluate both agents"""
        print(f"\n{'='*60}")
        print(f"EVALUATION ({self.mode.upper()}) - {episodes} episodes each")
        print(f"{'='*60}")
        
        # Baseline evaluation
        print("\n[Baseline Agent]")
        baseline_rewards = []
        for ep in range(episodes):
            state = self.env.reset()
            ep_reward = 0
            steps = 0
            
            while steps < 200:
                action = self.baseline_agent.select_action(state)
                next_state, reward, done = self.env.step(action)
                ep_reward += reward
                state = next_state
                steps += 1
                if done:
                    break
            
            baseline_rewards.append(ep_reward)
            if (ep + 1) % 10 == 0:
                print(f"  Episode {ep+1:2d}/{episodes} | Reward: {ep_reward:7.2f}")
        
        # HiTL evaluation
        print("\n[HiTL Agent]")
        hitl_rewards = []
        for ep in range(episodes):
            state = self.env.reset()
            ep_reward = 0
            steps = 0
            
            while steps < 200:
                action = self.hitl_agent.select_action(state)
                next_state, reward, done = self.env.step(action)
                
                emotional_reward = self.human_feedback.compute_emotional_reward(
                    state, action
                )
                ep_reward += reward + emotional_reward
                
                state = next_state
                steps += 1
                if done:
                    break
            
            hitl_rewards.append(ep_reward)
            if (ep + 1) % 10 == 0:
                print(f"  Episode {ep+1:2d}/{episodes} | Reward: {ep_reward:7.2f}")
        
        # Results
        print(f"\n{'='*60}")
        print(f"RESULTS - {self.mode.upper()}")
        print(f"{'='*60}")
        print(f"Baseline Avg: {np.mean(baseline_rewards):.2f} "
              f"(±{np.std(baseline_rewards):.2f})")
        print(f"HiTL Avg:     {np.mean(hitl_rewards):.2f} "
              f"(±{np.std(hitl_rewards):.2f})")
        
        improvement = (
            (np.mean(hitl_rewards) - np.mean(baseline_rewards)) / 
            abs(np.mean(baseline_rewards)) * 100
        )
        print(f"Improvement:  {improvement:+.1f}%")
        print(f"{'='*60}\n")
        
        return {
            'baseline': baseline_rewards,
            'hitl': hitl_rewards
        }
    
    def close(self):
        """Cleanup"""
        self.env.close()


def main():
    """Main training pipeline"""
    
    # Train in SIMULATION first (fast, no CARLA needed)
    print("\n" + "="*70)
    print("PHASE 1: SIMULATION TRAINING")
    print("="*70)
    
    sim_manager = HybridTrainingManager(mode='sim')
    sim_baseline_rew, sim_baseline_alerts = sim_manager.train_baseline(episodes=100)
    sim_hitl_rew, sim_hitl_alerts = sim_manager.train_hitl(episodes=100)
    sim_results = sim_manager.evaluate(episodes=50)
    sim_manager.close()
    
    # Train in CARLA if available
    print("\n" + "="*70)
    print("PHASE 2: CARLA TRAINING (Real Simulator)")
    print("="*70)
    
    try:
        carla_manager = HybridTrainingManager(mode='carla')
        carla_baseline_rew, carla_baseline_alerts = carla_manager.train_baseline(episodes=50)
        carla_hitl_rew, carla_hitl_alerts = carla_manager.train_hitl(episodes=50)
        carla_results = carla_manager.evaluate(episodes=20)
        carla_manager.close()
        
        # Compare
        print("\n" + "="*70)
        print("SIMULATION vs CARLA COMPARISON")
        print("="*70)
        print(f"Simulation HiTL Avg:  {np.mean(sim_results['hitl']):.2f}")
        print(f"CARLA HiTL Avg:       {np.mean(carla_results['hitl']):.2f}")
        print(f"Realism Impact:       "
              f"{np.mean(carla_results['hitl']) - np.mean(sim_results['hitl']):+.2f}")
        
    except Exception as e:
        print(f"\n⚠ CARLA training skipped: {e}")
        print("  (CARLA server may not be running)")


if __name__ == "__main__":
    main()
