"""
Training Loop for Baseline RL vs Human-in-the-Loop RL
Trains two agents: one without human feedback, one with emotional reward shaping
"""

import random
import numpy as np
from config import NUM_EPISODES, MAX_STEPS_PER_EPISODE, EVALUATION_INTERVAL, RANDOM_SEED
from env import VehicleEnvironment
from agent import QLearningAgent
from human import HumanFeedback


class TrainingManager:
    """
    Manages training of both baseline and Human-in-the-Loop agents.
    Tracks performance metrics for comparison.
    """
    
    def __init__(self, num_episodes=NUM_EPISODES, max_steps=MAX_STEPS_PER_EPISODE):
        """
        Initialize training setup.
        
        Args:
            num_episodes: Number of training episodes
            max_steps: Maximum steps per episode
        """
        # Set random seed for reproducibility
        random.seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)
        
        self.num_episodes = num_episodes
        self.max_steps = max_steps
        
        # Baseline RL agent (no human feedback)
        self.baseline_agent = QLearningAgent()
        self.baseline_env = VehicleEnvironment(seed=RANDOM_SEED)
        
        # Human-in-the-Loop agent (with emotional reward shaping)
        self.hitl_agent = QLearningAgent()
        self.hitl_env = VehicleEnvironment(seed=RANDOM_SEED)
        self.human_feedback = HumanFeedback()
        
        # Metrics tracking
        self.baseline_episode_rewards = []
        self.baseline_episode_alerts = []  # Track alert actions
        self.hitl_episode_rewards = []
        self.hitl_episode_alerts = []
    
    def train_baseline(self):
        """
        Train baseline agent WITHOUT human feedback.
        Only uses environment safety rewards.
        """
        print("=" * 60)
        print("TRAINING: Baseline RL (No Human Feedback)")
        print("=" * 60)
        
        for episode in range(self.num_episodes):
            state = self.baseline_env.reset()
            episode_reward = 0
            alert_count = 0  # Track high-alert and assistance actions
            
            for step in range(self.max_steps):
                # Agent chooses action
                action = self.baseline_agent.select_action(state)
                
                # Environment responds
                next_state, env_reward, done = self.baseline_env.step(action)
                
                # Track alerts (actions 2 and 3 are high-intensity)
                if action in [2, 3]:
                    alert_count += 1
                
                # Update Q-value (ONLY environment reward, no human feedback)
                total_reward = env_reward
                self.baseline_agent.update_q_value(state, action, total_reward, next_state)
                
                episode_reward += total_reward
                state = next_state
                
                if done:
                    break
            
            # Decay exploration rate
            self.baseline_agent.decay_epsilon()
            
            # Track metrics
            self.baseline_episode_rewards.append(episode_reward)
            self.baseline_episode_alerts.append(alert_count)
            
            # Print progress
            if (episode + 1) % EVALUATION_INTERVAL == 0:
                avg_reward = np.mean(self.baseline_episode_rewards[-EVALUATION_INTERVAL:])
                avg_alerts = np.mean(self.baseline_episode_alerts[-EVALUATION_INTERVAL:])
                print(f"Episode {episode + 1}/{self.num_episodes} | "
                      f"Avg Reward: {avg_reward:6.2f} | "
                      f"Avg Alerts: {avg_alerts:5.2f}")
        
        print("\n✓ Baseline training complete!\n")
    
    def train_hitl(self):
        """
        Train Human-in-the-Loop agent WITH human emotional feedback.
        Uses environment rewards + emotional reward shaping.
        """
        print("=" * 60)
        print("TRAINING: Human-in-the-Loop RL (With Emotional Feedback)")
        print("=" * 60)
        
        for episode in range(self.num_episodes):
            state = self.hitl_env.reset()
            episode_reward = 0
            alert_count = 0
            
            for step in range(self.max_steps):
                # Agent chooses action
                action = self.hitl_agent.select_action(state)
                
                # Environment responds
                next_state, env_reward, done = self.hitl_env.step(action)
                
                # Track alerts
                if action in [2, 3]:
                    alert_count += 1
                
                # Compute human emotional feedback reward
                emotional_reward = self.human_feedback.compute_emotional_reward(state, action)
                
                # Total reward = environment + human feedback
                total_reward = env_reward + emotional_reward
                
                # Update Q-value with combined reward
                self.hitl_agent.update_q_value(state, action, total_reward, next_state)
                
                episode_reward += total_reward
                state = next_state
                
                if done:
                    break
            
            # Decay exploration rate
            self.hitl_agent.decay_epsilon()
            
            # Track metrics
            self.hitl_episode_rewards.append(episode_reward)
            self.hitl_episode_alerts.append(alert_count)
            
            # Print progress
            if (episode + 1) % EVALUATION_INTERVAL == 0:
                avg_reward = np.mean(self.hitl_episode_rewards[-EVALUATION_INTERVAL:])
                avg_alerts = np.mean(self.hitl_episode_alerts[-EVALUATION_INTERVAL:])
                print(f"Episode {episode + 1}/{self.num_episodes} | "
                      f"Avg Reward: {avg_reward:6.2f} | "
                      f"Avg Alerts: {avg_alerts:5.2f}")
        
        print("\n✓ Human-in-the-Loop training complete!\n")
    
    def train_both(self):
        """
        Train both agents sequentially.
        """
        self.train_baseline()
        self.train_hitl()
    
    def get_metrics(self):
        """
        Get training metrics for both agents.
        
        Returns:
            dict: Metrics for comparison
        """
        baseline_avg_reward = np.mean(self.baseline_episode_rewards)
        baseline_avg_alerts = np.mean(self.baseline_episode_alerts)
        baseline_final_reward = np.mean(self.baseline_episode_rewards[-50:])
        baseline_final_alerts = np.mean(self.baseline_episode_alerts[-50:])
        
        hitl_avg_reward = np.mean(self.hitl_episode_rewards)
        hitl_avg_alerts = np.mean(self.hitl_episode_alerts)
        hitl_final_reward = np.mean(self.hitl_episode_rewards[-50:])
        hitl_final_alerts = np.mean(self.hitl_episode_alerts[-50:])
        
        return {
            'baseline': {
                'avg_reward': baseline_avg_reward,
                'avg_alerts': baseline_avg_alerts,
                'final_reward': baseline_final_reward,
                'final_alerts': baseline_final_alerts,
            },
            'hitl': {
                'avg_reward': hitl_avg_reward,
                'avg_alerts': hitl_avg_alerts,
                'final_reward': hitl_final_reward,
                'final_alerts': hitl_final_alerts,
            }
        }


def main():
    """Main training function"""
    manager = TrainingManager()
    manager.train_both()
    
    # Display final metrics
    metrics = manager.get_metrics()
    
    print("=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    print("\nBASELINE RL (No Human Feedback):")
    print(f"  Avg Episode Reward: {metrics['baseline']['avg_reward']:6.2f}")
    print(f"  Avg Alerts/Episode: {metrics['baseline']['avg_alerts']:5.2f}")
    print(f"  Final Avg Reward (last 50 eps): {metrics['baseline']['final_reward']:6.2f}")
    print(f"  Final Avg Alerts (last 50 eps): {metrics['baseline']['final_alerts']:5.2f}")
    
    print("\nHUMAN-IN-THE-LOOP RL (With Emotional Feedback):")
    print(f"  Avg Episode Reward: {metrics['hitl']['avg_reward']:6.2f}")
    print(f"  Avg Alerts/Episode: {metrics['hitl']['avg_alerts']:5.2f}")
    print(f"  Final Avg Reward (last 50 eps): {metrics['hitl']['final_reward']:6.2f}")
    print(f"  Final Avg Alerts (last 50 eps): {metrics['hitl']['final_alerts']:5.2f}")
    
    print("\nKEY OBSERVATIONS:")
    reward_diff = metrics['hitl']['final_reward'] - metrics['baseline']['final_reward']
    alert_diff = metrics['baseline']['final_alerts'] - metrics['hitl']['final_alerts']
    
    print(f"  Reward Difference: {reward_diff:+.2f} (HiTL vs Baseline)")
    print(f"  Alert Reduction: {alert_diff:+.2f} fewer alerts per episode with HiTL")
    
    print("\n" + "=" * 60)
    
    return manager


if __name__ == "__main__":
    main()
