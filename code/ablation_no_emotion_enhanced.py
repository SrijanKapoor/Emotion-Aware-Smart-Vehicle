"""
NO-EMOTION ABLATION EXPERIMENT - ENHANCED ANALYSIS
With better convergence detection and training curve analysis.
"""

import random
import numpy as np
from collections import deque


class SimpleEnv:
    """Minimal driver alert environment simulation."""
    
    def reset(self):
        self.sv = random.randint(0, 2)
        self.so = random.randint(0, 1)
        self.se = random.randint(0, 3)
        return (self.sv, self.so, self.se)

    def step(self, action):
        # Environment reward
        if self.sv == 2 and self.so == 0:
            r_env = -3
        else:
            r_env = +1

        # Stochastic transitions
        # Speed changes
        if action >= 2 and self.sv > 0:
            self.sv = max(0, self.sv - 1)   # high alert slows car
        elif random.random() < 0.1:
            self.sv = min(2, self.sv + 1)   # random speed increase

        # Obstacle changes
        if action == 3:
            self.so = 1                      # assistance clears obstacle
        elif random.random() < 0.15:
            self.so = random.randint(0, 1)  # random obstacle change

        # Emotion transitions
        if self.sv == 2 and self.so == 0:
            if random.random() < 0.4:
                self.se = min(3, self.se + 1)  # stress escalates
        elif action == 0 and self.se > 0:
            if random.random() < 0.2:
                self.se = max(0, self.se - 1)  # calm improves
        elif random.random() < 0.05:
            self.se = random.randint(0, 3)     # random emotion shift

        done = False  # episode ends by step limit only
        return (self.sv, self.so, self.se), r_env, done


def get_human_reward(se, action):
    """Compute human-centered reward based on actual emotion."""
    if se == 3 and action in [2, 3]:
        return -2
    elif se == 2 and action == 3:
        return +2
    elif se == 0 and action == 0:
        return +1
    return 0


class NoEmotionAgent:
    """Q-Learning agent with hidden emotion state."""
    
    def __init__(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        
        # Q-table: key is (sv, so, action)
        # Agent CANNOT see emotion se in its state representation
        self.Q = {}
        
        # Hyperparameters
        self.alpha = 0.1
        self.gamma = 0.9
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        
    def get_q(self, state, action):
        """Get Q-value for state-action pair."""
        key = (state[0], state[1], action)
        return self.Q.get(key, 0.0)
    
    def set_q(self, state, action, value):
        """Set Q-value for state-action pair."""
        key = (state[0], state[1], action)
        self.Q[key] = value
    
    def select_action(self, state):
        """Epsilon-greedy action selection on observable state (sv, so)."""
        if random.random() < self.epsilon:
            return random.randint(0, 3)
        else:
            q_values = [self.get_q(state, a) for a in range(4)]
            return np.argmax(q_values)
    
    def update(self, state, action, reward, next_state, done):
        """Q-learning update rule."""
        current_q = self.get_q(state, action)
        
        if done:
            target_q = reward
        else:
            next_q_values = [self.get_q(next_state, a) for a in range(4)]
            max_next_q = max(next_q_values)
            target_q = reward + self.gamma * max_next_q
        
        new_q = current_q + self.alpha * (target_q - current_q)
        self.set_q(state, action, new_q)
    
    def train(self, env, num_episodes):
        """Train agent for specified number of episodes."""
        episode_rewards = []
        
        for episode in range(num_episodes):
            state_full = env.reset()
            state = (state_full[0], state_full[1])  # Agent sees only (sv, so)
            
            episode_reward = 0
            for step in range(100):
                action = self.select_action(state)
                next_state_full, r_env, done = env.step(action)
                next_state = (next_state_full[0], next_state_full[1])
                
                # Compute total reward using actual emotion se
                r_human = get_human_reward(next_state_full[2], action)
                total_reward = r_env + r_human
                
                # Q-learning update
                self.update(state, action, total_reward, next_state, done)
                
                state = next_state
                episode_reward += total_reward
            
            # Decay epsilon after each episode
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            episode_rewards.append(episode_reward)
        
        return episode_rewards
    
    def evaluate(self, env, num_episodes):
        """Evaluate agent in pure exploitation mode (epsilon=0)."""
        self.epsilon = 0  # Pure exploitation
        episode_rewards = []
        episode_alerts = []
        
        for episode in range(num_episodes):
            state_full = env.reset()
            state = (state_full[0], state_full[1])
            
            episode_reward = 0
            num_alerts = 0
            
            for step in range(100):
                action = self.select_action(state)
                next_state_full, r_env, done = env.step(action)
                next_state = (next_state_full[0], next_state_full[1])
                
                # Compute reward (no updates during evaluation)
                r_human = get_human_reward(next_state_full[2], action)
                total_reward = r_env + r_human
                
                state = next_state
                episode_reward += total_reward
                
                # Count alert if action > 0
                if action > 0:
                    num_alerts += 1
            
            episode_rewards.append(episode_reward)
            episode_alerts.append(num_alerts)
        
        return episode_rewards, episode_alerts


def find_convergence_episode(train_rewards, window=100, patience=5):
    """
    Find convergence episode when moving average stabilizes.
    
    Convergence is detected when:
    1. We have at least window episodes to compute moving average
    2. The last 'patience' windows show decreasing or stable variance
    """
    if len(train_rewards) < window * 2:
        return len(train_rewards)
    
    # Compute moving average with given window
    moving_avg = deque(maxlen=window)
    variances = []
    
    for i in range(len(train_rewards)):
        moving_avg.append(train_rewards[i])
        if len(moving_avg) == window:
            variances.append(np.var(list(moving_avg)))
    
    # Check for stabilization in last part of training
    if len(variances) > patience:
        recent_vars = variances[-patience:]
        if np.mean(recent_vars) < 100:  # stabilized
            # Return where stabilization began
            return len(train_rewards) - len(recent_vars) * (window // patience)
    
    # Alternative: check when median reward in window stabilizes
    for i in range(window, len(train_rewards)):
        recent = train_rewards[i-window:i]
        recent_before = train_rewards[max(0, i-window-window):i-window]
        
        if len(recent_before) > 0:
            median_curr = np.median(recent)
            median_prev = np.median(recent_before)
            if abs(median_curr - median_prev) < 10:  # threshold
                return i - window
    
    return len(train_rewards)


def print_training_curve_summary(train_rewards):
    """Print summary statistics of training curve."""
    first_100 = np.mean(train_rewards[:100])
    last_100 = np.mean(train_rewards[-100:])
    max_reward = np.max(train_rewards)
    improvement = last_100 - first_100
    
    return {
        'first_100_mean': first_100,
        'last_100_mean': last_100,
        'max_reward': max_reward,
        'improvement': improvement
    }


def main():
    """Run ablation experiment with all seeds."""
    seeds = [42, 123, 999]
    all_results = {}
    
    print("="*70)
    print("RUNNING NO-EMOTION ABLATION EXPERIMENT (ENHANCED)")
    print("="*70)
    
    for seed in seeds:
        print(f"\n[Seed {seed}] Training agent for 1000 episodes...")
        env = SimpleEnv()
        agent = NoEmotionAgent(seed)
        
        # Training phase
        train_rewards = agent.train(env, 1000)
        
        # Evaluation phase
        print(f"[Seed {seed}] Evaluating agent for 100 episodes...")
        eval_rewards, eval_alerts = agent.evaluate(env, 100)
        
        # Training curve analysis
        curve_stats = print_training_curve_summary(train_rewards)
        
        all_results[seed] = {
            'train_rewards': train_rewards,
            'eval_rewards': eval_rewards,
            'eval_alerts': eval_alerts,
            'curve_stats': curve_stats
        }
        
        print(f"[Seed {seed}] Complete")
        print(f"  Training: first-100 avg={curve_stats['first_100_mean']:.2f}, "
              f"last-100 avg={curve_stats['last_100_mean']:.2f}, "
              f"improvement={curve_stats['improvement']:.2f}")
        print(f"  Evaluation: reward={np.mean(eval_rewards):.2f}, "
              f"alerts={np.mean(eval_alerts):.2f}")
    
    # Aggregate results across all seeds
    print("\n" + "="*70)
    print("AGGREGATING RESULTS ACROSS ALL SEEDS")
    print("="*70)
    
    all_eval_rewards = []
    all_eval_alerts = []
    convergence_episodes = []
    
    for seed in seeds:
        all_eval_rewards.extend(all_results[seed]['eval_rewards'])
        all_eval_alerts.extend(all_results[seed]['eval_alerts'])
        conv_ep = find_convergence_episode(all_results[seed]['train_rewards'])
        convergence_episodes.append(conv_ep)
        print(f"Seed {seed}: Convergence at ~episode {conv_ep}")
    
    # Compute aggregate statistics
    mean_reward = np.mean(all_eval_rewards)
    std_reward = np.std(all_eval_rewards)
    mean_alerts = np.mean(all_eval_alerts)
    std_alerts = np.std(all_eval_alerts)
    min_alerts = int(np.min(all_eval_alerts))
    max_alerts = int(np.max(all_eval_alerts))
    avg_convergence = int(np.mean(convergence_episodes))
    
    # Print final report
    print("\n" + "="*70)
    print("ABLATION RESULTS — NO-EMOTION AGENT (avg across seeds 42,123,999)")
    print("="*70)
    print(f"Mean reward         : {mean_reward:.2f}")
    print(f"Std dev reward      : {std_reward:.2f}")
    print(f"Mean alerts/episode : {mean_alerts:.2f}")
    print(f"Std dev alerts      : {std_alerts:.2f}")
    print(f"Min alerts          : {min_alerts}")
    print(f"Max alerts          : {max_alerts}")
    print(f"Convergence episode : ~{avg_convergence}")
    
    print("\n" + "-"*70)
    print("PER-SEED BREAKDOWN:")
    print("-"*70)
    for seed in seeds:
        seed_rewards = all_results[seed]['eval_rewards']
        seed_alerts = all_results[seed]['eval_alerts']
        conv_ep = find_convergence_episode(all_results[seed]['train_rewards'])
        curve = all_results[seed]['curve_stats']
        print(f"Seed {seed:>3} → reward: {np.mean(seed_rewards):>6.2f} | "
              f"alerts: {np.mean(seed_alerts):>6.2f} | "
              f"train-improvement: {curve['improvement']:>6.2f} | "
              f"conv: ~{conv_ep}")
    
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
