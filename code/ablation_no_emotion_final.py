"""
NO-EMOTION ABLATION EXPERIMENT - FINAL ANALYSIS
With robust convergence detection using moving averages.
"""

import random
import numpy as np


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


def compute_moving_average(rewards, window):
    """Compute moving average of rewards."""
    ma = []
    for i in range(len(rewards)):
        start = max(0, i - window + 1)
        ma.append(np.mean(rewards[start:i+1]))
    return ma


def find_convergence_episode(train_rewards):
    """
    Find convergence episode by analyzing moving averages.
    Convergence is when 50-episode moving average stops improving significantly.
    """
    window = 50
    ma = compute_moving_average(train_rewards, window)
    
    # Find where the moving average stops increasing by more than 0.5 per episode
    improvement_threshold = 0.5
    stable_count = 0
    stable_threshold = 50  # need 50 episodes of small improvements
    
    for i in range(window + 1, len(ma)):
        improvement = ma[i] - ma[i-1]
        if improvement < improvement_threshold:
            stable_count += 1
            if stable_count >= stable_threshold:
                return i - stable_threshold
        else:
            stable_count = 0
    
    # If not converged, use 70% of training as estimate
    return int(0.7 * len(train_rewards))


def print_training_progression(train_rewards):
    """Print training progression at key checkpoints."""
    checkpoints = [100, 250, 500, 750, 1000]
    print("  Training progression:")
    for cp in checkpoints:
        if cp <= len(train_rewards):
            recent = train_rewards[max(0, cp-50):cp]
            print(f"    Episode {cp:4d}: avg reward (last 50) = {np.mean(recent):7.2f}")


def main():
    """Run ablation experiment with all seeds."""
    seeds = [42, 123, 999]
    all_results = {}
    
    print("="*70)
    print("RUNNING NO-EMOTION ABLATION EXPERIMENT")
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
        
        # Store results
        all_results[seed] = {
            'train_rewards': train_rewards,
            'eval_rewards': eval_rewards,
            'eval_alerts': eval_alerts
        }
        
        # Print summary
        print_training_progression(train_rewards)
        print(f"  Evaluation: mean reward={np.mean(eval_rewards):.2f}, "
              f"mean alerts={np.mean(eval_alerts):.2f}")
    
    # Aggregate results across all seeds
    print("\n" + "="*70)
    print("COMPUTING AGGREGATED METRICS")
    print("="*70)
    
    all_eval_rewards = []
    all_eval_alerts = []
    convergence_episodes = []
    
    for seed in seeds:
        all_eval_rewards.extend(all_results[seed]['eval_rewards'])
        all_eval_alerts.extend(all_results[seed]['eval_alerts'])
        conv_ep = find_convergence_episode(all_results[seed]['train_rewards'])
        convergence_episodes.append(conv_ep)
    
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
        print(f"Seed {seed:>3} → reward: {np.mean(seed_rewards):>6.2f} | alerts: {np.mean(seed_alerts):>6.2f} | conv: ~{conv_ep}")
    
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
