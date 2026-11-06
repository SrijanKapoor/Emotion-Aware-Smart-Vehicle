"""
Q-Learning Agent Implementation
Learns optimal alert/assistance levels through epsilon-greedy exploration
"""

import random
from config import (
    NUM_ACTIONS, LEARNING_RATE, DISCOUNT_FACTOR, 
    EPSILON_START, EPSILON_END, EPSILON_DECAY
)


class QLearningAgent:
    """
    Q-Learning agent that learns to choose optimal actions based on state.
    
    Uses dictionary-based Q-table for state-action values.
    Employs epsilon-greedy exploration strategy.
    """
    
    def __init__(self, learning_rate=LEARNING_RATE, 
                 discount_factor=DISCOUNT_FACTOR,
                 epsilon_start=EPSILON_START,
                 epsilon_end=EPSILON_END,
                 epsilon_decay=EPSILON_DECAY):
        """
        Initialize Q-learning agent.
        
        Args:
            learning_rate: Step size for Q-value updates (alpha)
            discount_factor: Importance of future rewards (gamma)
            epsilon_start: Initial exploration probability
            epsilon_end: Minimum exploration probability
            epsilon_decay: Decay rate per episode
        """
        self.q_table = {}  # Dictionary: state -> dict of action: q_value
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.num_actions = NUM_ACTIONS
        self.episode_count = 0
    
    def get_q_value(self, state, action):
        """
        Get Q-value for state-action pair.
        Initialize to 0 if not seen before.
        
        Args:
            state: Tuple representing current state (speed, obstacle, emotion)
            action: Integer action (0-3)
        
        Returns:
            float: Q-value for this state-action pair
        """
        if state not in self.q_table:
            self.q_table[state] = {}
        
        if action not in self.q_table[state]:
            self.q_table[state][action] = 0.0
        
        return self.q_table[state][action]
    
    def select_action(self, state):
        """
        Select action using epsilon-greedy strategy.
        
        With probability epsilon: random action (exploration)
        With probability (1-epsilon): best Q-value action (exploitation)
        
        Args:
            state: Current state tuple
        
        Returns:
            int: Action to take (0-3)
        """
        # Exploration: random action
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        
        # Exploitation: best known action
        q_values = [self.get_q_value(state, a) for a in range(self.num_actions)]
        max_q_value = max(q_values)
        
        # Handle ties by selecting randomly among best actions
        best_actions = [a for a in range(self.num_actions) 
                       if self.get_q_value(state, a) == max_q_value]
        
        return random.choice(best_actions)
    
    def update_q_value(self, state, action, reward, next_state):
        """
        Update Q-value using Q-learning update rule.
        
        Q(s,a) = Q(s,a) + alpha * [R + gamma * max(Q(s',a')) - Q(s,a)]
        
        Args:
            state: Current state tuple
            action: Action taken
            reward: Reward received
            next_state: Resulting state tuple
        """
        # Get current Q-value
        current_q = self.get_q_value(state, action)
        
        # Get max Q-value for next state
        next_q_values = [self.get_q_value(next_state, a) 
                        for a in range(self.num_actions)]
        max_next_q = max(next_q_values)
        
        # Q-learning update
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        
        if state not in self.q_table:
            self.q_table[state] = {}
        self.q_table[state][action] = new_q
    
    def decay_epsilon(self):
        """
        Decay exploration rate at end of each episode.
        Gradually shift from exploration to exploitation.
        """
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        self.episode_count += 1
    
    def get_best_action(self, state):
        """
        Get best action for a state (pure exploitation, no randomness).
        Used during evaluation/testing.
        
        Args:
            state: State tuple
        
        Returns:
            int: Best action according to current Q-values
        """
        q_values = [self.get_q_value(state, a) for a in range(self.num_actions)]
        max_q_value = max(q_values)
        best_actions = [a for a in range(self.num_actions) 
                       if self.get_q_value(state, a) == max_q_value]
        return random.choice(best_actions)
