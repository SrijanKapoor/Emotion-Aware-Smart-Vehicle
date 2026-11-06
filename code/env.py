"""
Vehicle Environment Simulation
Simulates vehicle state transitions and calculates safety-based rewards
"""

import random
from config import (
    SPEED_STATES, OBSTACLE_DISTANCE_STATES, EMOTION_STATES,
    REWARD_SAFE_STATE, REWARD_UNSAFE_STATE
)


class VehicleEnvironment:
    """
    Simulates a vehicle environment with discretized states.
    
    State representation: (speed, obstacle_distance, driver_emotion)
    Transitions are probabilistic based on current state and action.
    """
    
    def __init__(self, seed=42):
        """Initialize the environment with random seed for reproducibility"""
        random.seed(seed)
        self.current_state = None
        self.reset()
    
    def reset(self):
        """
        Reset environment to initial random state.
        Returns initial state tuple: (speed, obstacle_distance, driver_emotion)
        """
        speed = random.choice(list(SPEED_STATES.values()))
        obstacle_distance = random.choice(list(OBSTACLE_DISTANCE_STATES.values()))
        emotion = random.choice(list(EMOTION_STATES.values()))
        self.current_state = (speed, obstacle_distance, emotion)
        return self.current_state
    
    def step(self, action):
        """
        Execute one environment step.
        
        Args:
            action (int): Action chosen by the agent (0-3)
        
        Returns:
            tuple: (next_state, environment_reward, done)
        """
        speed, obstacle_distance, emotion = self.current_state
        
        # ===== State Transitions (Stochastic) =====
        # Action 3 (Assistance) can help reduce speed safely
        if action == 3 and speed > 0:
            # Assistance helps reduce speed with 60% probability
            if random.random() < 0.6:
                speed = max(0, speed - 1)
        
        # Obstacle distance changes randomly (natural environment dynamics)
        if random.random() < 0.3:  # 30% chance obstacle gets closer
            obstacle_distance = min(1, obstacle_distance + 1)
        elif random.random() < 0.2:  # 20% chance obstacle gets closer
            obstacle_distance = max(0, obstacle_distance - 1)
        
        # Driver emotion changes based on stress (speed + near obstacle increases stress)
        stress_level = speed + (1 - obstacle_distance)  # Higher = more stressful
        if stress_level >= 2 and emotion < 3:
            if random.random() < 0.5:  # 50% chance emotion worsens
                emotion = min(3, emotion + 1)
        elif stress_level < 1 and emotion > 0:
            if random.random() < 0.3:  # 30% chance emotion improves
                emotion = max(0, emotion - 1)
        
        self.current_state = (speed, obstacle_distance, emotion)
        
        # ===== Environment Reward (Safety-Based) =====
        # Safe state: Low speed OR Far obstacle (or both)
        is_safe = (speed == 0) or (obstacle_distance == 1)
        
        if is_safe:
            env_reward = REWARD_SAFE_STATE
        else:
            # High speed AND Near obstacle = dangerous situation
            env_reward = REWARD_UNSAFE_STATE
        
        # Episode ends after MAX_STEPS or in critical unsafe state
        done = False
        
        return self.current_state, env_reward, done
    
    def get_state_hash(self):
        """
        Get hashable representation of current state for Q-table indexing.
        Returns tuple that can be used as dictionary key.
        """
        return self.current_state
