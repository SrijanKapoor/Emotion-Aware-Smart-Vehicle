"""
Configuration and constants for Emotion-Aware Smart Vehicle CPS Simulation
"""

# Vehicle States (Discretized)
SPEED_STATES = {
    'Low': 0,
    'Medium': 1,
    'High': 2
}

OBSTACLE_DISTANCE_STATES = {
    'Far': 0,
    'Near': 1
}

# Driver Emotional States
EMOTION_STATES = {
    'Calm': 0,
    'Stressed': 1,
    'Fatigued': 2,
    'Overloaded': 3
}

# Agent Actions (Alert Intensity / Assistance Levels)
ACTIONS = {
    'Low_alert': 0,
    'Medium_alert': 1,
    'High_alert': 2,
    'Activate_assistance': 3
}

NUM_ACTIONS = len(ACTIONS)

# Reward Values - Environment Safety
REWARD_SAFE_STATE = 1
REWARD_UNSAFE_STATE = -3

# Reward Values - Human Feedback (Emotional Feedback Shaping)
REWARD_OVERLOADED_HIGH_ALERT = -2  # Negative for giving too much alert to overloaded driver
REWARD_FATIGUED_ASSISTANCE = 2      # Positive for providing assistance to fatigued driver
REWARD_CALM_LOW_ALERT = 1           # Positive for low alert to calm driver
REWARD_DEFAULT = 0                  # No feedback

# Q-Learning Parameters
LEARNING_RATE = 0.1                 # Alpha - how much to update Q-values
DISCOUNT_FACTOR = 0.9               # Gamma - importance of future rewards
EPSILON_START = 1.0                 # Initial exploration rate
EPSILON_END = 0.01                  # Final exploration rate
EPSILON_DECAY = 0.995               # Decay rate per episode

# Training Configuration
NUM_EPISODES = 1000                 # Total training episodes
MAX_STEPS_PER_EPISODE = 100         # Max steps in each episode
EVALUATION_INTERVAL = 100           # Evaluate every N episodes

# Random seed for reproducibility
RANDOM_SEED = 42
