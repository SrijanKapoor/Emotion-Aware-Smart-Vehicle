"""
Human Feedback Logic for Human-in-the-Loop Reinforcement Learning
Computes emotional feedback-based reward shaping
"""

from config import (
    EMOTION_STATES, ACTIONS,
    REWARD_OVERLOADED_HIGH_ALERT, REWARD_FATIGUED_ASSISTANCE,
    REWARD_CALM_LOW_ALERT, REWARD_DEFAULT
)


class HumanFeedback:
    """
    Simulates human feedback based on driver emotional state and agent actions.
    
    Provides reward shaping to guide the agent toward behaviors that consider
    driver emotional well-being, not just safety.
    """
    
    def __init__(self):
        """Initialize human feedback system"""
        pass
    
    def compute_emotional_reward(self, state, action):
        """
        Compute reward based on emotional appropriateness of action to state.
        
        This is the "Human-in-the-Loop" component: humans provide feedback
        about whether actions match driver emotional state.
        
        Rules:
        - Overloaded driver + High alert/Assistance → -2 (overwhelming)
        - Fatigued driver + Assistance → +2 (helpful)
        - Calm driver + Low alert → +1 (appropriate)
        - Else → 0 (neutral)
        
        Args:
            state: Tuple (speed, obstacle_distance, driver_emotion)
            action: Integer action (0-3)
        
        Returns:
            float: Emotional feedback reward
        """
        _, _, emotion = state
        
        # Overloaded driver (emotion=3)
        if emotion == EMOTION_STATES['Overloaded']:
            # Too much alert is overwhelming
            if action in [ACTIONS['High_alert'], ACTIONS['Activate_assistance']]:
                return REWARD_OVERLOADED_HIGH_ALERT  # -2
        
        # Fatigued driver (emotion=2)
        if emotion == EMOTION_STATES['Fatigued']:
            # Assistance helps fatigued drivers
            if action == ACTIONS['Activate_assistance']:
                return REWARD_FATIGUED_ASSISTANCE  # +2
        
        # Calm driver (emotion=0)
        if emotion == EMOTION_STATES['Calm']:
            # Low alert is appropriate for calm drivers
            if action == ACTIONS['Low_alert']:
                return REWARD_CALM_LOW_ALERT  # +1
        
        # Default: no emotional feedback
        return REWARD_DEFAULT  # 0
    
    def get_feedback_explanation(self, state, action, reward):
        """
        Get human-readable explanation of feedback (for debugging/analysis).
        
        Args:
            state: State tuple
            action: Action taken
            reward: Emotional reward received
        
        Returns:
            str: Description of why this reward was given
        """
        speed, obstacle, emotion = state
        emotion_names = {v: k for k, v in EMOTION_STATES.items()}
        action_names = {v: k for k, v in ACTIONS.items()}
        
        if reward == 0:
            return "No emotional feedback"
        
        emotion_str = emotion_names[emotion]
        action_str = action_names[action]
        
        if reward > 0:
            return f"✓ Good match: {emotion_str} driver with {action_str}"
        else:
            return f"✗ Poor match: {emotion_str} driver with {action_str}"
