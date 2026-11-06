"""
ML-RL Integration Layer
Integrates trained ML emotion prediction model with Q-Learning RL agent.

This module:
1. Loads the ML model bundle (model + scaler)
2. Provides predict_emotion() function for inference
3. Handles feature preprocessing and emotion prediction
4. Creates RL states with ML-predicted emotions
"""

import pickle
import numpy as np
import pandas as pd
import os
from config import EMOTION_STATES


# ============================================================================
# FEATURE CONFIGURATION
# ============================================================================
# Expected feature order from training data
# Note: stress_index is derived, see compute_stress_index()
RAW_FEATURE_NAMES = [
    'traffic_density',
    'horn_events_per_min',
    'avg_speed',
    'signal_wait_time',
    'weather_condition',
    'road_quality_score',
    'driver_experience_level',
    'eye_closure_ratio',
    'heart_rate',
    'heart_rate_variability',
    'steering_variability'
]

ALL_FEATURE_NAMES = RAW_FEATURE_NAMES + ['stress_index']

# Sample feature vector for testing (11 raw features)
# stress_index will be computed automatically
SAMPLE_FEATURES = np.array([
    65,      # traffic_density: 0-100
    12,      # horn_events_per_min: 0-50
    45,      # avg_speed: km/h
    30,      # signal_wait_time: seconds
    1,       # weather_condition: 0=Clear, 1=Foggy, 2=Rainy, 3=Night
    7,       # road_quality_score: 1-10
    5,       # driver_experience_level: 1-10
    0.12,    # eye_closure_ratio: 0-1
    85,      # heart_rate: bpm
    0.45,    # heart_rate_variability: 0-1
    1.2      # steering_variability: 0-2
])


def compute_stress_index(features):
    """
    Compute stress_index from physiological and situational metrics.
    
    Stress index is a composite measure (typical range: 18-66) based on:
    - Heart rate elevation (compared to resting baseline)
    - Heart rate variability (low HRV = stressed, high HRV = calm)
    - Steering variability (indicates tension/distraction)
    - Traffic density and signal wait time (situational stressors)
    
    Args:
        features (array): Array of raw 11 features
    
    Returns:
        float: Computed stress index (typically 20-60 range)
    """
    # Extract relevant physiological and situational features
    heart_rate = features[8]              # ~60-120 bpm
    hrv = features[9]                     # 0-1 (low = stressed)
    steering_var = features[10]           # 0-2+ (high = stressed)
    traffic = features[0]                 # 0-100
    signal_wait = features[3]             # 0-200+ seconds
    
    # Base: resting stress (everyone has baseline stress level)
    base_stress = 35.0
    
    # Heart rate component: elevated HR indicates stress
    # 60-80 bpm = normal, 100+ = stressed
    hr_stress = max(0, (heart_rate - 70) * 0.3)
    
    # HRV component: inverse relationship to stress
    # High HRV (0.8+) = calm, Low HRV (0-0.3) = very stressed
    hrv_stress = (1 - np.clip(hrv, 0, 1)) * 20
    
    # Steering variability: high values indicate tension/poor control
    # 0-0.5 = normal, 1+ = stressed
    steering_stress = min(steering_var * 5, 15)
    
    # Situational stress from traffic
    traffic_stress = (traffic / 100) * 8
    
    # Wait time stress (more stressful in traffic)
    wait_stress = min((signal_wait / 150) * 8, 8)
    
    # Combine all components
    stress_index = (
        base_stress +
        hr_stress +
        hrv_stress +
        steering_stress +
        traffic_stress +
        wait_stress
    )
    
    # Clip to realistic range
    stress_index = np.clip(stress_index, 15, 70)
    
    return stress_index


# ============================================================================
# ML MODEL LOADING
# ============================================================================

class MLEmotionPredictor:
    """
    Loads and manages ML model for emotion prediction.
    Handles different pickle formats and provides prediction interface.
    """
    
    def __init__(self, model_path='ml_bundle.pkl'):
        """
        Initialize ML emotion predictor.
        
        Args:
            model_path (str): Path to ml_bundle.pkl file
        
        Raises:
            FileNotFoundError: If model file doesn't exist
            ValueError: If bundle has unexpected format
        """
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.le_weather = None
        self.le_state = None
        self._load_model(model_path)
    
    def _load_model(self, model_path):
        """
        Load ML model from pickle file.
        Handles formats:
        - Raw model object
        - Dict with {'model': model, 'scaler': scaler, 'feature_cols': [...]}
        
        Args:
            model_path (str): Path to pickle file
        """
        # Handle both absolute and relative paths
        if not os.path.isabs(model_path):
            model_path = os.path.join(os.path.dirname(__file__), model_path)
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        print(f"Loading ML model from: {model_path}")
        
        with open(model_path, 'rb') as f:
            loaded_obj = pickle.load(f)
        
        # Handle different pickle formats
        if isinstance(loaded_obj, dict):
            # Format: {"model": model, "scaler": scaler, "feature_cols": [...], ...}
            self.model = loaded_obj.get('model')
            self.scaler = loaded_obj.get('scaler')
            self.feature_names = loaded_obj.get('feature_cols', ALL_FEATURE_NAMES)
            
            # Store label encoders if present
            self.le_weather = loaded_obj.get('le_weather')
            self.le_state = loaded_obj.get('le_state')
            
            if self.model is None:
                raise ValueError("Bundle dict missing 'model' key")
            
            print(f"[OK] Loaded model + scaler from bundle")
            print(f"  Feature count: {len(self.feature_names)}")
            print(f"  Features: {self.feature_names}")
        else:
            # Format: Raw model object (no scaler)
            self.model = loaded_obj
            self.scaler = None
            self.feature_names = ALL_FEATURE_NAMES
            self.le_weather = None
            self.le_state = None
            print(f"[OK] Loaded raw model (no scaler)")
    
    def predict_emotion(self, features):
        """
        Predict driver emotion from feature vector.
        
        Args:
            features (array-like): Feature vector of shape (11,) or (12,)
                If 11 features: [traffic_density, horn_events_per_min, avg_speed, 
                                 signal_wait_time, weather_condition, road_quality_score,
                                 driver_experience_level, eye_closure_ratio, heart_rate,
                                 heart_rate_variability, steering_variability]
                If 12 features: same as above + stress_index at the end
        
        Returns:
            int: Predicted emotion class (0=Calm, 1=Fatigued, 2=Overloaded, 3=Stressed)
        
        Raises:
            ValueError: If feature shape is invalid
            RuntimeError: If model prediction fails
        """
        
        # --- INPUT VALIDATION & PROCESSING ---
        features = np.asarray(features).flatten()
        
        if len(features) == len(RAW_FEATURE_NAMES):
            # 11 features provided: compute stress_index
            print(f"\n  Input: {len(features)} features (raw)")
            stress_idx = compute_stress_index(features)
            features = np.append(features, stress_idx)
            print(f"  Computed stress_index: {stress_idx:.2f}")
        elif len(features) == len(ALL_FEATURE_NAMES):
            # 12 features provided: use as-is
            print(f"\n  Input: {len(features)} features (includes stress_index)")
        else:
            raise ValueError(
                f"Expected {len(RAW_FEATURE_NAMES)} or {len(ALL_FEATURE_NAMES)} features, "
                f"got {len(features)}. "
                f"Features: {RAW_FEATURE_NAMES} [+ stress_index]"
            )
        
        print(f"  Features shape: {features.shape} [OK]")
        
        # --- PREPROCESSING ---
        if self.scaler is not None:
            # Create DataFrame with feature names for proper scaling
            feature_dict = {name: [val] for name, val in zip(self.feature_names, features)}
            df = pd.DataFrame(feature_dict)
            features_scaled = self.scaler.transform(df)[0]
            print(f"  Scaling applied (StandardScaler) [OK]")
        else:
            features_scaled = features
            print(f"  No scaling applied")
        
        # --- PREDICTION ---
        try:
            # Model expects 2D input: (n_samples, n_features)
            prediction = self.model.predict([features_scaled])[0]
            
            # Ensure output is integer
            emotion_id = int(prediction)
            
            # Validate emotion ID
            if emotion_id not in EMOTION_STATES.values():
                raise ValueError(f"Invalid emotion ID from model: {emotion_id}")
            
            return emotion_id
        
        except Exception as e:
            raise RuntimeError(f"Model prediction failed: {str(e)}")


# ============================================================================
# RL STATE CREATION
# ============================================================================

def create_rl_state(speed, obstacle, emotion):
    """
    Create RL state tuple from components.
    
    Args:
        speed (int): Vehicle speed state (0=Low, 1=Medium, 2=High)
        obstacle (int): Obstacle distance state (0=Far, 1=Near)
        emotion (int): Driver emotion state (0=Calm, 1=Stressed, 2=Fatigued, 3=Overloaded)
    
    Returns:
        tuple: (speed, obstacle, emotion)
    """
    return (speed, obstacle, emotion)


def predict_emotion(features, predictor=None):
    """
    Convenience function to predict emotion from features.
    Lazy-loads predictor if not provided.
    
    Args:
        features (array-like): Feature vector of shape (11,)
        predictor (MLEmotionPredictor, optional): Shared predictor instance
    
    Returns:
        int: Predicted emotion (0-3)
    """
    if predictor is None:
        predictor = MLEmotionPredictor()
    
    return predictor.predict_emotion(features)


# ============================================================================
# INTEGRATION LOOP EXAMPLE
# ============================================================================

def run_integration_example(num_steps=10):
    """
    Example integration loop showing ML model feeding into RL agent.
    
    Args:
        num_steps (int): Number of steps to run
    """
    from agent import QLearningAgent
    
    print("\n" + "="*70)
    print("ML-RL INTEGRATION EXAMPLE")
    print("="*70)
    
    # Initialize ML predictor
    predictor = MLEmotionPredictor()
    
    # Initialize RL agent
    agent = QLearningAgent()
    
    # Simulate multiple time steps
    for step in range(num_steps):
        print(f"\n--- STEP {step+1}/{num_steps} ---")
        
        # Simulate sensor input (in real scenario, from vehicle/biometric sensors)
        features = SAMPLE_FEATURES + np.random.normal(0, 2, len(SAMPLE_FEATURES))
        features = np.maximum(features, 0)  # Ensure non-negative
        
        # Get emotion prediction from ML model
        print("\n1. ML Model Prediction:")
        emotion = predict_emotion(features, predictor)
        emotion_name = {v: k for k, v in EMOTION_STATES.items()}[emotion]
        print(f"   Predicted Emotion: {emotion_name} (id={emotion}) [OK]")
        
        # Simulate vehicle state (normally from CARLA or actual vehicle)
        speed = np.random.randint(0, 3)     # 0, 1, or 2
        obstacle = np.random.randint(0, 2)  # 0 or 1
        
        speed_names = {0: 'Low', 1: 'Medium', 2: 'High'}
        obstacle_names = {0: 'Far', 1: 'Near'}
        
        # Create RL state
        print("\n2. RL State Creation:")
        state = create_rl_state(speed, obstacle, emotion)
        print(f"   Speed: {speed_names[speed]} ({speed})")
        print(f"   Obstacle: {obstacle_names[obstacle]} ({obstacle})")
        print(f"   Emotion: {emotion_name} ({emotion})")
        print(f"   RL State: {state} [OK]")
        
        # Agent selects action
        print("\n3. RL Agent Decision:")
        action = agent.select_action(state)
        
        action_names = {
            0: 'Low_alert',
            1: 'Medium_alert',
            2: 'High_alert',
            3: 'Activate_assistance'
        }
        
        print(f"   Selected Action: {action_names[action]} (id={action}) [OK]")
        
        # Simulate step
        next_state = create_rl_state(
            np.random.randint(0, 3),
            np.random.randint(0, 2),
            emotion
        )
        reward = np.random.choice([-3, 0, 1])
        
        print(f"   Reward: {reward}")
        
        # Update Q-value
        agent.update_q_value(state, action, reward, next_state)
    
    print("\n" + "="*70)
    print("Integration example complete!")
    print("="*70 + "\n")


if __name__ == "__main__":
    # Run example integration
    run_integration_example(num_steps=5)
