"""
Real-World Integration Example: ML Emotion Prediction + RL Agent
Demonstrates how to use the ML emotion predictor in a live driving scenario.

Usage:
1. Collect sensor data (from CARLA or real vehicle)
2. Get vehicle state (speed, obstacles)
3. Predict emotion using ML model
4. Feed into RL agent to get optimal action
5. Apply action to vehicle/UI
"""

import numpy as np
from ml_rl_integration import MLEmotionPredictor, create_rl_state, SAMPLE_FEATURES
from agent import QLearningAgent
from config import EMOTION_STATES, ACTIONS


class RealWorldMLRLPipeline:
    """
    Production pipeline for ML-based emotion prediction + RL-based decisions.
    """
    
    def __init__(self):
        """Initialize ML emotion predictor and RL agent."""
        # Load ML model once (cache it)
        self.emotion_predictor = MLEmotionPredictor()
        
        # Initialize or load trained RL agent
        self.rl_agent = QLearningAgent()
        
        print("\n[OK] ML-RL Pipeline initialized")
        print("  - ML Model: Ready for emotion prediction")
        print("  - RL Agent: Ready for decision making")
    
    def get_sensor_features(self):
        """
        Get sensor features from vehicle/biometric sensors.
        
        In real deployment, these would come from:
        - Traffic monitoring systems
        - Vehicle CAN bus (horn, speed, etc)
        - Biometric sensors (heart rate, eye closure)
        - Environmental sensors (weather, road quality)
        
        Returns:
            array: 11-element feature vector
        """
        # Simulated sensor data
        # In production: read from actual sensors via CAN bus, APIs, etc.
        return SAMPLE_FEATURES + np.random.normal(0, 2, 11)
    
    def get_vehicle_state(self, raw_speed):
        """
        Discretize continuous vehicle state to RL state space.
        
        Args:
            raw_speed (float): Continuous speed in km/h
        
        Returns:
            tuple: (speed_discrete, obstacle_distance_discrete)
        """
        # Discretize speed: Low (0-30), Medium (30-60), High (60+)
        if raw_speed < 30:
            speed = 0
        elif raw_speed < 60:
            speed = 1
        else:
            speed = 2
        
        # Simulate obstacle detection (in real scenario: from radar/lidar)
        # 0 = Far (>50m), 1 = Near (<50m)
        obstacle = np.random.randint(0, 2)
        
        return speed, obstacle
    
    def run_step(self, verbose=True):
        """
        Execute one step of the pipeline.
        
        Pipeline:
        1. Collect sensor data
        2. Predict emotion from ML model
        3. Get vehicle state (speed, obstacles)
        4. Combine into RL state
        5. Select action using RL agent
        6. Return action for vehicle control
        
        Args:
            verbose (bool): Print debug info
        
        Returns:
            dict: Step results with all information
        """
        result = {}
        
        # --- STEP 1: COLLECT SENSOR DATA ---
        if verbose:
            print("\n" + "="*70)
            print("STEP 1: Collect Sensor Data")
            print("="*70)
        
        sensor_features = self.get_sensor_features()
        sensor_features = np.maximum(sensor_features, 0)  # Ensure positive
        result['sensor_features'] = sensor_features
        
        if verbose:
            print(f"Traffic Density: {sensor_features[0]:.0f}")
            print(f"Horn Events/min: {sensor_features[1]:.0f}")
            print(f"Avg Speed: {sensor_features[2]:.1f} km/h")
            print(f"Signal Wait Time: {sensor_features[3]:.1f} s")
            print(f"Heart Rate: {sensor_features[8]:.0f} bpm")
            print(f"Steering Variability: {sensor_features[10]:.2f}")
        
        # --- STEP 2: PREDICT EMOTION ---
        if verbose:
            print("\n" + "="*70)
            print("STEP 2: ML Model Prediction")
            print("="*70)
        
        emotion_id = self.emotion_predictor.predict_emotion(sensor_features)
        emotion_name = {v: k for k, v in EMOTION_STATES.items()}[emotion_id]
        result['emotion_id'] = emotion_id
        result['emotion_name'] = emotion_name
        
        if verbose:
            print(f"Predicted Emotion: {emotion_name} (id={emotion_id})")
        
        # --- STEP 3: GET VEHICLE STATE ---
        if verbose:
            print("\n" + "="*70)
            print("STEP 3: Vehicle State")
            print("="*70)
        
        raw_speed = sensor_features[2]
        speed, obstacle = self.get_vehicle_state(raw_speed)
        
        speed_names = {0: 'Low', 1: 'Medium', 2: 'High'}
        obstacle_names = {0: 'Far', 1: 'Near'}
        
        result['speed'] = speed
        result['obstacle'] = obstacle
        
        if verbose:
            print(f"Speed: {speed_names[speed]} (raw={raw_speed:.1f} km/h)")
            print(f"Obstacle Distance: {obstacle_names[obstacle]}")
        
        # --- STEP 4: CREATE RL STATE ---
        if verbose:
            print("\n" + "="*70)
            print("STEP 4: Create RL State")
            print("="*70)
        
        state = create_rl_state(speed, obstacle, emotion_id)
        result['rl_state'] = state
        
        if verbose:
            print(f"RL State: {state}")
            print(f"  [speed={speed}, obstacle={obstacle}, emotion={emotion_id}]")
        
        # --- STEP 5: SELECT ACTION ---
        if verbose:
            print("\n" + "="*70)
            print("STEP 5: RL Agent Decision")
            print("="*70)
        
        action_id = self.rl_agent.select_action(state)
        
        action_names = {
            0: 'Low_alert',
            1: 'Medium_alert',
            2: 'High_alert',
            3: 'Activate_assistance'
        }
        
        action_name = action_names[action_id]
        result['action_id'] = action_id
        result['action_name'] = action_name
        
        # Map action to human-readable recommendations
        action_recommendations = {
            0: "Standard monitoring mode",
            1: "Increase attentiveness, provide gentle reminders",
            2: "Active alerts, offer assistance",
            3: "Activate adaptive cruise control, reduce manual demands"
        }
        
        if verbose:
            print(f"Selected Action: {action_name} (id={action_id})")
            print(f"Recommendation: {action_recommendations[action_id]}")
        
        result['recommendation'] = action_recommendations[action_id]
        
        # --- STEP 6: PREPARE FOR EXECUTION ---
        if verbose:
            print("\n" + "="*70)
            print("SUMMARY")
            print("="*70)
            print(f"\n[OK] Driver State: {emotion_name}")
            print(f"[OK] Vehicle State: Speed={speed_names[speed]}, Obstacle={obstacle_names[obstacle]}")
            print(f"[OK] Recommended Action: {action_name}")
            print(f"[OK] UI/Vehicle Control: {action_recommendations[action_id]}\n")
        
        return result
    
    def run_episode(self, num_steps=10):
        """
        Run a full episode with multiple steps.
        
        Args:
            num_steps (int): Number of steps in episode
        
        Returns:
            list: Results from each step
        """
        print("\n" + "="*70)
        print(f"ML-RL EPISODE: {num_steps} steps")
        print("="*70)
        
        results = []
        for step in range(num_steps):
            print(f"\n>>> EPISODE STEP {step+1}/{num_steps}")
            result = self.run_step(verbose=False)
            
            # Quick summary
            print(f"    Emotion: {result['emotion_name']:12} | "
                  f"Speed: {['Low', 'Med', 'High'][result['speed']]}: | "
                  f"Action: {result['action_name']}")
            
            results.append(result)
        
        print("\n" + "="*70)
        print("Episode complete!")
        print("="*70 + "\n")
        
        return results


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Initialize pipeline
    pipeline = RealWorldMLRLPipeline()
    
    # Example 1: Single step with verbose output
    print("\n" + "="*70)
    print("EXAMPLE 1: Single Step (Verbose)")
    print("="*70)
    result = pipeline.run_step(verbose=True)
    
    # Example 2: Full episode
    print("\n" + "="*70)
    print("EXAMPLE 2: Full Episode (10 steps)")
    print("="*70)
    results = pipeline.run_episode(num_steps=10)
    
    # Print statistics
    emotions = [r['emotion_name'] for r in results]
    actions = [r['action_name'] for r in results]
    
    print("\nEpisode Statistics:")
    print(f"  - Emotions: {set(emotions)}")
    print(f"  - Actions taken: {set(actions)}")
    print(f"  - Most common emotion: {max(set(emotions), key=emotions.count)}")
    print(f"  - Most common action: {max(set(actions), key=actions.count)}")
