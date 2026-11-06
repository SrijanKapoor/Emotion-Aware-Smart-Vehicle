import pandas as pd
import numpy as np
import pickle
import os

# --- 1. DYNAMIC PATHING ---
# This ensures the script finds the pickle file even inside the 'code' folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PKL_PATH = os.path.join(BASE_DIR, 'ml_bundle.pkl')

# --- 2. LOAD ML MODEL ---
with open(PKL_PATH, 'rb') as f:
    bundle = pickle.load(f)

model = bundle['model']
scaler = bundle['scaler']
print("ML Model & Scaler Loaded Successfully!")

# --- 3. RL PIPELINE LOGIC ---
class DriverRLAgent:
    def __init__(self):
        self.state_map = {0: 'Calm', 1: 'Fatigued', 2: 'Overloaded', 3: 'Stressed'}
        
    def get_action(self, state_idx):
        state_name = self.state_map[state_idx]
        
        # RL Policy Mapping (State -> Action)
        actions = {
            'Calm': "Maintain standard assistance.",
            'Fatigued': "ACTION: High-intensity vibration alert + Rest stop suggestion.",
            'Stressed': "ACTION: Adjust ambient lighting + Enable adaptive cruise control.",
            'Overloaded': "ACTION: Silence non-critical notifications + Voice UI only."
        }
        return state_name, actions.get(state_name, "Standard Safety Protocol")

# --- 4. EXECUTION ---
# Example raw sensor reading (11 features)
# Order: traffic_density, horn_events, avg_speed, signal_wait, weather, road_quality, 
# experience, eye_closure, heart_rate, hrv, steering_var
sample_sensor_data = [65, 12, 45, 30, 1, 7, 5, 0.12, 85, 0.45, 1.2]

# Preprocess & Predict
input_df = pd.DataFrame([sample_sensor_data], columns=bundle['feature_names'])
scaled_data = scaler.transform(input_df)
prediction = model.predict(scaled_data)[0]

# Link to RL Agent
agent = DriverRLAgent()
state, action = agent.get_action(prediction)

print(f"\n[SENSOR DATA RECEIVED]")
print(f"Predicted Driver State: {state}")
print(f"RL Pipeline Action: {action}")