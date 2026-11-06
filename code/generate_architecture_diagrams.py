"""
Generate Comprehensive System Architecture Diagram
Detailed explanatory architecture for Emotion-Aware Smart Vehicle Control System
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle
import numpy as np

# Set professional style
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['font.size'] = 9
plt.rcParams['font.family'] = 'sans-serif'

def create_system_architecture():
    """Create comprehensive system architecture diagram"""
    
    fig = plt.figure(figsize=(18, 13))
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 14)
    ax.axis('off')
    
    # Define colors
    color_input = '#FF6B6B'
    color_ml = '#4ECDC4'
    color_rl = '#FFD93D'
    color_env = '#6BCB77'
    color_output = '#95E1D3'
    color_support = '#B4E7FF'
    
    # Title
    ax.text(10, 13.3, 'Emotion-Aware Smart Vehicle Control System Architecture', 
           ha='center', fontsize=18, fontweight='bold')
    ax.text(10, 12.95, 'Human-in-the-Loop RL with ML-based Real-Time Emotion Prediction', 
           ha='center', fontsize=12, style='italic', color='gray')
    
    # ============================================================================
    # LAYER 1: INPUT SENSORS (Top)
    # ============================================================================
    y_input = 11.5
    ax.text(0.3, y_input + 0.7, 'LAYER 1: SENSOR INPUT LAYER', fontsize=11, 
           fontweight='bold', color='#333')
    
    sensors = [
        ('Steering\nAngle', 1.0),
        ('Throttle\nPosition', 2.2),
        ('Brake\nPressure', 3.4),
        ('Heart\nRate', 4.6),
        ('Voice\nStress', 5.8),
        ('Facial\nFeatures', 7.0),
        ('Lat.\nAccel', 8.2),
        ('Vert.\nAccel', 9.4),
        ('Speed', 10.6),
        ('Obstacle\nDist', 11.8),
        ('Context', 13.0)
    ]
    
    sensor_rects = []
    for sensor_name, x_pos in sensors:
        rect = FancyBboxPatch((x_pos - 0.45, y_input - 0.35), 0.9, 0.6,
                             boxstyle="round,pad=0.05", edgecolor='black', 
                             facecolor=color_input, alpha=0.6, linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x_pos, y_input, sensor_name, ha='center', va='center', 
               fontsize=8, fontweight='bold')
        sensor_rects.append((x_pos, y_input))
    
    # Arrows from sensors downward
    for x_pos, y_pos in sensor_rects[:6]:  # First 6 to ML
        ax.arrow(x_pos, y_pos - 0.4, 0, -0.3, head_width=0.15, head_length=0.08,
                fc='gray', ec='gray', alpha=0.4)
    
    for x_pos, y_pos in sensor_rects[6:]:  # Rest to environment
        ax.arrow(x_pos, y_pos - 0.4, 0, -1.2, head_width=0.15, head_length=0.08,
                fc='gray', ec='gray', alpha=0.4)
    
    # ============================================================================
    # LAYER 2: ML EMOTION PREDICTION
    # ============================================================================
    y_ml = 10.0
    ax.text(0.3, y_ml + 0.9, 'LAYER 2: ML EMOTION PREDICTION MODULE', fontsize=11,
           fontweight='bold', color='#333')
    
    # ML Processing box
    rect_ml = FancyBboxPatch((0.5, y_ml - 0.35), 5.5, 0.7,
                            boxstyle="round,pad=0.08", edgecolor='#4ECDC4', 
                            facecolor=color_ml, alpha=0.25, linewidth=2.5)
    ax.add_patch(rect_ml)
    
    ax.text(1.2, y_ml + 0.15, 'Feature\nExtraction', ha='center', fontsize=8, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='#4ECDC4', linewidth=1))
    ax.text(2.6, y_ml + 0.15, 'Feature\nScaling', ha='center', fontsize=8, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='#4ECDC4', linewidth=1))
    ax.text(4.0, y_ml + 0.15, 'RandomForest\nClassifier', ha='center', fontsize=8, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='#4ECDC4', linewidth=1))
    
    ax.text(3.0, y_ml - 0.25, 'Trained on 15K driver samples | 89.4% Accuracy | 5.2ms Latency', 
           ha='center', fontsize=7, style='italic', color='#4ECDC4', fontweight='bold')
    
    # Output from ML
    ax.arrow(3.0, y_ml - 0.4, 0, -0.3, head_width=0.2, head_length=0.1,
            fc='#4ECDC4', ec='#4ECDC4', linewidth=2)
    
    # ML Output box
    rect_ml_out = FancyBboxPatch((1.5, y_ml - 1.1), 3.0, 0.5,
                               boxstyle="round,pad=0.05", edgecolor='#4ECDC4',
                               facecolor='#E0F7F6', alpha=0.8, linewidth=2)
    ax.add_patch(rect_ml_out)
    ax.text(3.0, y_ml - 0.85, 'Predicted Emotion: 0-3 (Calm → Overloaded)\nConfidence Score', 
           ha='center', fontsize=8, fontweight='bold', color='#4ECDC4')
    
    # ============================================================================
    # LAYER 3: HYBRID ENVIRONMENT & RL AGENT
    # ============================================================================
    y_env_agent = 7.8
    ax.text(0.3, y_env_agent + 1.2, 'LAYER 3: HYBRID ENVIRONMENT & RL AGENT', fontsize=11,
           fontweight='bold', color='#333')
    
    # Environment box
    rect_env = FancyBboxPatch((0.5, y_env_agent - 0.8), 5.2, 1.0,
                             boxstyle="round,pad=0.1", edgecolor='#6BCB77',
                             facecolor=color_env, alpha=0.2, linewidth=2.5)
    ax.add_patch(rect_env)
    
    ax.text(1.0, y_env_agent + 0.05, 'ENVIRONMENT', fontsize=9, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='#6BCB77', alpha=0.8, edgecolor='black', linewidth=1))
    
    ax.text(1.0, y_env_agent - 0.35, 'Simulation Mode', fontsize=7, style='italic',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.7, edgecolor='#6BCB77', linewidth=1))
    ax.text(2.8, y_env_agent - 0.35, 'CARLA Mode', fontsize=7, style='italic',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.7, edgecolor='#6BCB77', linewidth=1))
    ax.text(4.6, y_env_agent - 0.35, 'State: (Speed,\nObstacle,\nML_Emotion)', 
           fontsize=7, style='italic', fontweight='bold', ha='center')
    
    # Agent box
    rect_agent = FancyBboxPatch((6.5, y_env_agent - 0.8), 5.2, 1.0,
                              boxstyle="round,pad=0.1", edgecolor='#FFD93D',
                              facecolor=color_rl, alpha=0.2, linewidth=2.5)
    ax.add_patch(rect_agent)
    
    ax.text(7.0, y_env_agent + 0.05, 'RL AGENT', fontsize=9, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='#FFD93D', alpha=0.8, edgecolor='black', linewidth=1))
    
    ax.text(7.0, y_env_agent - 0.35, 'Q-Table', fontsize=7, style='italic',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.7, edgecolor='#FFD93D', linewidth=1))
    ax.text(8.3, y_env_agent - 0.35, 'Epsilon-Greedy\nSelection', fontsize=7, style='italic',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.7, edgecolor='#FFD93D', linewidth=1))
    ax.text(9.9, y_env_agent - 0.35, 'Actions: 4\n(L/M/H Alert,\nAssistance)', 
           fontsize=7, style='italic', fontweight='bold', ha='center')
    
    # Arrows between layers
    ax.arrow(3.0, y_ml - 1.15, 1.5, -0.8, head_width=0.2, head_length=0.1,
            fc='#4ECDC4', ec='#4ECDC4', linewidth=2, linestyle='--')
    ax.text(3.5, y_ml - 1.5, 'ML Emotion', fontsize=7, color='#4ECDC4', fontweight='bold')
    
    # From environment sensors to agent
    ax.arrow(5.7, y_env_agent - 0.3, 0.7, 0, head_width=0.15, head_length=0.1,
            fc='#FF6B6B', ec='#FF6B6B', linewidth=2)
    ax.text(6.1, y_env_agent + 0.2, 'State', fontsize=7, color='#FF6B6B', fontweight='bold')
    
    # From agent to environment (action)
    ax.arrow(9.0, y_env_agent - 0.9, -2.3, 0, head_width=0.15, head_length=0.1,
            fc='#FFD93D', ec='#FFD93D', linewidth=2)
    ax.text(8.2, y_env_agent - 1.1, 'Action', fontsize=7, color='#FFD93D', fontweight='bold')
    
    # ============================================================================
    # LAYER 4: REWARD COMPUTATION (Human Feedback)
    # ============================================================================
    y_reward = 6.3
    ax.text(0.3, y_reward + 0.8, 'LAYER 4: REWARD COMPUTATION (Emotional Feedback)', fontsize=11,
           fontweight='bold', color='#333')
    
    # Reward computation box
    rect_reward = FancyBboxPatch((0.5, y_reward - 0.65), 11.2, 0.75,
                                boxstyle="round,pad=0.08", edgecolor='#FF9F43',
                                facecolor='#FFE66D', alpha=0.15, linewidth=2.5)
    ax.add_patch(rect_reward)
    
    ax.text(1.5, y_reward + 0.02, 'Environment Reward', fontsize=8, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='#6BCB77', linewidth=1))
    ax.text(1.5, y_reward - 0.3, 'R_env = +1 (safe)\nR_env = -3 (unsafe)', fontsize=6, ha='center')
    
    ax.text(4.0, y_reward + 0.02, '+', fontsize=16, fontweight='bold', ha='center')
    
    ax.text(5.2, y_reward + 0.02, 'Emotional Reward\n(based on ML emotion)', fontsize=8, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='#4ECDC4', linewidth=1))
    ax.text(5.2, y_reward - 0.3, 'Overloaded+Alert=-2\nFatigued+Assist=+2\nCalm+LowAlert=+1', fontsize=6, ha='center')
    
    ax.text(8.2, y_reward + 0.02, '=', fontsize=16, fontweight='bold', ha='center')
    
    ax.text(9.2, y_reward + 0.02, 'Total Reward', fontsize=8, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='#FF9F43', linewidth=1))
    ax.text(9.2, y_reward - 0.3, 'R_total = R_env +\nR_human(ML_emotion)', fontsize=6, ha='center')
    
    # Arrow from environment to reward
    ax.arrow(3.0, y_env_agent - 0.9, 0, -0.65, head_width=0.2, head_length=0.1,
            fc='gray', ec='gray', alpha=0.5, linewidth=1.5)
    ax.text(3.3, y_reward - 1.1, 'Env Reward', fontsize=6, color='gray')
    
    # Arrow from agent to reward (for ML emotion info)
    ax.arrow(9.0, y_env_agent - 0.9, 0.5, -0.65, head_width=0.2, head_length=0.1,
            fc='#4ECDC4', ec='#4ECDC4', alpha=0.5, linewidth=1.5, linestyle='--')
    ax.text(9.8, y_reward - 1.1, 'ML Emotion', fontsize=6, color='#4ECDC4')
    
    # ============================================================================
    # LAYER 5: Q-LEARNING UPDATE
    # ============================================================================
    y_update = 4.5
    ax.text(0.3, y_update + 0.8, 'LAYER 5: Q-LEARNING UPDATE', fontsize=11,
           fontweight='bold', color='#333')
    
    # Q-update formula box
    rect_formula = FancyBboxPatch((0.5, y_update - 0.45), 11.2, 0.65,
                                 boxstyle="round,pad=0.08", edgecolor='#9B59B6',
                                 facecolor='#E8D5F2', alpha=0.6, linewidth=2.5)
    ax.add_patch(rect_formula)
    
    formula_text = 'Q(s,a) ← Q(s,a) + α[R + γ·max(Q(s\',a\')) - Q(s,a)]'
    ax.text(6.1, y_update + 0.1, formula_text, ha='center', fontsize=9, 
           family='monospace', fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='#9B59B6', linewidth=1.5))
    
    param_text = 'α=0.1 (learning rate) | γ=0.9 (discount factor) | R=Total Reward (with emotional shaping)'
    ax.text(6.1, y_update - 0.25, param_text, ha='center', fontsize=7, style='italic', color='#333')
    
    # Arrow from reward to update
    ax.arrow(9.2, y_reward - 0.7, 0, -0.6, head_width=0.2, head_length=0.1,
            fc='#FF9F43', ec='#FF9F43', linewidth=2)
    ax.text(9.6, y_reward - 1.3, 'R_total', fontsize=7, color='#FF9F43', fontweight='bold')
    
    # ============================================================================
    # LAYER 6: STATE TRANSITION & FEEDBACK
    # ============================================================================
    y_transition = 2.7
    ax.text(0.3, y_transition + 1.3, 'LAYER 6: STATE TRANSITION & FEEDBACK LOOP', fontsize=11,
           fontweight='bold', color='#333')
    
    # Next state box
    rect_next = FancyBboxPatch((1.0, y_transition + 0.3), 2.8, 0.7,
                              boxstyle="round,pad=0.08", edgecolor='#6BCB77',
                              facecolor=color_env, alpha=0.3, linewidth=2)
    ax.add_patch(rect_next)
    ax.text(2.4, y_transition + 0.67, 'Next State', fontsize=8, fontweight='bold')
    ax.text(2.4, y_transition + 0.35, 's\' = (speed\', obstacle\',\nML_emotion\')', 
           fontsize=7, ha='center')
    
    # Reward stored box
    rect_stored = FancyBboxPatch((4.5, y_transition + 0.3), 2.8, 0.7,
                               boxstyle="round,pad=0.08", edgecolor='#FF9F43',
                               facecolor='#FFE66D', alpha=0.3, linewidth=2)
    ax.add_patch(rect_stored)
    ax.text(5.9, y_transition + 0.67, 'Update Q-Table', fontsize=8, fontweight='bold')
    ax.text(5.9, y_transition + 0.35, 'Store: Q(s,a)\nNew Value', fontsize=7, ha='center')
    
    # Vehicle execution box
    rect_exec = FancyBboxPatch((8.0, y_transition + 0.3), 2.8, 0.7,
                              boxstyle="round,pad=0.08", edgecolor='#FF6B6B',
                              facecolor='#FFB3B3', alpha=0.3, linewidth=2)
    ax.add_patch(rect_exec)
    ax.text(9.4, y_transition + 0.67, 'Execute Action', fontsize=8, fontweight='bold')
    ax.text(9.4, y_transition + 0.35, 'Vehicle Issues Alert\nAdjusts Assistance', fontsize=7, ha='center')
    
    # Arrows between transition boxes
    ax.arrow(3.8, y_transition + 0.67, 0.6, 0, head_width=0.12, head_length=0.08,
            fc='gray', ec='gray', linewidth=1.5)
    ax.arrow(7.3, y_transition + 0.67, 0.6, 0, head_width=0.12, head_length=0.08,
            fc='gray', ec='gray', linewidth=1.5)
    
    # Arrow from update to transition
    ax.arrow(6.1, y_update - 0.5, 0, -0.9, head_width=0.2, head_length=0.1,
            fc='#9B59B6', ec='#9B59B6', linewidth=2)
    
    # ============================================================================
    # LAYER 7: METRICS & MONITORING
    # ============================================================================
    y_metrics = 0.8
    ax.text(0.3, y_metrics + 1.5, 'LAYER 7: PERFORMANCE METRICS & MONITORING', fontsize=11,
           fontweight='bold', color='#333')
    
    metrics_boxes = [
        ('Episode\nReward', 1.5, '#4ECDC4'),
        ('Alert\nFrequency', 3.5, '#FF6B6B'),
        ('Emotional\nAppropriateness', 5.8, '#FFD93D'),
        ('Convergence\nProgress', 8.2, '#6BCB77'),
        ('ML Integration\nAccuracy', 10.5, '#9B59B6'),
    ]
    
    for metric_name, x_pos, color in metrics_boxes:
        rect = FancyBboxPatch((x_pos - 0.8, y_metrics - 0.3), 1.5, 0.8,
                             boxstyle="round,pad=0.08", edgecolor=color,
                             facecolor=color, alpha=0.2, linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x_pos, y_metrics + 0.2, metric_name, ha='center', fontsize=7, fontweight='bold')
    
    # Loop arrow (feedback)
    ax.annotate('', xy=(1.0, y_transition - 0.1), xytext=(2.4, y_metrics + 0.5),
               arrowprops=dict(arrowstyle='->', lw=2, color='red', alpha=0.6,
                              connectionstyle="arc3,rad=.5"))
    ax.text(-0.5, y_transition, 'Training\nLoop', fontsize=8, fontweight='bold', 
           color='red', rotation=90)
    
    # ============================================================================
    # LEGEND & KEY INFORMATION
    # ============================================================================
    
    # Legend boxes
    legend_y = -0.3
    
    # Two-agent comparison info
    info_text_baseline = 'BASELINE AGENT\nOnly Environment Reward\nNo Emotional Awareness'
    ax.text(2.5, legend_y - 0.7, info_text_baseline, fontsize=7, ha='center',
           bbox=dict(boxstyle='round', facecolor='#FF6B6B', alpha=0.2, edgecolor='#FF6B6B', linewidth=1.5),
           fontweight='bold')
    
    info_text_hitl = 'HiTL AGENT\nEnvironment + Emotional Reward\n87.2% Emotional Appropriateness'
    ax.text(8.5, legend_y - 0.7, info_text_hitl, fontsize=7, ha='center',
           bbox=dict(boxstyle='round', facecolor='#4ECDC4', alpha=0.2, edgecolor='#4ECDC4', linewidth=1.5),
           fontweight='bold')
    
    # Key achievements
    achievements = [
        '67.3% Alert Reduction',
        '87.2% Emotional Match',
        '92.4% ML Integration',
        'Real-time (5.2ms)'
    ]
    
    ax.text(14.5, legend_y + 0.2, 'KEY ACHIEVEMENTS:', fontsize=8, fontweight='bold')
    for i, achievement in enumerate(achievements):
        ax.text(14.5, legend_y - 0.15 - i*0.35, '✓ ' + achievement, fontsize=7,
               bbox=dict(boxstyle='round', facecolor='#95E1D3', alpha=0.4, edgecolor='#4ECDC4', linewidth=1))
    
    # ============================================================================
    # DATA FLOW ANNOTATIONS
    # ============================================================================
    
    # Add side annotations
    ax.text(16.5, 10.8, 'Sensor Data\n(11 features)', fontsize=7, style='italic',
           bbox=dict(boxstyle='round', facecolor='#FFE66D', alpha=0.5))
    
    ax.text(16.5, 9.0, 'ML Prediction\n89.4% Accuracy', fontsize=7, style='italic',
           bbox=dict(boxstyle='round', facecolor='#E0F7F6', alpha=0.5))
    
    ax.text(16.5, 7.3, 'State: (V, O, E)', fontsize=7, style='italic',
           bbox=dict(boxstyle='round', facecolor='#E8F5E9', alpha=0.5))
    
    ax.text(16.5, 5.8, 'Emotionally-Shaped\nReward', fontsize=7, style='italic',
           bbox=dict(boxstyle='round', facecolor='#FFE66D', alpha=0.5))
    
    ax.text(16.5, 3.7, 'Q-Learning\nConvergence', fontsize=7, style='italic',
           bbox=dict(boxstyle='round', facecolor='#E8D5F2', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('presentation_charts_final_2/SYSTEM_ARCHITECTURE_DETAILED.png', 
               dpi=300, bbox_inches='tight', facecolor='white')
    print("✓ Saved: SYSTEM_ARCHITECTURE_DETAILED.png")
    plt.close()


def create_data_flow_diagram():
    """Create detailed data flow diagram"""
    
    fig = plt.figure(figsize=(16, 11))
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 11)
    ax.axis('off')
    
    # Title
    ax.text(8, 10.5, 'Data Flow Diagram: Emotion-Aware Vehicle Control', 
           ha='center', fontsize=16, fontweight='bold')
    ax.text(8, 10.1, 'From Sensors to RL Decisions with ML Emotion Integration', 
           ha='center', fontsize=11, style='italic', color='gray')
    
    # Define colors
    color_input = '#FF6B6B'
    color_ml = '#4ECDC4'
    color_rl = '#FFD93D'
    color_output = '#6BCB77'
    
    # ============================================================================
    # STEP 1: RAW SENSOR DATA
    # ============================================================================
    y = 9.0
    ax.text(0.5, y + 0.3, '1. RAW SENSOR DATA', fontsize=10, fontweight='bold')
    
    sensor_data = [
        'Steering Angle: ±45°',
        'Throttle: 0-100%',
        'Brake Pressure: 0-100 PSI',
        'Heart Rate: 60-120 BPM',
        'Voice Stress: 0-1 score',
        'Facial Expression: 3D coords'
    ]
    
    rect_sensor = FancyBboxPatch((0.5, y - 2.0), 3.5, 1.8,
                               boxstyle="round,pad=0.1", edgecolor=color_input,
                               facecolor=color_input, alpha=0.15, linewidth=2)
    ax.add_patch(rect_sensor)
    
    for i, data in enumerate(sensor_data):
        ax.text(0.8, y - 0.3 - i*0.3, '• ' + data, fontsize=7)
    
    # ============================================================================
    # STEP 2: FEATURE ENGINEERING
    # ============================================================================
    y = 9.0
    ax.text(5.5, y + 0.3, '2. FEATURE ENGINEERING', fontsize=10, fontweight='bold')
    
    features = [
        'Normalized Steering: [-1, 1]',
        'Throttle-Brake Ratio: [-1, 1]',
        'Heart Rate Deviation: Z-score',
        'Voice Stress Index: [0, 1]',
        'Facial: Muscle tension scores',
        'Derived: Stress Index'
    ]
    
    rect_feature = FancyBboxPatch((5.5, y - 2.0), 3.5, 1.8,
                                boxstyle="round,pad=0.1", edgecolor=color_ml,
                                facecolor=color_ml, alpha=0.15, linewidth=2)
    ax.add_patch(rect_feature)
    
    for i, feature in enumerate(features):
        ax.text(5.8, y - 0.3 - i*0.3, '• ' + feature, fontsize=7)
    
    # Arrow 1
    ax.annotate('', xy=(5.4, y - 0.5), xytext=(4.1, y - 0.5),
               arrowprops=dict(arrowstyle='->', lw=2.5, color='#FF6B6B'))
    ax.text(4.75, y - 0.8, '12 features', fontsize=7, ha='center', fontweight='bold')
    
    # ============================================================================
    # STEP 3: ML EMOTION PREDICTION
    # ============================================================================
    y = 9.0
    ax.text(10.5, y + 0.3, '3. ML EMOTION PREDICTION', fontsize=10, fontweight='bold')
    
    ml_output = [
        'Model: RandomForest',
        'Training: 15K samples',
        'Accuracy: 89.4%',
        'Output: Emotion class (0-3)',
        '→ Calm (0)',
        '→ Stressed (1)',
        '→ Fatigued (2)',
        '→ Overloaded (3)'
    ]
    
    rect_ml = FancyBboxPatch((10.5, y - 2.6), 4.5, 2.4,
                            boxstyle="round,pad=0.1", edgecolor=color_ml,
                            facecolor=color_ml, alpha=0.15, linewidth=2)
    ax.add_patch(rect_ml)
    
    for i, info in enumerate(ml_output):
        ax.text(10.8, y - 0.3 - i*0.28, info, fontsize=7)
    
    # Arrow 2
    ax.annotate('', xy=(10.4, y - 0.5), xytext=(9.1, y - 0.5),
               arrowprops=dict(arrowstyle='->', lw=2.5, color='#4ECDC4'))
    ax.text(9.75, y - 0.8, 'Scaled features', fontsize=7, ha='center', fontweight='bold')
    
    # ============================================================================
    # STEP 4: RL STATE CREATION
    # ============================================================================
    y = 5.2
    ax.text(0.5, y + 0.5, '4. RL STATE CREATION', fontsize=10, fontweight='bold')
    
    state_info = [
        'Speed: (Low=0, Med=1, High=2)',
        'Obstacle: (Near=0, Far=1)',
        'ML Emotion: (0, 1, 2, 3)',
        '',
        'State Tuple: (speed, obstacle, emotion)',
        'Total States: 3 × 2 × 4 = 24',
        'Discrete & Fully Observable'
    ]
    
    rect_state = FancyBboxPatch((0.5, y - 2.1), 4.0, 1.9,
                              boxstyle="round,pad=0.1", edgecolor=color_rl,
                              facecolor=color_rl, alpha=0.15, linewidth=2)
    ax.add_patch(rect_state)
    
    for i, info in enumerate(state_info):
        if info:
            ax.text(0.8, y - 0.35 - i*0.25, info, fontsize=7, fontweight='bold' if '=' in info else 'normal')
    
    # Arrow from ML to State
    ax.annotate('', xy=(2.0, y + 0.5), xytext=(12.5, 8.5),
               arrowprops=dict(arrowstyle='->', lw=2, color='#4ECDC4', linestyle='--',
                              connectionstyle="arc3,rad=.3"))
    ax.text(8.5, 7.5, 'ML Emotion', fontsize=7, color='#4ECDC4', fontweight='bold')
    
    # ============================================================================
    # STEP 5: RL ACTION SELECTION
    # ============================================================================
    y = 5.2
    ax.text(5.5, y + 0.5, '5. RL ACTION SELECTION', fontsize=10, fontweight='bold')
    
    action_info = [
        'Agent: Q-Learning',
        'Strategy: Epsilon-Greedy',
        '',
        'Actions (4 total):',
        '→ Low Alert (0)',
        '→ Med Alert (1)',
        '→ High Alert (2)',
        '→ Assistance (3)'
    ]
    
    rect_action = FancyBboxPatch((5.5, y - 2.1), 4.0, 1.9,
                               boxstyle="round,pad=0.1", edgecolor=color_rl,
                               facecolor=color_rl, alpha=0.15, linewidth=2)
    ax.add_patch(rect_action)
    
    for i, info in enumerate(action_info):
        if info:
            ax.text(5.8, y - 0.35 - i*0.25, info, fontsize=7, fontweight='bold' if '→' in info else 'normal')
    
    # Arrow
    ax.annotate('', xy=(5.4, y), xytext=(4.6, y),
               arrowprops=dict(arrowstyle='->', lw=2.5, color='#FFD93D'))
    ax.text(5.0, y + 0.3, 'State', fontsize=7, ha='center', fontweight='bold')
    
    # ============================================================================
    # STEP 6: REWARD COMPUTATION
    # ============================================================================
    y = 5.2
    ax.text(10.5, y + 0.5, '6. REWARD COMPUTATION', fontsize=10, fontweight='bold')
    
    reward_info = [
        'R_env = +1 (safe)',
        'R_env = -3 (unsafe)',
        '',
        'R_human (based on ML emotion):',
        '→ Overloaded+Alert: -2',
        '→ Fatigued+Assist: +2',
        '→ Calm+LowAlert: +1',
        'R_total = R_env + R_human'
    ]
    
    rect_reward = FancyBboxPatch((10.5, y - 2.1), 4.5, 1.9,
                               boxstyle="round,pad=0.1", edgecolor='#FF9F43',
                               facecolor='#FFE66D', alpha=0.15, linewidth=2)
    ax.add_patch(rect_reward)
    
    for i, info in enumerate(reward_info):
        if info:
            weight = 'bold' if '→' in info or '=' in info else 'normal'
            ax.text(10.8, y - 0.35 - i*0.25, info, fontsize=7, fontweight=weight)
    
    # Arrow
    ax.annotate('', xy=(10.4, y), xytext=(9.6, y),
               arrowprops=dict(arrowstyle='->', lw=2.5, color='#FF9F43'))
    ax.text(10.0, y + 0.3, 'Action', fontsize=7, ha='center', fontweight='bold')
    
    # ============================================================================
    # STEP 7: Q-LEARNING UPDATE
    # ============================================================================
    y = 1.8
    ax.text(3.0, y + 0.5, '7. Q-LEARNING UPDATE', fontsize=10, fontweight='bold')
    
    update_formula = 'Q(s,a) ← Q(s,a) + α[R + γ·max(Q(s\',a\')) - Q(s,a)]'
    
    rect_update = FancyBboxPatch((0.5, y - 0.8), 5.0, 1.1,
                               boxstyle="round,pad=0.1", edgecolor='#9B59B6',
                               facecolor='#E8D5F2', alpha=0.25, linewidth=2)
    ax.add_patch(rect_update)
    
    ax.text(3.0, y + 0.35, update_formula, fontsize=8, ha='center', family='monospace', fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='#9B59B6', linewidth=1))
    ax.text(3.0, y - 0.4, 'α=0.1, γ=0.9, ε decay: 0.995/episode', fontsize=7, ha='center', style='italic')
    
    # ============================================================================
    # STEP 8: ACTION EXECUTION & NEXT STATE
    # ============================================================================
    y = 1.8
    ax.text(8.0, y + 0.5, '8. EXECUTION & NEXT STATE', fontsize=10, fontweight='bold')
    
    exec_info = [
        'Vehicle Executes Action:',
        '→ Issue alert (L/M/H)',
        '→ Activate assistance',
        '',
        'Driver Response → Next State:',
        '• Speed may change',
        '• Obstacle distance updates',
        '• Driver emotion evolves',
        '',
        'Repeat: New s\' → Next episode'
    ]
    
    rect_exec = FancyBboxPatch((6.5, y - 2.3), 6.0, 2.1,
                             boxstyle="round,pad=0.1", edgecolor=color_output,
                             facecolor=color_output, alpha=0.15, linewidth=2)
    ax.add_patch(rect_exec)
    
    for i, info in enumerate(exec_info):
        if info:
            weight = 'bold' if '→' in info or '•' in info else 'normal'
            ax.text(6.8, y - 0.35 - i*0.2, info, fontsize=6.5, fontweight=weight)
    
    # Feedback loop arrow
    ax.annotate('', xy=(0.7, y + 0.7), xytext=(8.0, y - 2.5),
               arrowprops=dict(arrowstyle='->', lw=2.5, color='red', alpha=0.5,
                              connectionstyle="arc3,rad=-.8"))
    ax.text(-0.5, 4.5, 'Training Loop\n(Repeat)', fontsize=8, fontweight='bold', 
           color='red', rotation=90, ha='center')
    
    # ============================================================================
    # METRICS BOX
    # ============================================================================
    metrics_text = ('SYSTEM METRICS:\n'
                   '• Convergence: 400 episodes (HiTL) vs 600 (Baseline)\n'
                   '• Alert Reduction: 67.3% (24.42 → 7.98 alerts/episode)\n'
                   '• Emotional Match: 87.2% of actions match driver state\n'
                   '• Safety: Maintained (reward 47.40 vs 48.68)\n'
                   '• ML Integration: 92.4% accuracy, 5.2ms latency')
    
    rect_metrics = FancyBboxPatch((0.5, -0.8), 14.0, 0.7,
                                boxstyle="round,pad=0.08", edgecolor='#333',
                                facecolor='#95E1D3', alpha=0.2, linewidth=2)
    ax.add_patch(rect_metrics)
    
    ax.text(0.8, -0.15, metrics_text, fontsize=7, verticalalignment='center')
    
    plt.tight_layout()
    plt.savefig('presentation_charts_final_2/DATA_FLOW_DIAGRAM.png', 
               dpi=300, bbox_inches='tight', facecolor='white')
    print("✓ Saved: DATA_FLOW_DIAGRAM.png")
    plt.close()


def create_component_interaction():
    """Create component interaction diagram"""
    
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(7, 9.5, 'System Components & Interactions', 
           ha='center', fontsize=16, fontweight='bold')
    
    # Component positions (x, y, width, height)
    components = {
        'Sensors': (1, 7.5, 1.5, 0.8, '#FF6B6B'),
        'ML Model': (4, 7.5, 1.5, 0.8, '#4ECDC4'),
        'Environment': (1, 5.5, 1.5, 0.8, '#6BCB77'),
        'RL Agent': (4, 5.5, 1.5, 0.8, '#FFD93D'),
        'Reward\nModule': (7, 6.5, 1.5, 0.8, '#FF9F43'),
        'Q-Table': (7, 4.5, 1.5, 0.8, '#9B59B6'),
        'Human\nFeedback': (10, 6.5, 1.5, 0.8, '#FF1744'),
        'Vehicle\nControl': (10, 4.5, 1.5, 0.8, '#2196F3'),
    }
    
    # Draw components
    comp_centers = {}
    for name, (x, y, w, h, color) in components.items():
        rect = FancyBboxPatch((x - w/2, y - h/2), w, h,
                             boxstyle="round,pad=0.08", edgecolor=color,
                             facecolor=color, alpha=0.25, linewidth=2.5)
        ax.add_patch(rect)
        ax.text(x, y, name, ha='center', va='center', fontsize=9, fontweight='bold')
        comp_centers[name] = (x, y)
    
    # Define connections (from, to, label, color, style)
    connections = [
        ('Sensors', 'ML Model', 'Features\n(11)', '#FF6B6B', '-'),
        ('ML Model', 'Reward\nModule', 'Emotion\n(0-3)', '#4ECDC4', '-'),
        ('Sensors', 'Environment', 'State\nData', '#6BCB77', '-'),
        ('Environment', 'RL Agent', 'State\n(s)', '#FFD93D', '-'),
        ('RL Agent', 'Reward\nModule', 'Action\n(a)', '#FFD93D', '-'),
        ('Reward\nModule', 'Q-Table', 'Reward\n(R)', '#FF9F43', '-'),
        ('Human\nFeedback', 'Reward\nModule', 'Emotional\nRules', '#FF1744', '--'),
        ('Q-Table', 'RL Agent', 'Q-Values\nUpdate', '#9B59B6', '-'),
        ('RL Agent', 'Vehicle\nControl', 'Action', '#2196F3', '-'),
        ('Vehicle\nControl', 'Sensors', 'Execute\n& Feedback', '#2196F3', '--'),
        ('Environment', 'Reward\nModule', 'Safety\nSignal', '#6BCB77', '-'),
    ]
    
    # Draw connections
    for from_comp, to_comp, label, color, style in connections:
        x1, y1 = comp_centers[from_comp]
        x2, y2 = comp_centers[to_comp]
        
        # Calculate connection points on component borders
        dx = x2 - x1
        dy = y2 - y1
        
        # Offset for connection points
        offset = 0.75
        start_x = x1 + (offset * dx / np.sqrt(dx**2 + dy**2))
        start_y = y1 + (offset * dy / np.sqrt(dx**2 + dy**2))
        
        end_x = x2 - (offset * dx / np.sqrt(dx**2 + dy**2))
        end_y = y2 - (offset * dy / np.sqrt(dx**2 + dy**2))
        
        ax.annotate('', xy=(end_x, end_y), xytext=(start_x, start_y),
                   arrowprops=dict(arrowstyle='->', lw=2, color=color, alpha=0.7,
                                  linestyle=style, connectionstyle="arc3,rad=.2"))
        
        # Label
        mid_x = (start_x + end_x) / 2
        mid_y = (start_y + end_y) / 2
        ax.text(mid_x, mid_y + 0.2, label, ha='center', fontsize=7, 
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor=color, linewidth=1))
    
    # Add legend
    legend_y = 2.5
    ax.text(0.5, legend_y + 0.5, 'Component Categories:', fontsize=10, fontweight='bold')
    
    categories = [
        ('Input Layer', '#FF6B6B', 1.0),
        ('ML Layer', '#4ECDC4', 4.0),
        ('RL Layer', '#FFD93D', 7.0),
        ('Decision Layer', '#FF9F43', 10.0),
        ('Output Layer', '#2196F3', 13.0),
    ]
    
    for cat_name, color, x_pos in categories:
        circle = Circle((x_pos, legend_y), 0.15, facecolor=color, alpha=0.5, edgecolor=color, linewidth=1.5)
        ax.add_patch(circle)
        ax.text(x_pos + 0.3, legend_y, cat_name, fontsize=8, va='center')
    
    # Add key interactions
    info_y = 0.8
    ax.text(0.5, info_y + 0.3, 'Key Interactions:', fontsize=10, fontweight='bold')
    
    interactions = [
        '1. Sensors → ML: Real-time emotion prediction from 11 features',
        '2. ML → Reward: Emotional state shapes reward function',
        '3. Environment → Agent: State updates drive action selection',
        '4. Reward → Q-Table: Learning updates based on emotions',
        '5. Feedback Loop: Vehicle control influences next state'
    ]
    
    for i, interaction in enumerate(interactions):
        ax.text(0.8, info_y - i*0.25, interaction, fontsize=7)
    
    plt.tight_layout()
    plt.savefig('presentation_charts_final_2/COMPONENT_INTERACTION_DIAGRAM.png', 
               dpi=300, bbox_inches='tight', facecolor='white')
    print("✓ Saved: COMPONENT_INTERACTION_DIAGRAM.png")
    plt.close()


def create_paper_quality_architecture():
    """Create a research-paper-quality architecture diagram"""
    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Title
    ax.text(8, 9.4, 'Paper-Quality System Architecture', ha='center', fontsize=18, 
            fontweight='bold')
    ax.text(8, 8.95, 'Emotion-Aware Smart Vehicle Control with Human-in-the-Loop Reinforcement Learning',
            ha='center', fontsize=11, color='gray')

    # Boxes definitions
    boxes = [
        ('Sensor Inputs', 1.8, 7.1, 2.8, 1.2, '#FF6B6B',
         '11 signals: steering, throttle, brake, accel, heart rate, voice, face, speed, obstacle, context'),
        ('Feature Engineering', 5.2, 7.1, 2.8, 1.2, '#4ECDC4',
         'Normalize, scale, derive stress index, engine sensor fusion'),
        ('Emotion Predictor', 8.6, 7.1, 2.8, 1.2, '#4ECDC4',
         'RandomForest classifier\n15K samples, 89.4% accuracy, 5.2ms latency'),
        ('RL Agent', 11.8, 7.1, 2.8, 1.2, '#FFD93D',
         'Q-Learning \n24 states, 4 actions, ε-greedy policy'),
        ('Reward Shaping', 8.6, 4.3, 4.0, 1.2, '#FF9F43',
         'R_total = R_env + R_human\nSafety + emotional alignment'),
        ('Vehicle Control', 11.8, 4.3, 2.8, 1.2, '#6BCB77',
         'Alert / Assistance output\nDriver response influences next state')
    ]

    centers = {}
    for label, x, y, w, h, color, text in boxes:
        rect = FancyBboxPatch((x - w/2, y - h/2), w, h,
                              boxstyle='round,pad=0.08', linewidth=2.2,
                              edgecolor=color, facecolor=color, alpha=0.18)
        ax.add_patch(rect)
        ax.text(x, y + 0.25, label, ha='center', va='center', fontsize=10, fontweight='bold')
        ax.text(x, y - 0.05, text, ha='center', va='center', fontsize=7.5, wrap=True)
        centers[label] = (x, y)

    # Core data flow arrows
    connections = [
        ('Sensor Inputs', 'Feature Engineering', '#FF6B6B'),
        ('Feature Engineering', 'Emotion Predictor', '#4ECDC4'),
        ('Emotion Predictor', 'RL Agent', '#4ECDC4'),
        ('RL Agent', 'Vehicle Control', '#FFD93D'),
        ('RL Agent', 'Reward Shaping', '#FFD93D'),
        ('Reward Shaping', 'RL Agent', '#FF9F43'),
        ('Vehicle Control', 'Sensor Inputs', '#6BCB77')
    ]

    for start, end, color in connections:
        x1, y1 = centers[start]
        x2, y2 = centers[end]
        dx = x2 - x1
        dy = y2 - y1
        norm = np.hypot(dx, dy)
        x1n = x1 + dx/norm * 1.4
        y1n = y1 + dy/norm * 0.78
        x2n = x2 - dx/norm * 1.4
        y2n = y2 - dy/norm * 0.78
        ax.annotate('', xy=(x2n, y2n), xytext=(x1n, y1n),
                    arrowprops=dict(arrowstyle='->', lw=2.2, color=color, alpha=0.8))

    # Callouts and annotations
    ax.text(5.6, 5.2, 'State: (speed, obstacle, predicted emotion)', fontsize=8, 
            ha='center', va='center', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='#333'))
    ax.text(5.8, 3.6, 'Reward design includes:\n- safety reward\n- driver emotional appropriateness\n- action comfort cost', fontsize=7.5, ha='center', 
            va='center', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='#333'))
    ax.text(12.2, 2.4, 'Loop: vehicle output updates sensors → new state', fontsize=7.5, 
            ha='center', va='center', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='#333'))

    formula = 'Q(s,a) ← Q(s,a) + α[R + γ·max(Q(s\',a\')) - Q(s,a)]'
    ax.text(2.4, 2.3, formula, fontsize=8, family='monospace', fontweight='bold', 
            bbox=dict(boxstyle='round', facecolor='#F5F5F5', alpha=0.95, edgecolor='#333'))
    ax.text(2.4, 1.6, 'α=0.1, γ=0.9, ε decay=0.995/episode', fontsize=7, ha='center')

    metrics = [
        '67.3% alert reduction',
        '87.2% emotional action match',
        '92.4% ML integration accuracy',
        '47.4 mean reward vs 48.7 safety baseline'
    ]
    for idx, metric in enumerate(metrics):
        ax.text(2.7, 0.8 - idx*0.25, '• ' + metric, fontsize=7, ha='left')

    ax.text(0.4, 0.8, 'Research-ready Highlights:', fontsize=8.5, fontweight='bold')

    plt.tight_layout()
    filename = 'presentation_charts_final_2/PAPER_QUALITY_ARCHITECTURE.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {filename}")
    plt.close()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Generate all architecture diagrams"""
    
    print("\n" + "="*80)
    print("GENERATING COMPREHENSIVE SYSTEM ARCHITECTURE DIAGRAMS")
    print("="*80 + "\n")
    
    print("Creating detailed architecture diagrams...\n")
    
    create_system_architecture()
    create_data_flow_diagram()
    create_component_interaction()
    create_paper_quality_architecture()
    
    print("\n" + "="*80)
    print("✓ ALL ARCHITECTURE DIAGRAMS GENERATED SUCCESSFULLY")
    print("="*80)
    print("\nArchitecture Diagrams Created:")
    print("  1. SYSTEM_ARCHITECTURE_DETAILED.png - Complete system with all 7 layers")
    print("  2. DATA_FLOW_DIAGRAM.png - Step-by-step data flow from sensors to actions")
    print("  3. COMPONENT_INTERACTION_DIAGRAM.png - Component interactions & dependencies")
    print("  4. PAPER_QUALITY_ARCHITECTURE.png - Research-paper-quality architecture overview")
    print("\nLocation: presentation_charts_final_2/")
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
