"""
Quick Demo: How to Use the Emotion-Aware Smart Vehicle CPS
Shows step-by-step how the system works
"""

from env import VehicleEnvironment
from agent import QLearningAgent
from human import HumanFeedback
from config import EMOTION_STATES, ACTIONS, NUM_ACTIONS


def demo_environment():
    """Demo 1: Understand the environment"""
    print("\n" + "="*60)
    print("DEMO 1: Vehicle Environment")
    print("="*60)
    
    env = VehicleEnvironment()
    
    print("\n🚗 Initializing environment...")
    state = env.reset()
    print(f"Initial state: {state}")
    print(f"  Speed: {state[0]} (0=Low, 1=Medium, 2=High)")
    print(f"  Obstacle Distance: {state[1]} (0=Near, 1=Far)")
    print(f"  Driver Emotion: {state[2]} (0=Calm, 1=Stressed, 2=Fatigued, 3=Overloaded)")
    
    print("\n📍 Taking random actions and observing transitions...")
    for step in range(3):
        action = step % NUM_ACTIONS  # Cycle through actions
        action_names = {v: k for k, v in ACTIONS.items()}
        print(f"\n  Step {step + 1}: Taking action '{action_names[action]}'")
        
        next_state, env_reward, done = env.step(action)
        print(f"    New state: {next_state}")
        print(f"    Safety reward: {env_reward}")


def demo_agent():
    """Demo 2: Understand Q-Learning Agent"""
    print("\n" + "="*60)
    print("DEMO 2: Q-Learning Agent")
    print("="*60)
    
    agent = QLearningAgent()
    
    # Create some fake states for demo
    demo_state = (1, 0, 2)  # Medium speed, Near obstacle, Fatigued driver
    
    print(f"\n🤖 Initializing Q-Learning agent...")
    print(f"   Learning rate: {agent.learning_rate}")
    print(f"   Discount factor: {agent.discount_factor}")
    print(f"   Initial epsilon: {agent.epsilon:.3f}")
    
    print(f"\n📊 Q-values for state {demo_state}:")
    action_names = {v: k for k, v in ACTIONS.items()}
    for action in range(NUM_ACTIONS):
        q_val = agent.get_q_value(demo_state, action)
        print(f"   {action_names[action]}: {q_val:.4f}")
    
    print(f"\n🎯 Agent selects action (with exploration):")
    for trial in range(3):
        action = agent.select_action(demo_state)
        print(f"   Trial {trial + 1}: {action_names[action]}")
    
    print(f"\n📈 Simulating Q-value update:")
    print(f"   State: {demo_state}, Action: 0 (Low alert)")
    print(f"   Reward received: +5")
    next_state = (1, 1, 1)
    print(f"   Next state: {next_state}")
    
    agent.update_q_value(demo_state, 0, 5, next_state)
    new_q = agent.get_q_value(demo_state, 0)
    print(f"   New Q-value: {new_q:.4f}")


def demo_human_feedback():
    """Demo 3: Understand Human Feedback System"""
    print("\n" + "="*60)
    print("DEMO 3: Human Feedback (Emotional Reward Shaping)")
    print("="*60)
    
    human = HumanFeedback()
    emotion_names = {v: k for k, v in EMOTION_STATES.items()}
    action_names = {v: k for k, v in ACTIONS.items()}
    
    print("\n😊 Testing emotional feedback for different state-action pairs:")
    
    test_cases = [
        ((0, 1, 0), 0, "Calm driver, Low alert"),           # Should be +1
        ((2, 0, 3), 3, "Overloaded driver, Assistance"),    # Should be -2
        ((1, 1, 2), 3, "Fatigued driver, Assistance"),      # Should be +2
        ((0, 0, 1), 1, "Stressed driver, Medium alert"),    # Should be 0
    ]
    
    for state, action, description in test_cases:
        reward = human.compute_emotional_reward(state, action)
        explanation = human.get_feedback_explanation(state, action, reward)
        print(f"\n  Case: {description}")
        print(f"    Emotional reward: {reward:+d}")
        print(f"    Explanation: {explanation}")


def demo_complete_episode():
    """Demo 4: Complete Episode Simulation"""
    print("\n" + "="*60)
    print("DEMO 4: Complete Training Episode (HiTL)")
    print("="*60)
    
    env = VehicleEnvironment()
    agent = QLearningAgent()
    human = HumanFeedback()
    
    state = env.reset()
    total_reward = 0
    
    emotion_names = {v: k for k, v in EMOTION_STATES.items()}
    action_names = {v: k for k, v in ACTIONS.items()}
    
    print(f"\n🎬 Running a 5-step training episode...")
    print(f"Initial state: Speed={state[0]}, Obstacle={state[1]}, Emotion={emotion_names[state[2]]}")
    
    for step in range(5):
        # Agent selects action
        action = agent.select_action(state)
        
        # Environment responds
        next_state, env_reward, done = env.step(action)
        
        # Human provides feedback
        emotional_reward = human.compute_emotional_reward(state, action)
        total_reward_this_step = env_reward + emotional_reward
        
        # Agent learns
        agent.update_q_value(state, action, total_reward_this_step, next_state)
        
        total_reward += total_reward_this_step
        
        print(f"\n  Step {step + 1}:")
        print(f"    Action: {action_names[action]}")
        print(f"    Safety reward: {env_reward:+d}, Emotional reward: {emotional_reward:+d}")
        print(f"    Total step reward: {total_reward_this_step:+d}")
        print(f"    New state: Speed={next_state[0]}, Obstacle={next_state[1]}, Emotion={emotion_names[next_state[2]]}")
        
        state = next_state
    
    print(f"\n✓ Episode complete! Total reward: {total_reward}")


def main():
    """Run all demos"""
    print("\n" + "█" * 60)
    print("EMOTION-AWARE SMART VEHICLE CPS - DEMO")
    print("█" * 60)
    
    demo_environment()
    demo_agent()
    demo_human_feedback()
    demo_complete_episode()
    
    print("\n" + "█" * 60)
    print("✓ All demos complete!")
    print("█" * 60)
    print("\nNext steps:")
    print("  1. Review how each component works")
    print("  2. Run 'python train.py' to train both agents")
    print("  3. Run 'python evaluate.py' to see full comparison")
    print("\nHappy learning! 🚀")


if __name__ == "__main__":
    main()
