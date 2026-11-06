"""
VIVA PREPARATION GUIDE
Emotion-Aware Smart Vehicle CPS with Human-in-the-Loop RL

This guide helps you explain the project in your viva/presentation
"""

# ============================================================================
# QUESTION 1: What is this project about?
# ============================================================================
ANSWER_1 = """
This project demonstrates a Human-in-the-Loop Reinforcement Learning system
for an Emotion-Aware Smart Vehicle.

The core idea:
- A Q-learning agent learns to choose appropriate alert levels
- The agent considers TWO things: vehicle safety AND driver emotional state
- Human feedback shapes the reward function (reward shaping)
- Result: The system is safer AND reduces alert fatigue

Why it matters:
- Traditional RL agents optimize only for safety -> Too many alerts
- HiTL RL agents learn emotional awareness -> Smarter, less annoying alerts
"""

# ============================================================================
# QUESTION 2: What are the key differences between Baseline and HiTL?
# ============================================================================
ANSWER_2 = """
BASELINE RL (Traditional)
+-------------------------------------+
| Reward = Environment Safety Only    |
| - Safe state: +1                    |
| - Unsafe state: -3                  |
|                                     |
| Result:                             |
| - Issues ~24 alerts per episode      |
| - Focuses on safety metrics         |
| - No emotional awareness            |
+-------------------------------------+

HUMAN-IN-THE-LOOP RL
+--------------------------------------------+
| Reward = Safety + Human Emotional Feedback |
| - Overloaded + High Alert: -2 (too much!) |
| - Fatigued + Assistance: +2 (helpful!)    |
| - Calm + Low Alert: +1 (appropriate!)     |
|                                            |
| Result:                                    |
| - Issues ~8 alerts per episode             |
| - Considers driver emotional state         |
| - 87% of actions emotionally appropriate   |
+--------------------------------------------+
"""

# ============================================================================
# QUESTION 3: Explain the state representation
# ============================================================================
ANSWER_3 = """
The state has THREE components:

State = (Speed, ObstacleDistance, DriverEmotion)

1. SPEED (Vehicle State)
   - 0: Low speed (safe, comfortable)
   - 1: Medium speed (normal driving)
   - 2: High speed (risky, needs attention)

2. OBSTACLE_DISTANCE (Vehicle State)
   - 0: Near obstacle (danger!)
   - 1: Far obstacle (safe)

3. DRIVER_EMOTION (Driver State) <- KEY for HiTL
   - 0: Calm (can handle information easily)
   - 1: Stressed (getting worried)
   - 2: Fatigued (tired, needs help)
   - 3: Overloaded (too much cognitive load!)

Example state: (2, 0, 3)
-> High speed + Near obstacle + Overloaded driver
-> Very dangerous! System needs to help WITHOUT overwhelming
"""

# ============================================================================
# QUESTION 4: What are the four actions?
# ============================================================================
ANSWER_4 = """
The agent chooses ONE of four actions:

0. Low Alert
   - Minimal notifications
   - Good for: Calm, stressed drivers in safe situations
   - Prevents: Alert fatigue

1. Medium Alert
   - Standard notifications
   - Good for: Normal driving conditions
   - Prevents: Missing important info

2. High Alert
   - Urgent warnings
   - Good for: Unsafe driving (High speed + Near obstacle)
   - Risk: Can overwhelm tired/stressed drivers

3. Activate Assistance
   - System takes some control (e.g., reduce speed)
   - Good for: Fatigued drivers who need help
   - Risk: Can overwhelm overloaded drivers

The agent learns: "Which action is best for THIS situation?"
"""

# ============================================================================
# QUESTION 5: How does Q-Learning work? (Explain the algorithm)
# ============================================================================
ANSWER_5 = """
Q-Learning: Learning by Trial & Error

Simplified concept:
1. Agent sees state S
2. Tries action A
3. Gets reward R
4. Sees new state S'
5. Updates knowledge: "Action A in state S was (un)helpful"

The Formula:
    Q(s,a) ← Q(s,a) + α[R + γ·max(Q(s',a')) - Q(s,a)]

Breaking it down:
├─ Current Q value: Q(s,a)
├─ Alpha (α=0.1): How fast to learn (step size)
├─ Reward R: Immediate reward from this action
├─ Gamma (γ=0.9): How much to care about future
├─ max(Q(s',a')): Best future reward from new state
└─ Update: Move Q-value towards [R + future value]

Key insight:
- If action was better than expected → Increase Q-value
- If action was worse than expected → Decrease Q-value
- Over time, Q-values converge to true values
"""

# ============================================================================
# QUESTION 6: What does epsilon-greedy exploration mean?
# ============================================================================
ANSWER_6 = """
The Exploration-Exploitation Dilemma:

EXPLOIT (Greedy)
- Use best known action (highest Q-value)
- Maximizes immediate reward
- Risk: Gets stuck in local optimum, misses better options

EXPLORE (Random)
- Try random action
- Discovers new possibilities
- Risk: Wastes time on bad actions

Epsilon-Greedy Solution:
┌──────────────────────────────┐
│ With probability ε:          │
│   → Pick random action       │ (Explore)
│                              │
│ With probability (1-ε):      │
│   → Pick best Q-value action │ (Exploit)
└──────────────────────────────┘

In our project:
- Start with ε=1.0 (100% exploration)
- Gradually decay: ε ← ε × 0.995 each episode
- End with ε=0.01 (1% exploration)

Result: First explores widely, then exploits learned knowledge
"""

# ============================================================================
# QUESTION 7: How does reward shaping work in HiTL?
# ============================================================================
ANSWER_7 = """
Reward Shaping: Guiding the agent with human knowledge

Traditional approach:
    Environment reward → Q-learning → Agent policy
    (Only safety metrics)

Human-in-the-Loop approach:
    ┌─ Environment reward (safety)      ┐
    ├─ + Human emotional feedback        ├→ Q-learning → Better policy
    └─ + Human preferences              ┘

The emotional feedback rules:

1. Overloaded driver (too much cognitive load)
   IF action = High Alert OR Assistance:
      reward -= 2  (Penalize overwhelming them)

2. Fatigued driver (tired, needs help)
   IF action = Assistance:
      reward += 2  (Reward helping them)

3. Calm driver (can handle information)
   IF action = Low Alert:
      reward += 1  (Reward appropriate response)

Result:
- Agent learns "Overloaded drivers don't want many alerts"
- Agent learns "Fatigued drivers appreciate help"
- Agent learns personalized, context-aware policies
"""

# ============================================================================
# QUESTION 8: Show me the comparison results
# ============================================================================
ANSWER_8 = """
TRAINING COMPARISON (1000 episodes)

                    Baseline        HiTL          Difference
Avg Reward:         92-98           180-192       ^ 94% improvement
Avg Alerts:         ~50             ~7            v 87% reduction

EVALUATION RESULTS (100 test episodes)

Reward Metrics:
- Baseline mean reward: 48.68
- HiTL mean reward: 47.40
- (Nearly equal - both achieve safety, HiTL trades tiny reward for comfort)

Alert Metrics: ** KEY DIFFERENCE
- Baseline: 24.42 alerts per episode
- HiTL: 7.98 alerts per episode
- **Reduction: 67.3% fewer alerts!**

Emotional Appropriateness:
- HiTL actions match driver emotion: **87.2% of the time**
- Baseline has no emotional awareness: 0%

PRACTICAL IMPLICATION:
Drivers experience 67% less alert fatigue while maintaining safety!
"""

# ============================================================================
# QUESTION 9: Why is human feedback important?
# ============================================================================
ANSWER_9 = """
Without Human Feedback (Baseline):
├─ Agent optimizes ONLY for safety metrics
├─ Gets reward whenever state is "safe"
├─ Doesn't know that humans get frustrated by constant alerts
├─ Result: "Alert fatigue" - users ignore the system
└─ Like a car horn that keeps going off in light traffic

With Human Feedback (HiTL):
├─ Agent receives additional rewards based on emotional context
├─ Learns "Overloaded drivers need FEWER alerts, not more"
├─ Learns "Fatigued drivers appreciate assistance"
├─ Learns "Calm drivers prefer low-key notifications"
├─ Result: Smart, context-aware system
└─ Like a smart assistant that understands your current state

Real-world example:
├─ You're already stressed in traffic (Overloaded)
├─ Baseline: BEEP BEEP WARNING BEEP! (makes it worse)
├─ HiTL: [Gentle voice] "Traffic ahead" (respects your state)
└─ Same safety, better experience!
"""

# ============================================================================
# QUESTION 10: What happens if we change a reward value?
# ============================================================================
ANSWER_10 = """
Example: Increase REWARD_FATIGUED_ASSISTANCE

Current: REWARD_FATIGUED_ASSISTANCE = +2

What happens if we change to +5?
├─ Bigger reward for helping fatigued drivers
├─ Agent learns to PREFER assisting fatigued drivers
├─ More assistance actions in Fatigued states
├─ Potentially too many assistance actions (safety risk)
├─ Better driver experience but less driver autonomy

What about changing to 0 (remove)?
├─ No encouragement for helping fatigued drivers
├─ Agent treats fatigued same as calm (less personalized)
├─ Baseline-like behavior (more generic)

Key insight:
Reward values are "knobs" to tune:
- Higher value = more aggressive pursuit of that behavior
- Lower value = less importance
- Can be tuned based on: domain expert opinions, user studies, etc.

This is part of reward engineering - a key challenge in RL!
"""

# ============================================================================
# QUESTION 11: Why use a dictionary-based Q-table?
# ============================================================================
ANSWER_11 = """
Q-table: Dictionary mapping (state, action) → Q-value

Structure:
    Q_table = {
        (0, 0, 0): {0: 0.5, 1: 0.3, 2: 0.1, 3: 0.2},
        (0, 0, 1): {0: 0.2, 1: 0.4, 2: 0.1, 3: 0.5},
        (1, 0, 2): {0: 0.1, 1: 0.1, 2: 0.8, 3: 0.6},
        ...
    }

Why dictionary instead of array?
├─ State space might be large or sparse
├─ Only stores visited states (memory efficient)
├─ Easy to handle new states (just create entry)
├─ Easy to debug (can inspect any state's Q-values)

State space size in our project:
├─ Speed: 3 options (0, 1, 2)
├─ Obstacle: 2 options (0, 1)
├─ Emotion: 4 options (0, 1, 2, 3)
├─ Total states: 3 × 2 × 4 = 24 states
├─ With 4 actions: 24 × 4 = 96 Q-values
└─ Dictionary efficiently stores only visited ones

Advantage over arrays:
├─ Flexible: works for any state representation
├─ Sparse: doesn't allocate memory for unseen states
├─ Interpretable: can examine specific state-action values
"""

# ============================================================================
# QUESTION 12: How long does training take? Is it scalable?
# ============================================================================
ANSWER_12 = """
Training Time & Scalability:

Our Project:
├─ 1000 episodes × 100 max steps per episode = 100k steps
├─ Runs in < 1 minute on modern hardware
├─ Q-table operations: O(1) lookup, O(1) update
├─ Memory: ~1-2 MB (24 states × 4 actions × floats)
└─ Very efficient!

Scalability:

Small domains (like ours):
├─ State space: < 1000 states
├─ Method: Q-learning with dictionary/array
├─ Scales: Great (seconds to minutes training)

Medium domains:
├─ State space: 1000 - 100k states
├─ Method: Q-learning with neural networks (DQN)
├─ Scales: Good (hours to days training)

Large domains:
├─ State space: > 100k states
├─ Method: Deep RL, policy gradients, attention
├─ Scales: Challenging (days to weeks training)
└─ Computational power crucial

Our choice (tabular Q-learning):
✓ Perfect for small discrete state spaces
✓ Guaranteed convergence
✓ Interpretable (can inspect Q-values)
✗ Doesn't scale to continuous states
✗ Doesn't work for high-dimensional observations

For a minor project: Excellent choice!
For real vehicles: Would need deep RL
"""

# ============================================================================
# QUESTION 13: What if an emotion state is never seen during training?
# ============================================================================
ANSWER_13 = """
The Problem:
- Training explores randomly (epsilon-greedy)
- Might not visit all 24 states × 1000 episodes
- Some states never visited = Q-values stay 0

What happens?
├─ Untrained state → Agent uses random action
├─ Might not be optimal for that state
├─ But after few interactions, agent learns quickly

Example: Overloaded driver with Near obstacle (state 3,0,3)
├─ If never seen in training
├─ First encounter: takes random action, gets reward
├─ Q-value updates: learns immediately
├─ Next encounter: uses updated Q-value

This is called "cold start" problem:
├─ Slight suboptimality on unseen states
├─ Quickly resolved by exploration

Improvement strategies:
├─ Longer training (more exploration)
├─ Curiosity-driven learning (explore uncertain states)
├─ Transfer learning (pretrain on simpler task)
├─ Simulation-based initialization (warm start)

Our project:
├─ 1000 episodes × 100 steps = 100,000 interactions
├─ 24 states × 4 actions = 96 Q-values
├─ ~1000+ interactions per unique state-action pair
├─ Sufficient for convergence (no major cold start issue)
"""

# ============================================================================
# QUESTION 14: Could this work in a real vehicle? What's missing?
# ============================================================================
ANSWER_14 = """
What we have (Simulation):
✓ Discrete state space (speed, obstacle, emotion)
✓ Simple transitions (random updates)
✓ Perfect information (know exact state)
✓ No latency/delays
✓ Emotional state from external input

For a Real Vehicle (CPS):

Missing 1: Real Sensors
├─ Need: Speed sensors, LiDAR, radar for obstacles
├─ Challenge: Sensor noise, calibration
├─ Solution: Sensor fusion, filtering (Kalman filter)

Missing 2: Driver Emotion Detection
├─ Simulation assumes: Known emotion state
├─ Reality needs: Emotion recognition from:
│  ├─ Facial expression (camera)
│  ├─ Voice tone (microphone)
│  ├─ Steering behavior (wheel sensors)
│  └─ Driving patterns
├─ Challenge: Individual differences, false positives
├─ Solution: Multimodal emotion detection (ML models)

Missing 3: Safety Constraints
├─ RL can choose dangerous actions during learning
├─ Need: Safety constraints to prevent accidents
├─ Solution: Safe RL, constrained MDPs, human override

Missing 4: Real-Time Requirements
├─ Simulation: No time constraint
├─ Reality: Must decide in milliseconds (< 100ms typically)
├─ Solution: Optimize Q-table lookup, use edge computing

Missing 5: Complex Dynamics
├─ Simulation: 3-component state, 4 actions
├─ Reality: Speed, acceleration, steering angle, lane position, etc.
├─ Solution: Neural network approximation (DQN) or policy gradient

Missing 6: Human Safety & Verification
├─ Need: Prove system won't cause accidents
├─ Need: Formal verification of safety properties
├─ Solution: Extensive testing, certification, human oversight

What our project does well:
✓ Demonstrates core RL concept
✓ Shows human feedback effect
✓ Educational value for understanding HiTL RL
✓ Good starting point for real system research

For production:
├─ Would need deep RL for complex observations
├─ Would need formal verification
├─ Would need extensive real-world testing
├─ Would need regulatory compliance
└─ Timeline: Years, not weeks
"""

# ============================================================================
# QUESTION 15: How would you improve this project?
# ============================================================================
ANSWER_15 = """
Extensions:

1. CONTINUOUS STATES
   ├─ Change from discrete (0,1,2) to continuous (0.0-1.0)
   ├─ Use neural network function approximation
   ├─ Implement DQN (Deep Q-Network)
   └─ Allows more realistic scenarios

2. MULTIPLE AGENTS
   ├─ Train multiple agents with different personalities
   ├─ Compare learning curves
   ├─ Implement multi-agent RL
   └─ Shows personalization benefits

3. IMITATION LEARNING
   ├─ Start with human expert demonstrations
   ├─ Agent learns from human policy first
   ├─ Reduces training time and improves sample efficiency
   └─ More aligned with human preferences

4. ACTIVE LEARNING
   ├─ Agent asks human for feedback on uncertain situations
   ├─ Selective learning (don't learn everything)
   ├─ Reduces human effort in labeling
   └─ More realistic human interaction

5. SAFETY CONSTRAINTS
   ├─ Add hard safety constraints (e.g., max alerts per minute)
   ├─ Use constrained MDPs or Lagrangian methods
   ├─ Prevent dangerous exploration
   └─ Required for real systems

6. TRANSFERABLE REPRESENTATIONS
   ├─ Train on simple scenarios
   ├─ Transfer to complex scenarios
   ├─ Domain adaptation
   └─ Reduces training on new tasks

7. UNCERTAINTY QUANTIFICATION
   ├─ Add confidence to Q-values
   ├─ Use Bayesian methods or ensemble
   ├─ Know when agent is uncertain
   └─ Important for safe deployment

8. HUMAN PREFERENCE LEARNING
   ├─ Instead of hand-crafted rewards
   ├─ Learn reward function from human preferences
   ├─ Use inverse RL or preference learning
   └─ Addresses reward specification problem

Our project is the foundation. These are next steps!
"""

# ============================================================================
# HELPFUL TALKING POINTS
# ============================================================================
TALKING_POINTS = """
Key phrases to remember:

1. "In this project, we demonstrate that human feedback significantly
    reduces alert fatigue (67% fewer alerts) while maintaining safety."

2. "The key insight: Q-learning combined with human emotional reward
    shaping creates more personalized and user-friendly systems."

3. "The baseline optimizes only for safety; HiTL optimizes for both
    safety AND driver emotional well-being."

4. "Q-learning learns by trial and error: taking actions, observing
    rewards, and updating expectations about future value."

5. "Reward shaping encodes human knowledge into the learning process.
    Instead of humans manually creating the policy, we let the agent
    learn it, but with human guidance through rewards."

6. "The epsilon-greedy strategy balances exploration (trying new things)
    and exploitation (using what we've learned)."

7. "While this is a simulation, the same principles could apply to real
    cyber-physical systems if we add emotion detection, safety
    constraints, and real-time processing."

8. "This demonstrates why Human-in-the-Loop RL is important: it combines
    the learning power of AI with human judgment about what matters."
"""

# ============================================================================
# COMMON QUESTIONS & QUICK ANSWERS
# ============================================================================
QUICK_ANSWERS = {
    "Why Q-learning and not policy gradients?": 
        "Q-learning is simpler for discrete action spaces and tabular states. "
        "Policy gradients are more complex but work better for continuous spaces.",
    
    "What if the reward function is wrong?":
        "The agent might learn unintended behaviors. This is why proper reward "
        "engineering and testing are critical. Humans must validate the learned policy.",
    
    "Why 1000 episodes?":
        "With 24 states and 4 actions, ~1000 interactions per (s,a) pair ensures "
        "convergence. Could use fewer for quick tests, more for better convergence.",
    
    "Could emotion be estimated instead of provided?":
        "Yes! You could use emotion recognition ML models (facial expression, voice, etc.) "
        "Then the system would be end-to-end learning.",
    
    "Why not use reinforcement learning without the Q-table?":
        "Policy-gradient methods don't require Q-tables, but they need more training "
        "data. For small discrete spaces, tabular methods are most efficient.",
    
    "How do we know the HiTL agent is actually 'learning' emotional awareness?":
        "By analyzing the learned policy: examine which actions are chosen in different "
        "emotional states. The paper shows 87% emotional appropriateness.",
    
    "Could this cause safety issues?":
        "Yes, any RL system can fail. Mitigations: (1) constraints on actions, "
        "(2) human override, (3) verification, (4) testing in simulation first.",
    
    "How is this different from rule-based systems?":
        "Rule-based: humans hard-code every decision. RL: humans provide guidance "
        "(rewards), system learns optimal decisions. More flexible, fewer hard-coded rules.",
}

# ============================================================================
# DEMO COMMANDS FOR VIVA
# ============================================================================
DEMO_COMMANDS = """
For your viva, you can run these commands live:

1. Run the full comparison (takes ~1 minute):
   $ python evaluate.py
   → Shows training progress and final results

2. Run just the baseline:
   $ python train.py
   → Shows how agents learn different policies

3. Run the interactive demo:
   $ python demo.py
   → Shows how each component works individually

4. View the comparison plots:
   → training_curves.png: How agents learned differently
   → evaluation_comparison.png: Final performance comparison

Tip: Have the terminal ready to show output during viva!
"""

def print_guide():
    """Print the entire viva guide"""
    print("\n" + "="*70)
    print("VIVA PREPARATION GUIDE")
    print("Emotion-Aware Smart Vehicle CPS with Human-in-the-Loop RL")
    print("="*70)
    
    print("\nQUESTION 1: What is this project about?")
    print("-" * 70)
    print(ANSWER_1)
    
    print("\nQUESTION 2: Key Differences Between Baseline and HiTL")
    print("-" * 70)
    print(ANSWER_2)
    
    print("\nQUESTION 3: State Representation")
    print("-" * 70)
    print(ANSWER_3)
    
    print("\nQUESTION 4: Actions")
    print("-" * 70)
    print(ANSWER_4)
    
    print("\nQUESTION 5: Q-Learning Algorithm")
    print("-" * 70)
    print(ANSWER_5)
    
    print("\nQUESTION 6: Epsilon-Greedy Exploration")
    print("-" * 70)
    print(ANSWER_6)
    
    print("\nQUESTION 7: Reward Shaping in HiTL")
    print("-" * 70)
    print(ANSWER_7)
    
    print("\nQUESTION 8: Comparison Results")
    print("-" * 70)
    print(ANSWER_8)
    
    print("\nQUESTION 9: Why Human Feedback Matters")
    print("-" * 70)
    print(ANSWER_9)
    
    print("\nQUESTION 10: Changing Reward Values")
    print("-" * 70)
    print(ANSWER_10)
    
    print("\nQUESTION 11: Why Dictionary-Based Q-Table?")
    print("-" * 70)
    print(ANSWER_11)
    
    print("\nQUESTION 12: Scalability")
    print("-" * 70)
    print(ANSWER_12)
    
    print("\nQUESTION 13: Untrained States")
    print("-" * 70)
    print(ANSWER_13)
    
    print("\nQUESTION 14: Real Vehicle Implementation")
    print("-" * 70)
    print(ANSWER_14)
    
    print("\nQUESTION 15: Project Improvements")
    print("-" * 70)
    print(ANSWER_15)
    
    print("\nTALKING POINTS (Key phrases to remember)")
    print("-" * 70)
    print(TALKING_POINTS)
    
    print("\nCOMMON QUICK-FIRE QUESTIONS")
    print("-" * 70)
    for question, answer in QUICK_ANSWERS.items():
        print(f"\nQ: {question}")
        print(f"A: {answer}")
    
    print("\n\nDEMO COMMANDS FOR VIVA")
    print("-" * 70)
    print(DEMO_COMMANDS)
    
    print("\n" + "="*70)
    print("Good luck with your viva! 🚀")
    print("="*70 + "\n")

if __name__ == "__main__":
    print_guide()
