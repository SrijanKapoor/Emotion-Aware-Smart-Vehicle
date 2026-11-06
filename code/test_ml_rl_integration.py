"""
Integration Validation Tests
Validates that ML model + RL agent integration works correctly.
"""

import numpy as np
from ml_rl_integration import (
    MLEmotionPredictor, 
    predict_emotion, 
    create_rl_state,
    compute_stress_index,
    SAMPLE_FEATURES,
    RAW_FEATURE_NAMES,
    ALL_FEATURE_NAMES
)
from agent import QLearningAgent
from config import EMOTION_STATES, ACTIONS


# ============================================================================
# TEST FUNCTIONS
# ============================================================================

def test_ml_model_loading():
    """Test 1: ML model loads correctly"""
    print("\n" + "="*70)
    print("TEST 1: ML Model Loading")
    print("="*70)
    
    try:
        predictor = MLEmotionPredictor()
        print(f"[PASS] Model loaded successfully")
        print(f"  - Model type: {type(predictor.model)}")
        print(f"  - Scaler: {type(predictor.scaler)}")
        print(f"  - Features: {len(predictor.feature_names)}")
        return True
    except Exception as e:
        print(f"[FAIL] Model loading failed: {e}")
        return False


def test_feature_validation():
    """Test 2: Feature validation and stress_index computation"""
    print("\n" + "="*70)
    print("TEST 2: Feature Validation & Stress Index")
    print("="*70)
    
    try:
        # Test with 11 features
        features = SAMPLE_FEATURES.copy()
        stress = compute_stress_index(features)
        
        assert 15 <= stress <= 70, f"Stress index out of range: {stress}"
        print(f"[PASS] Stress index computed: {stress:.2f}")
        
        # Test with invalid features
        try:
            compute_stress_index(features[:10])  # Too few
            print("[FAIL] Should reject <11 features")
            return False
        except:
            print(f"[PASS] Correctly rejects invalid feature count")
        
        return True
    except Exception as e:
        print(f"[FAIL] Feature validation failed: {e}")
        return False


def test_emotion_prediction():
    """Test 3: Emotion prediction"""
    print("\n" + "="*70)
    print("TEST 3: Emotion Prediction")
    print("="*70)
    
    try:
        predictor = MLEmotionPredictor()
        features = SAMPLE_FEATURES.copy()
        
        emotion_id = predictor.predict_emotion(features)
        
        assert emotion_id in EMOTION_STATES.values(), f"Invalid emotion ID: {emotion_id}"
        emotion_name = {v: k for k, v in EMOTION_STATES.items()}[emotion_id]
        
        print(f"[PASS] Emotion predicted: {emotion_name} (id={emotion_id})")
        
        # Test with random features
        random_features = np.random.uniform(
            [0, 0, 0, 0, 0, 1, 1, 0, 40, 0, 0],
            [100, 50, 150, 200, 3, 10, 10, 1, 150, 1, 2]
        )
        emotion_id2 = predictor.predict_emotion(random_features)
        print(f"[PASS] Random features predicted: {emotion_id2}")
        
        return True
    except Exception as e:
        print(f"[FAIL] Emotion prediction failed: {e}")
        return False


def test_rl_state_creation():
    """Test 4: RL state creation"""
    print("\n" + "="*70)
    print("TEST 4: RL State Creation")
    print("="*70)
    
    try:
        # Test valid states
        for speed in [0, 1, 2]:
            for obstacle in [0, 1]:
                for emotion in [0, 1, 2, 3]:
                    state = create_rl_state(speed, obstacle, emotion)
                    assert state == (speed, obstacle, emotion), f"State mismatch: {state}"
        
        print(f"[PASS] Created 4*2*3=24 valid RL states")
        return True
    except Exception as e:
        print(f"[FAIL] RL state creation failed: {e}")
        return False


def test_rl_agent_with_ml_emotion():
    """Test 5: RL agent integration with ML emotion"""
    print("\n" + "="*70)
    print("TEST 5: RL Agent + ML Integration")
    print("="*70)
    
    try:
        predictor = MLEmotionPredictor()
        agent = QLearningAgent()
        
        # Simulate 10 steps
        for step in range(10):
            # Get ML prediction
            features = SAMPLE_FEATURES + np.random.normal(0, 2, 11)
            features = np.maximum(features, 0)
            emotion = predictor.predict_emotion(features)
            
            # Create state
            speed = np.random.randint(0, 3)
            obstacle = np.random.randint(0, 2)
            state = create_rl_state(speed, obstacle, emotion)
            
            # Get action
            action = agent.select_action(state)
            
            assert 0 <= action < 4, f"Invalid action: {action}"
        
        print(f"[PASS] Completed 10 integration steps")
        print(f"  - Q-table entries: {len(agent.q_table)}")
        return True
    except Exception as e:
        print(f"[FAIL] RL agent integration failed: {e}")
        return False


def test_batch_prediction():
    """Test 6: Batch prediction"""
    print("\n" + "="*70)
    print("TEST 6: Batch Prediction")
    print("="*70)
    
    try:
        predictor = MLEmotionPredictor()
        
        emotions = []
        for i in range(10):
            features = SAMPLE_FEATURES + np.random.normal(0, 3, 11)
            features = np.maximum(features, 0)
            emotion = predictor.predict_emotion(features)
            emotions.append(emotion)
        
        emotion_counts = {e: emotions.count(e) for e in set(emotions)}
        print(f"[PASS] Predicted 10 emotions")
        print(f"  - Distribution: {emotion_counts}")
        return True
    except Exception as e:
        print(f"[FAIL] Batch prediction failed: {e}")
        return False


def test_edge_cases():
    """Test 7: Edge cases"""
    print("\n" + "="*70)
    print("TEST 7: Edge Cases")
    print("="*70)
    
    try:
        predictor = MLEmotionPredictor()
        
        # Very low stress
        low_stress = np.array([10, 0, 20, 5, 0, 10, 10, 0, 60, 0.8, 0.1])
        emotion_low = predictor.predict_emotion(low_stress)
        print(f"[PASS] Low-stress features: emotion={emotion_low}")
        
        # Very high stress
        high_stress = np.array([100, 50, 150, 200, 3, 1, 1, 0.5, 150, 0.1, 2])
        emotion_high = predictor.predict_emotion(high_stress)
        print(f"[PASS] High-stress features: emotion={emotion_high}")
        
        # Test RL state with all emotions
        agent = QLearningAgent()
        for emotion in [0, 1, 2, 3]:
            state = (1, 0, emotion)
            action = agent.select_action(state)
            assert 0 <= action < 4
        
        print(f"[PASS] RL agent handles all emotions")
        return True
    except Exception as e:
        print(f"[FAIL] Edge case test failed: {e}")
        return False


def test_output_formats():
    """Test 8: Output format validation"""
    print("\n" + "="*70)
    print("TEST 8: Output Format Validation")
    print("="*70)
    
    try:
        predictor = MLEmotionPredictor()
        
        # Test that emotion is integer
        emotion = predict_emotion(SAMPLE_FEATURES, predictor)
        assert isinstance(emotion, (int, np.integer)), f"Emotion is not integer: {type(emotion)}"
        print(f"[PASS] Emotion output is integer: {emotion}")
        
        # Test that RL state is tuple
        state = create_rl_state(1, 0, emotion)
        assert isinstance(state, tuple), f"State is not tuple: {type(state)}"
        assert len(state) == 3, f"State has wrong length: {len(state)}"
        print(f"[PASS] RL state format correct: {state}")
        
        # Test that actions are integers
        agent = QLearningAgent()
        action = agent.select_action(state)
        assert isinstance(action, (int, np.integer)), f"Action is not integer: {type(action)}"
        print(f"[PASS] Action output is integer: {action}")
        
        return True
    except Exception as e:
        print(f"[FAIL] Output format test failed: {e}")
        return False


# ============================================================================
# MAIN TEST RUNNER
# ============================================================================

def run_all_tests():
    """Run all validation tests"""
    print("\n")
    print("=" * 70)
    print(" " * 15 + "ML-RL INTEGRATION VALIDATION TESTS")
    print("=" * 70)
    
    tests = [
        test_ml_model_loading,
        test_feature_validation,
        test_emotion_prediction,
        test_rl_state_creation,
        test_rl_agent_with_ml_emotion,
        test_batch_prediction,
        test_edge_cases,
        test_output_formats,
    ]
    
    results = []
    for test_func in tests:
        try:
            results.append(test_func())
        except Exception as e:
            print(f"[ERROR] Unexpected error in {test_func.__name__}: {e}")
            results.append(False)
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    passed = sum(results)
    total = len(results)
    
    print(f"\nRESULT: {passed}/{total} tests passed\n")
    
    if passed == total:
        print("[OK] All tests passed! Integration is ready for use.")
        return True
    else:
        print(f"[WARNING] {total - passed} test(s) failed. Review output above.")
        return False


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    
    success = run_all_tests()
    exit(0 if success else 1)
