#!/usr/bin/env python3
"""
Test compatibility between train.py and env.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from envs.env import OptionsEnv
from RL_model.train import GRUPPOAgent

def test_environment():
    """Test basic environment functionality"""
    print("\n" + "="*60)
    print("TESTING ENVIRONMENT")
    print("="*60)
    
    try:
        env = OptionsEnv()
        print("✓ OptionsEnv instantiated successfully")
        
        # Test reset
        obs, info = env.reset()
        print(f"✓ reset() works: obs shape = {obs.shape}")
        
        # Test action_space
        print(f"✓ action_space keys: {list(env.action_space.spaces.keys())}")
        
        # Create a sample action (from train.py output format)
        action = {
            "action_type": np.array(1),  # Trade
            "call_or_put": np.array([0.5]),
            "strike": np.array([45000.0]),
            "maturity": np.array([30.0]),
            "quantity_signed": np.array([1.0]),
            "log_prob": np.array([0.0]),
        }
        
        # Test step
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"✓ step() works: obs shape = {obs.shape}, reward = {reward:.4f}")
        print(f"✓ portfolio size: {len(env.portfolio)}")
        
        # Test multiple steps
        for i in range(5):
            action = {
                "action_type": np.array(0),  # Do nothing
                "call_or_put": np.array([0.0]),
                "strike": np.array([50000.0]),
                "maturity": np.array([20.0]),
                "quantity_signed": np.array([0.0]),
                "log_prob": np.array([0.0]),
            }
            obs, _, _, _, _ = env.step(action)
        
        print(f"✓ Multiple steps work: current episode_day = {env.episode_day}")
        print("\n✓ ENVIRONMENT TESTS PASSED\n")
        return True
        
    except Exception as e:
        import traceback
        print(f"✗ ENVIRONMENT TEST FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()
        return False

def test_agent():
    """Test agent instantiation"""
    print("\n" + "="*60)
    print("TESTING AGENT")
    print("="*60)
    
    try:
        config = {
            "device": "cpu",
            "encoder_hidden_sizes": [128, 64],
            "gru_hidden_size": 64,
            "actor_hidden_sizes": [64, 32],
            "critic_hidden_sizes": [64, 32],
            "learning_rate": 3e-4,
        }
        
        agent = GRUPPOAgent(obs_dim=2 + 365 + 500*6 + 10000*8 + 2, config=config)
        print("✓ GRUPPOAgent instantiated successfully")
        
        # Test get_action
        dummy_obs = np.random.randn(agent.obs_dim).astype(np.float32)
        action_dict, h_gru, value = agent.get_action(dummy_obs)
        
        print(f"✓ get_action() works")
        print(f"  - Returns dict with keys: {list(action_dict.keys())}")
        print(f"  - Value estimate: {value:.4f}")
        print(f"  - GRU hidden state shape: {h_gru.shape if hasattr(h_gru, 'shape') else 'N/A'}")
        
        # Check action dict format matches environment expectations
        expected_keys = {"action_type", "call_or_put", "strike", "maturity", "quantity_signed", "log_prob"}
        actual_keys = set(action_dict.keys())
        if expected_keys == actual_keys:
            print(f"✓ Agent output keys match environment expectations")
        else:
            print(f"✗ Key mismatch! Expected {expected_keys}, got {actual_keys}")
            return False
        
        print("\n✓ AGENT TESTS PASSED\n")
        return True
        
    except Exception as e:
        import traceback
        print(f"✗ AGENT TEST FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()
        return False

def test_integration():
    """Test agent-environment integration"""
    print("\n" + "="*60)
    print("TESTING INTEGRATION")
    print("="*60)
    
    try:
        env = OptionsEnv()
        
        config = {
            "device": "cpu",
            "encoder_hidden_sizes": [128, 64],
            "gru_hidden_size": 64,
            "actor_hidden_sizes": [64, 32],
            "critic_hidden_sizes": [64, 32],
            "learning_rate": 3e-4,
        }
        
        agent = GRUPPOAgent(obs_dim=env.observation_space.shape[0], config=config)
        
        print(f"✓ Environment obs_dim: {env.observation_space.shape[0]}")
        print(f"✓ Agent obs_dim: {agent.obs_dim}")
        
        # Reset environment
        obs, _ = env.reset()
        h_gru = None
        
        # Run a few steps
        for step in range(5):
            action, h_gru, value = agent.get_action(obs, h_gru)
            obs, reward, terminated, truncated, info = env.step(action)
            
            if step == 0:
                print(f"✓ Agent-environment integration works!")
                print(f"  - Observation shape: {obs.shape}")
                print(f"  - Action type returned: {type(action['action_type'])}")
        
        print("\n✓ INTEGRATION TESTS PASSED\n")
        return True
        
    except Exception as e:
        import traceback
        print(f"✗ INTEGRATION TEST FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("\n" + "="*60)
    print("TRAIN.PY & ENV.PY COMPATIBILITY CHECK")
    print("="*60)
    
    results = []
    results.append(("Environment Tests", test_environment()))
    results.append(("Agent Tests", test_agent()))
    results.append(("Integration Tests", test_integration()))
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status:8} {name}")
    
    all_passed = all(passed for _, passed in results)
    if all_passed:
        print("\n✓ ALL TESTS PASSED - train.py and env.py are compatible!")
        sys.exit(0)
    else:
        print("\n✗ SOME TESTS FAILED - see details above")
        sys.exit(1)
