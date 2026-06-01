#!/usr/bin/env python3
"""Debug integration test"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from envs.env import OptionsEnv
from RL_model.train import GRUPPOAgent

def test_simple():
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
    
    obs, _ = env.reset()
    h_gru = None
    
    for step in range(5):
        print(f"Step {step}...")
        
        try:
            action, h_gru, value = agent.get_action(obs, h_gru)
            print(f"  Action obtained")
            print(f"  action_type: {action['action_type']} (type: {type(action['action_type'])})")
            
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"  Step executed: reward={reward}, terminated={terminated}")
            
        except Exception as e:
            import traceback
            print(f"  ERROR: {type(e).__name__}: {e}")
            traceback.print_exc()
            return False
    
    return True

if __name__ == "__main__":
    if test_simple():
        print("\n✓ Test passed!")
    else:
        print("\n✗ Test failed!")
