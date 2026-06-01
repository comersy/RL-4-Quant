#!/usr/bin/env python3
"""Test random position initialization"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from envs.env import OptionsEnv

def test_random_positions():
    env = OptionsEnv()
    
    print("Testing 3 resets with random initial positions:\n")
    
    for episode in range(3):
        obs, info = env.reset()
        print(f"Episode {episode}:")
        print(f"  - Starting positions: {len(env.portfolio)}")
        
        if env.portfolio:
            for i, pos in enumerate(env.portfolio):
                short_str = "SHORT" if pos["is_short"] else "LONG "
                print(f"    {short_str} {pos['option_type'].upper():4} strike={pos['strike']:7.0f} qty={pos['quantity']:.1f} mat={pos['maturity']}d")
        
        # Verify entry price = current price (unrealized P&L should be ~0)
        if env.portfolio:
            unrealized_manual = 0.0
            for p in env.portfolio:
                pnl_per_pos = (p["price"] - p["entry_price"]) * p["quantity"]
                if p["is_short"]:
                    pnl_per_pos = -pnl_per_pos
                unrealized_manual += pnl_per_pos
            print(f"  - Initial unrealized P&L: {env.unrealized_pnl:.6f} (should be ~0)")
        
        # Run a few steps
        for step in range(3):
            action = {
                "action_type": np.array(0),  # do nothing
                "call_or_put": np.array([0.0]),
                "strike": np.array([50000.0]),
                "maturity": np.array([30.0]),
                "quantity_signed": np.array([0.0]),
                "log_prob": np.array([0.0]),
            }
            obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"  - After 3 steps: portfolio size = {len(env.portfolio)}, realized_pnl = {env.realized_pnl:.6f}")
        print()

if __name__ == "__main__":
    test_random_positions()
    print("✓ Test passed!")
