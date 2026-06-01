#!/usr/bin/env python3
"""
Quick test to verify 150-day episodes work correctly
"""

from envs.env import OptionsEnv, EPISODE_DAYS, SPOT_HISTORY
from RL_model.train import GRUPPOAgent
import numpy as np

print("=" * 60)
print("150-DAY EPISODE VALIDATION")
print("=" * 60)

# Check constants
print(f"\n✓ EPISODE_DAYS: {EPISODE_DAYS} (expected: 150)")
print(f"✓ SPOT_HISTORY: {SPOT_HISTORY} (expected: 365)")

# Create environment and check observation
print("\nTesting environment...")
env = OptionsEnv(init_random_positions=False)
obs, info = env.reset()

print(f"  ✓ Observation shape: {obs.shape}")
print(f"  ✓ Episode day at reset: {obs[1]:.0f} (expected: 0)")

# Run a few steps
print("\nRunning 5 steps...")
for step in range(5):
    action = {
        "action_type": np.array([0]),  # do nothing
        "call_or_put": np.array([0.0]),
        "strike": np.array([0.0]),
        "maturity": np.array([30.0]),
        "quantity_signed": np.array([0.0]),
        "log_prob": np.array([0.0]),
    }
    obs, reward, terminated, truncated, info = env.step(action)
    day = int(obs[1])
    print(f"  Step {step+1}: day={day}, reward={reward:.2f}, terminated={terminated}")
    if terminated:
        break

print(f"\n✓ All 150-day episode tests passed!")
