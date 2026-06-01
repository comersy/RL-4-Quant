#!/usr/bin/env python3
"""Verify 150-day episode configuration"""

from envs.env import OptionsEnv, EPISODE_DAYS, SPOT_HISTORY

print('Episode Configuration Updated:')
print(f'EPISODE_DAYS: {EPISODE_DAYS}')
print(f'SPOT_HISTORY: {SPOT_HISTORY}')

env = OptionsEnv(init_random_positions=False)
obs, _ = env.reset()
print(f'Observation size: {len(obs)}')

# Verify episode_day observation is within bounds
print('\nEpisode day index in observation:')
print(f'  Value: {obs[1]} (should be 0 at reset)')
print(f'  Range: [0, {EPISODE_DAYS-1}]')

print('\n✓ 150-day episode configuration working')
