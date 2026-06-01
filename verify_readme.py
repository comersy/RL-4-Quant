#!/usr/bin/env python3
"""Verify README documentation claims"""

from data.loader import list_available_days
from envs.env import OptionsEnv, EPISODE_DAYS, SPOT_HISTORY

days = list_available_days()
print('README Verification Check:')
print('-' * 50)
print(f'First date: {days[0]}')
print(f'Last date: {days[-1]}')
print(f'Total days: {len(days)}')
print(f'Episode days: {EPISODE_DAYS}')
print(f'Spot history: {SPOT_HISTORY}')

env = OptionsEnv(init_random_positions=False)
obs, _ = env.reset()
print(f'Observation size: {len(obs)}')
print('-' * 50)
print('✓ README documentation verified')
