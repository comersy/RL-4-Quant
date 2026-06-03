import numpy as np

from envs.env import EPISODE_DAYS, SPOT_HISTORY, OptionsEnv


def hold_action():
    return {
        "action_type": np.array(0),
        "call_or_put": np.array([0.0]),
        "strike": np.array([0.0]),
        "maturity": np.array([30.0]),
        "quantity_signed": np.array([0.0]),
        "log_prob": np.array([0.0]),
    }


def test_episode_configuration_and_reset_observation():
    env = OptionsEnv(init_random_positions=False)
    obs, info = env.reset(seed=123)

    assert EPISODE_DAYS == 150
    assert SPOT_HISTORY == 365
    assert info == {}
    assert obs.shape == env.observation_space.shape
    assert obs.dtype == np.float32
    assert obs[1] == 0


def test_environment_accepts_train_action_format():
    env = OptionsEnv(init_random_positions=False)
    obs, _ = env.reset(seed=123)

    action = {
        "action_type": np.array(1),
        "call_or_put": np.array([0.5]),
        "strike": np.array([45000.0]),
        "maturity": np.array([30.0]),
        "quantity_signed": np.array([1.0]),
        "log_prob": np.array([0.0]),
    }

    next_obs, reward, terminated, truncated, info = env.step(action)

    assert next_obs.shape == obs.shape
    assert isinstance(reward, float)
    assert terminated is False
    assert truncated is False
    assert info == {}


def test_environment_advances_multiple_hold_steps():
    env = OptionsEnv(init_random_positions=False)
    env.reset(seed=123)

    for _ in range(5):
        obs, reward, terminated, truncated, info = env.step(hold_action())
        assert obs.shape == env.observation_space.shape
        assert truncated is False
        if terminated:
            break

    assert env.episode_day == 5


def test_random_initial_positions_start_marked_flat_when_present():
    env = OptionsEnv(init_random_positions=True)
    env.reset(seed=1)

    for position in env.portfolio:
        assert position["entry_price"] == position["price"]

    assert abs(env.unrealized_pnl) < 1e-8
