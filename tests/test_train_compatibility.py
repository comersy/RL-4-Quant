import numpy as np

from envs.env import OptionsEnv
from RL_model.train import CONFIG, GRUPPOAgent, build_env_and_agent


EXPECTED_ACTION_KEYS = {
    "action_type",
    "call_or_put",
    "strike",
    "maturity",
    "quantity_signed",
    "log_prob",
}


def tiny_config():
    config = CONFIG.copy()
    config.update(
        {
            "device": "cpu",
            "encoder_hidden_sizes": [32, 16],
            "gru_hidden_size": 16,
            "actor_hidden_sizes": [16],
            "critic_hidden_sizes": [16],
            "learning_rate": 3e-4,
            "tensorboard_log_dir": None,
        }
    )
    return config


def test_agent_action_matches_environment_contract():
    env = OptionsEnv(init_random_positions=False)
    agent = GRUPPOAgent(obs_dim=env.observation_space.shape[0], config=tiny_config())

    obs, _ = env.reset(seed=123)
    action, h_gru, value = agent.get_action(obs)

    assert set(action) == EXPECTED_ACTION_KEYS
    assert h_gru is not None
    assert isinstance(value, float)


def test_agent_environment_integration_for_a_few_steps():
    env = OptionsEnv(init_random_positions=False)
    agent = GRUPPOAgent(obs_dim=env.observation_space.shape[0], config=tiny_config())

    obs, _ = env.reset(seed=123)
    h_gru = None

    for _ in range(3):
        action, h_gru, value = agent.get_action(obs, h_gru)
        obs, reward, terminated, truncated, info = env.step(action)

        assert obs.shape == env.observation_space.shape
        assert isinstance(reward, float)
        assert isinstance(value, float)
        assert info == {}
        if terminated or truncated:
            break


def test_agent_accepts_flat_numpy_observation():
    env = OptionsEnv(init_random_positions=False)
    agent = GRUPPOAgent(obs_dim=env.observation_space.shape[0], config=tiny_config())

    obs = np.random.randn(agent.obs_dim).astype(np.float32)
    action, _, _ = agent.get_action(obs)

    assert set(action) == EXPECTED_ACTION_KEYS


def test_default_training_components_are_wired_to_options_env():
    env, agent, config = build_env_and_agent(config=tiny_config(), init_random_positions=False)

    assert isinstance(env, OptionsEnv)
    assert agent.obs_dim == env.observation_space.shape[0]
    assert config["episode_length"] <= 150
