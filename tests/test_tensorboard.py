from RL_model import train as train_module


class DummyWriter:
    def __init__(self):
        self.scalars = []
        self.flushed = False
        self.closed = False

    def add_scalar(self, tag, value, step):
        self.scalars.append((tag, value, step))

    def flush(self):
        self.flushed = True

    def close(self):
        self.closed = True


def test_tensorboard_log_dir_is_configured():
    assert train_module.CONFIG["tensorboard_log_dir"] == "runs/gru_ppo"


def test_training_loop_writes_tensorboard_episode_metrics(monkeypatch):
    class DummyEnv:
        observation_space = type("ObservationSpace", (), {"shape": (2,)})()

        def __init__(self):
            self.realized_pnl = 10.0
            self.unrealized_pnl = -2.5
            self.portfolio = [
                {"is_short": False},
                {"is_short": True},
            ]

        def reset(self):
            return [0.0, 0.0], {}

        def step(self, action):
            self.realized_pnl = 12.0
            self.unrealized_pnl = 3.0
            self.portfolio.append({"is_short": True})
            return [1.0, 1.0], 1.5, True, False, {}

    class DummyAgent:
        def get_action(self, obs, h_gru=None):
            return {"action_type": 1, "log_prob": 0.0}, None, 0.0

        def train_step(self, episodes_batch, config):
            return {"policy_loss": 0.25}

    config = train_module.CONFIG.copy()
    config.update(
        {
            "episode_length": 1,
            "buffer_capacity": 2,
            "batch_size": 1,
            "tensorboard_log_dir": None,
        }
    )
    writer = DummyWriter()

    train_module.train(DummyEnv(), DummyAgent(), config, num_episodes=1, writer=writer)

    tags = {tag for tag, _, _ in writer.scalars}
    assert "episode/reward" in tags
    assert "episode/steps" in tags
    assert "episode/hold_actions" in tags
    assert "episode/trade_actions" in tags
    assert "episode/close_actions" in tags
    assert "buffer/size" in tags
    assert "pnl/realized" in tags
    assert "pnl/unrealized" in tags
    assert "pnl/total" in tags
    assert "portfolio/open_positions" in tags
    assert "portfolio/long_positions" in tags
    assert "portfolio/short_positions" in tags
    assert "loss/policy_loss" in tags
    assert writer.flushed is True
    assert writer.closed is False

    latest = {tag: value for tag, value, _ in writer.scalars}
    assert latest["pnl/realized"] == 12.0
    assert latest["pnl/unrealized"] == 3.0
    assert latest["pnl/total"] == 15.0
    assert latest["portfolio/open_positions"] == 3
    assert latest["portfolio/long_positions"] == 1
    assert latest["portfolio/short_positions"] == 2
    assert latest["episode/trade_actions"] == 1
