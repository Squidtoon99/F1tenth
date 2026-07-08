"""Training environment.

Scaffold placeholder. Wraps the synthetic-data simulator into a vectorized RL
environment. The observation layout is imported from the shared contract so the
policy trained here matches the on-car inference node.

Migration note: port from F1tenth-Genesis `f1tenth_env/{env,car,observations,
rewards,terminations,opponents}.py`.
"""

from f1tenth_contract import ActionSpec, ObservationSpec


class F1tenthEnv:
    """Placeholder environment. TODO: implement reset/step over the simulator."""

    def __init__(self, num_envs: int = 1) -> None:
        self.num_envs = num_envs
        self.observation_spec = ObservationSpec()
        self.action_spec = ActionSpec()

    def reset(self):
        raise NotImplementedError

    def step(self, actions):
        raise NotImplementedError
