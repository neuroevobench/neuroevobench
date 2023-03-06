"""Tuned hparams for Brax 4L-32U tanh MLP w. popsize 256"""


class OpenES_configs(object):
    def __init__(self, env_name: str):
        self.env_name = env_name

    @property
    def es_config(self):
        if self.env_name == "ant":
            return {
                "centered_rank": True,
                "maximize": True,
                "lrate_init": 0.01,
                "lrate_decay": 0.999,
                "lrate_limit": 0.001,
                "sigma_init": 0.05,
                "sigma_decay": 0.999,
                "sigma_limit": 0.01,
            }
        elif self.env_name == "fetch":
            return {
                "centered_rank": True,
                "maximize": True,
                "lrate_init": 0.02,
                "lrate_decay": 0.999,
                "lrate_limit": 0.001,
                "sigma_init": 0.125,
                "sigma_decay": 0.999,
                "sigma_limit": 0.01,
            }
        elif self.env_name == "grasp":
            return {
                "centered_rank": True,
                "maximize": True,
                "lrate_init": 0.01,
                "lrate_decay": 0.999,
                "lrate_limit": 0.001,
                "sigma_init": 0.05,
                "sigma_decay": 0.999,
                "sigma_limit": 0.01,
            }
        elif self.env_name == "halfcheetah":
            return {
                "centered_rank": True,
                "maximize": True,
                "lrate_init": 0.01,
                "lrate_decay": 0.999,
                "lrate_limit": 0.001,
                "sigma_init": 0.075,
                "sigma_decay": 0.999,
                "sigma_limit": 0.01,
            }
        elif self.env_name == "hopper":
            return {
                "centered_rank": True,
                "maximize": True,
                "lrate_init": 0.04,
                "lrate_decay": 0.999,
                "lrate_limit": 0.001,
                "sigma_init": 0.125,
                "sigma_decay": 0.999,
                "sigma_limit": 0.01,
            }
        elif self.env_name == "humanoid":
            return {
                "centered_rank": True,
                "maximize": True,
                "lrate_init": 0.02,
                "lrate_decay": 0.999,
                "lrate_limit": 0.001,
                "sigma_init": 0.1,
                "sigma_decay": 0.999,
                "sigma_limit": 0.01,
            }
        elif self.env_name == "reacher":
            return {
                "centered_rank": True,
                "maximize": True,
                "lrate_init": 0.01,
                "lrate_decay": 0.999,
                "lrate_limit": 0.001,
                "sigma_init": 0.025,
                "sigma_decay": 0.999,
                "sigma_limit": 0.01,
            }
        elif self.env_name == "ur5e":
            return {
                "centered_rank": True,
                "maximize": True,
                "lrate_init": 0.01,
                "lrate_decay": 0.999,
                "lrate_limit": 0.001,
                "sigma_init": 0.125,
                "sigma_decay": 0.999,
                "sigma_limit": 0.01,
            }
        elif self.env_name == "walker2d":
            return {
                "centered_rank": True,
                "maximize": True,
                "lrate_init": 0.02,
                "lrate_decay": 0.999,
                "lrate_limit": 0.001,
                "sigma_init": 0.025,
                "sigma_decay": 0.999,
                "sigma_limit": 0.01,
            }
        else:
            return {
                "centered_rank": True,
                "maximize": True,
            }
