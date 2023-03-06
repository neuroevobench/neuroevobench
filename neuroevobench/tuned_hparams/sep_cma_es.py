"""Tuned hparams for Brax 4L-32U tanh MLP w. popsize 256"""


class Sep_CMA_ES_configs(object):
    def __init__(self, env_name: str):
        self.env_name = env_name

    @property
    def es_config(self):
        if self.env_name == "ant":
            return {
                "maximize": True,
                "elite_ratio": 0.4,
                "sigma_init": 0.05,
            }
        elif self.env_name == "fetch":
            return {
                "maximize": True,
                "elite_ratio": 0.2,
                "sigma_init": 0.125,
            }
        elif self.env_name == "grasp":
            return {
                "maximize": True,
                "elite_ratio": 0.4,
                "sigma_init": 0.1,
            }
        elif self.env_name == "halfcheetah":
            return {
                "maximize": True,
                "elite_ratio": 0.5,
                "sigma_init": 0.05,
            }
        elif self.env_name == "hopper":
            return {
                "maximize": True,
                "elite_ratio": 0.1,
                "sigma_init": 0.1,
            }
        elif self.env_name == "humanoid":
            return {
                "maximize": True,
                "elite_ratio": 0.2,
                "sigma_init": 0.1,
            }
        elif self.env_name == "reacher":
            return {
                "maximize": True,
                "elite_ratio": 0.2,
                "sigma_init": 0.05,
            }
        elif self.env_name == "ur5e":
            return {
                "maximize": True,
                "elite_ratio": 0.5,
                "sigma_init": 0.15,
            }
        elif self.env_name == "walker2d":
            return {
                "maximize": True,
                "elite_ratio": 0.2,
                "sigma_init": 0.1,
            }
        else:
            return {
                "sigma_init": 0.1,
                "maximize": True,
            }
