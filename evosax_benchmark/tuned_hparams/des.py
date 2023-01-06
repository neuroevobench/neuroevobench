"""Default hparams for Brax 4L-32U tanh MLP w. popsize 256"""


class DES_configs(object):
    def __init__(self, env_name: str):
        self.env_name = env_name

    @property
    def es_config(self):
        if self.env_name == "ant":
            return {
                "maximize": True,
                "temperature": 12.5,
                "sigma_init": 0.05,
            }
        elif self.env_name == "fetch":
            return {
                "maximize": True,
                "temperature": 12.5,
                "sigma_init": 0.075,
            }
        elif self.env_name == "grasp":
            return {
                "maximize": True,
                "temperature": 12.5,
                "sigma_init": 0.05,
            }
        elif self.env_name == "halfcheetah":
            return {
                "maximize": True,
                "temperature": 12.5,
                "sigma_init": 0.05,
            }
        elif self.env_name == "hopper":
            return {
                "maximize": True,
                "temperature": 12.5,
                "sigma_init": 0.075,
            }
        elif self.env_name == "humanoid":
            return {
                "maximize": True,
                "temperature": 12.5,
                "sigma_init": 0.075,
            }
        elif self.env_name == "reacher":
            return {
                "maximize": True,
                "temperature": 12.5,
                "sigma_init": 0.05,
            }
        elif self.env_name == "ur5e":
            return {
                "maximize": True,
                "temperature": 12.5,
                "sigma_init": 0.075,
            }
        elif self.env_name == "walker2d":
            return {
                "maximize": True,
                "temperature": 12.5,
                "sigma_init": 0.075,
            }
        else:
            raise ValueError("Provide valid env_name.")
