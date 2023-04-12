import os
import yaml
import pkgutil


class HyperParams(object):
    def __init__(
        self,
        strategy_name: str = "OpenES",
        problem_type: str = "brax",
        env_name: str = "ant",
    ):
        self.base_path = "tuned_params/"
        self.strategy_name = strategy_name
        self.problem_type = problem_type
        self.env_name = env_name
        self.params = self._get_hyper_params()

    def _get_hyper_params(self) -> dict:
        """Load strategy-specific tuned hyperparameters from YAML file."""
        self.es_path = os.path.join(
            self.base_path,
            self.problem_type,
            self.env_name,
            self.strategy_name + ".yaml",
        )
        data = pkgutil.get_data(__name__, self.es_path)
        hyperspace = yaml.safe_load(data)
        return hyperspace


class HyperSpace(object):
    def __init__(self, strategy_name: str = "OpenES"):
        self.base_path = "search_spaces/"
        self.strategy_name = strategy_name
        self.space = self._get_hyper_space()

    def _get_hyper_space(self) -> dict:
        """Load strategy-specific hyperparameter space from YAML file."""
        es_path = os.path.join(
            self.base_path,
            self.strategy_name + ".yaml",
        )
        data = pkgutil.get_data(__name__, es_path)
        hyperspace = yaml.safe_load(data)
        return hyperspace
