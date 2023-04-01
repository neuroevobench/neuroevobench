import os
import yaml
import pkgutil


eo_space_paths = {
    "brax": {
        "OpenES": "search_spaces/brax/OpenES.yaml",
        "PGPE": "search_spaces/brax/PGPE.yaml",
        "SNES": "search_spaces/brax/SNES.yaml",
        "Sep_CMA_ES": "search_spaces/brax/Sep_CMA_ES.yaml",
    }
}


class HyperSpace(object):
    def __init__(self, strategy_name: str = "OpenES", env_name: str = "brax"):
        self.strategy_name = strategy_name
        self.env_name = env_name
        self.space_dict = self._get_hyper_space()

    def _get_hyper_space(self) -> dict:
        """Load strategy-specific hyperparameter space from YAML file."""
        es_path = eo_space_paths[self.env_name][self.strategy_name]
        data = pkgutil.get_data(__name__, es_path)
        hyperspace = yaml.safe_load(data)
        return hyperspace
