from typing import Optional
import os
import yaml
import pkgutil


class HyperParams(object):
    def __init__(
        self, strategy_name: str = "OpenES", params: Optional[dict] = None
    ):
        self.base_path = "default/"
        self.strategy_name = strategy_name
        if params is not None:
            self.params = params
        else:
            self.params = self._get_hyper_params()

    def _get_hyper_params(self) -> dict:
        """Load strategy-specific default hyperparameters from YAML file."""
        es_path = os.path.join(
            self.base_path,
            self.strategy_name + ".yaml",
        )
        data = pkgutil.get_data(__name__, es_path)
        hyperparams = yaml.safe_load(data)
        return hyperparams


class HyperSpace(object):
    def __init__(
        self, strategy_name: str = "OpenES", space: Optional[dict] = None
    ):
        self.base_path = "search_spaces/"
        self.strategy_name = strategy_name
        if space is not None:
            self.space = space
        else:
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
