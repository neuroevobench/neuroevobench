from .load import load_config
from .csv_logger import CSV_Logger
from evosax import Strategies
from neuroevobench.blines import BayesOptNevergrad, BayesOptJAX


def collect_strategies():
    """Manually add all strategies added as baselines to evosax."""
    Strategies["BayesOptNevergrad"] = BayesOptNevergrad
    Strategies["BayesOptJAX"] = BayesOptJAX
    return Strategies


__all__ = ["load_config", "CSV_Logger", "collect_strategies"]
