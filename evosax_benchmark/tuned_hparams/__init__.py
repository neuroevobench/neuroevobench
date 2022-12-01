from .open_es import OpenES_configs
from .pgpe import PGPE_configs
from .sep_cma_es import Sep_CMA_ES_configs
from .snes import SNES_configs


all_configs = {
    "OpenES": OpenES_configs,
    "PGPE": PGPE_configs,
    "Sep_CMA_ES": Sep_CMA_ES_configs,
    "SNES": SNES_configs,
}


def get_tuned_hparams(strategy_name: str, env_name: str) -> dict:
    """Return tuned hparam config of interest."""
    return all_configs[strategy_name](env_name).es_config


__all__ = [
    "OpenES_configs",
    "PGPE_configs",
    "Sep_CMA_ES_configs",
    "SNES_configs",
    "get_tuned_hparams",
]
