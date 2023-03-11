import os
import numpy as np
from .hpob_wrapper import Evosax2HPO_Wrapper
from .hpob_task import HPOBTask


class HPOBEvaluator(object):
    def __init__(
        self,
        popsize: int,
        es_strategy,
        es_config={},
        es_params={},
        seed_id: int = 0,
    ):
        self.popsize = popsize
        self.es_strategy = es_strategy
        self.es_config = es_config
        self.es_params = es_params
        self.seed_id = seed_id
        self.setup()

    def setup(self):
        """Initialize task, strategy & policy"""
        self.strategy = Evosax2HPO_Wrapper(
            self.es_strategy,
            popsize=self.popsize,
            es_config=self.es_config,
            es_params=self.es_params,
            seed=self.seed_id,
        )

    def run(self, num_generations, log=None):
        """Run evolution loop with logging."""
        print(f"START EVOLVING HPOB PARAMETERS.")
        # Run ES Loop on HPO tasks -> Return mean timeseries across all tasks
        mean_perf, mean_steps = eval_cont_hbo_sweep(
            self.strategy, num_generations
        )

        if log is not None:
            # Loop over all mean results and return performance
            for g in range(mean_perf.shape[0]):
                log.update(
                    {"num_gens": g + 1, "num_evals": mean_steps[g]},
                    {"test_perf": mean_perf[g]},
                    save=True,
                )
        else:
            print(mean_perf)


def eval_cont_hbo_sweep(
    method: Evosax2HPO_Wrapper,
    num_generations: int,
):
    """Runs BBO evaluation on all HPO-B tasks."""
    # Instantiate data/task handler
    data_dir = os.path.join(os.path.dirname(__file__), "hpob-data/")
    surrogates_dir = os.path.join(
        os.path.dirname(__file__), "saved-surrogates/"
    )
    hpob_hdlr = HPOBTask(
        root_dir=data_dir,
        mode="v3-test",
        surrogates_dir=surrogates_dir,
    )

    # Get all 'spaces' (model types to opt hyperparams for)
    all_spaces = hpob_hdlr.get_search_spaces()
    all_data, all_steps = [], []
    # Loop over all spaces (models -> hyperparameters)
    for k in all_spaces:
        data = hpob_hdlr.get_datasets(k)
        # Loop over all datasets (model + datasets -> hyperparameters)
        for d in data:
            num_dims = hpob_hdlr.get_search_space_dim(k)
            method.init_bbo(num_dims)
            acc, steps = hpob_hdlr.evaluate_continuous(
                method,
                search_space_id=k,
                dataset_id=d,
                seed="test0",
                n_trials=num_generations,
            )
            all_data.append(acc)
            all_steps.append(steps)

    stacked_perf = np.stack(all_data)
    stacked_steps = np.stack(all_steps)
    # Mean performance across HPO tasks
    mean_perf = stacked_perf.mean(axis=0)
    mean_steps = stacked_steps.mean(axis=0)
    return mean_perf, mean_steps
