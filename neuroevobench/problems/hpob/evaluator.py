from typing import Optional, Any
import os
import numpy as np
from .eo_wrapper import Evosax2HPO_Wrapper
from .task import HPOBTask


class HPOBEvaluator(object):
    def __init__(
        self,
        popsize: int,
        es_strategy,
        es_config={},
        es_params={},
        seed_id: int = 0,
        log: Optional[Any] = None,
        time_tick_str: str = "num_gens",
        iter_id: Optional[int] = None,
        maximize_objective: bool = True,
    ):
        self.popsize = popsize
        self.es_strategy = es_strategy
        self.es_config = es_config
        self.es_params = es_params
        self.seed_id = seed_id
        self.log = log
        self.time_tick_str = time_tick_str
        self.iter_id = iter_id
        self.maximize_objective = maximize_objective
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

    def run(self, num_generations: int):
        """Run evolution loop with logging."""
        print("hpob: START EVOLVING HPOB PARAMETERS.")
        # Run ES Loop on HPO tasks -> Return mean timeseries across all tasks
        mean_perf, mean_steps = eval_cont_hbo_sweep(
            self.strategy, num_generations
        )

        # Loop over all mean results and return performance
        for g in range(mean_perf.shape[0]):
            time_tic = {self.time_tick_str: g}
            if self.iter_id is not None:
                time_tic["iter_id"] = self.iter_id
            stats_tic = {"test_eval_perf": mean_perf[g]}
            self.update_log(time_tic, stats_tic)

        self.fitness_eval = float(mean_perf[-1])
        self.solution_eval = None

    def update_log(self, time_tic, stats_tic, model: Optional[Any] = None):
        """Update logger with newest data."""
        if self.log is not None:
            self.log.update(time_tic, stats_tic, model=model, save=True)
        else:
            print(time_tic, stats_tic)


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
