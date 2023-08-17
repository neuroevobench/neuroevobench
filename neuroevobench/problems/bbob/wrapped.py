from typing import Optional
from evosax import Strategy
from .evaluator import BBOBEvaluator
from ...utils import collect_strategies

Strategies = collect_strategies()


def bbob_run(
    config,
    log,
    search_iter: Optional[int] = None,
    strategy_class: Optional[Strategy] = None,
):
    """Running an ES loop on BBOB task."""
    # 1. Setup task evaluator with strategy
    if strategy_class is not None:
        base_strategy = strategy_class
    else:
        base_strategy = Strategies[config.strategy_name]

    evaluator = BBOBEvaluator(
        popsize=config.popsize,
        num_dims=config.num_dims,
        es_strategy=base_strategy,
        es_config=config.es_config,
        es_params=config.es_params,
        num_eval_runs=config.num_eval_runs,
        log=log,
        seed_id=config.seed_id,
        iter_id=search_iter,
    )

    # 2. Run the ES loop with logging
    evaluator.run(config.num_generations)

    # 3. Return mean params and final performance
    return evaluator.fitness_eval, evaluator.solution_eval
