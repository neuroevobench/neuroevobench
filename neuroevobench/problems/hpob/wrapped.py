from typing import Optional
from evosax import Strategy
from .evaluator import HPOBEvaluator
from ...utils import collect_strategies

Strategies = collect_strategies()


def hpob_run(
    config,
    log,
    search_iter: Optional[int] = None,
    strategy_class: Optional[Strategy] = None,
):
    """Running an ES loop on HPO task."""
    # 1. Setup task evaluator with strategy
    if strategy_class is not None:
        base_strategy = strategy_class
    else:
        base_strategy = Strategies[config.strategy_name]

    evaluator = HPOBEvaluator(
        popsize=config.popsize,
        es_strategy=base_strategy,
        es_config=config.es_config,
        es_params=config.es_params,
        seed_id=config.seed_id,
        log=log,
        iter_id=search_iter,
    )

    # 2. Run the ES loop with logging
    evaluator.run(config.num_generations)

    # 3. Return mean params and final performance
    return evaluator.fitness_eval, evaluator.solution_eval
