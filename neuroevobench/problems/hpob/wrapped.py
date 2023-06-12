from evosax import Strategies
from .evaluator import HPOBEvaluator


def hpob_run(config, log):
    """Running an ES loop on HPO task."""
    # 1. Setup task evaluator with strategy
    evaluator = HPOBEvaluator(
        popsize=config.popsize,
        es_strategy=Strategies[config.strategy_name],
        es_config=config.es_config,
        es_params=config.es_params,
        seed_id=config.seed_id,
    )

    # 2. Run the ES loop with logging
    evaluator.run(config.num_generations)

    # 3. Return mean params and final performance
    return evaluator.fitness_eval, evaluator.solution_eval
