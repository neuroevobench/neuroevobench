from typing import Optional
from evosax import Strategy
from .policy import AdditionPolicy
from .task import AdditionTask
from .evaluator import AdditionEvaluator
from ...utils import collect_strategies

Strategies = collect_strategies()


def addition_run(
    config,
    log,
    search_iter: Optional[int] = None,
    strategy_class: Optional[Strategy] = None,
):
    """Running an ES loop on Addition task."""
    # 1. Create placeholder env to get number of actions for policy init
    policy = AdditionPolicy(hidden_dims=config.model_config.num_hidden_units)

    # 2. Define train/test task based on configs/eval settings
    train_task = AdditionTask(
        config.task_config.batch_size,
        seq_length=config.task_config.seq_length,
        seed_id=config.seed_id,  # Fix seed for data generation
        test=False,
    )
    test_task = AdditionTask(
        10000,
        seq_length=config.task_config.seq_length,
        seed_id=config.seed_id,  # Fix seed for data generation
        test=True,
    )

    # 3. Setup task evaluator with strategy and policy
    if strategy_class is not None:
        base_strategy = strategy_class
    else:
        base_strategy = Strategies[config.strategy_name]

    evaluator = AdditionEvaluator(
        policy=policy,
        train_task=train_task,
        test_task=test_task,
        popsize=config.popsize,
        es_strategy=base_strategy,
        es_config=config.es_config,
        es_params=config.es_params,
        seed_id=config.seed_id,
        log=log,
        iter_id=search_iter,
    )

    # 4. Run the ES loop with logging
    evaluator.run(config.num_generations, config.eval_every_gen)

    # 5. Return mean params and final performance
    # NOTE:
    return evaluator.fitness_eval, evaluator.solution_eval
