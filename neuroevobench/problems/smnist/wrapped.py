from typing import Optional
from evosax import Strategies
from .policy import SMNISTPolicy
from .task import SMNISTTask
from .evaluator import SMNISTEvaluator


def smnist_run(config, log, search_iter: Optional[int] = None):
    # 1. Create placeholder env to get number of actions for policy init
    policy = SMNISTPolicy(hidden_dims=config.model_config.num_hidden_units)

    # 2. Define train/test task based on configs/eval settings
    train_task = SMNISTTask(
        config.task_config.batch_size,
        permute_seq=config.task_config.permute_seq,
        seed_id=config.seed_id,  # Fix seed for permutation
        test=False,
    )
    test_task = SMNISTTask(
        10000,
        permute_seq=config.task_config.permute_seq,
        seed_id=config.seed_id,  # Fix seed for permutation
        test=True,
    )

    # 3. Setup task evaluator with strategy and policy
    evaluator = SMNISTEvaluator(
        policy=policy,
        train_task=train_task,
        test_task=test_task,
        popsize=config.popsize,
        es_strategy=Strategies[config.strategy_name],
        es_config=config.es_config,
        es_params=config.es_params,
        seed_id=config.seed_id,
        log=log,
        iter_id=search_iter,
    )

    # 4. Run the ES loop with logging
    evaluator.run(config.num_generations, config.eval_every_gen)

    # 5. Return mean params and final performance
    return evaluator.fitness_eval, evaluator.solution_eval
