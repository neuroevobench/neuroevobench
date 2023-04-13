from typing import Optional
from evosax import Strategies
from .policy import SVHNPolicy
from .task import SVHNTask
from .evaluator import SVHNEvaluator


def svhn_run(config, log, search_iter: Optional[int] = None):
    """Running an ES loop on SVHN task."""
    # 1. Create placeholder env to get number of actions for policy init
    policy = SVHNPolicy(config.model_config.resnet_no)

    # 2. Define train/test task based on configs/eval settings
    train_task = SVHNTask(config.task_config.batch_size, test=False)
    test_task = SVHNTask(12000, test=True)

    # 3. Setup task evaluator with strategy and policy
    evaluator = SVHNEvaluator(
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
