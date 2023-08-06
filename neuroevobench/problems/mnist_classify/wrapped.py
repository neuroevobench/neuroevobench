from typing import Optional
from evosax import Strategies, Strategy
from .policy import MNIST_Classify_Policy
from .task import MNIST_Classify_Task
from .evaluator import MNIST_Classify_Evaluator
from ...blines import BayesOpt

Strategies["BayesOpt"] = BayesOpt


def mnist_classify_run(
    config,
    log,
    search_iter: Optional[int] = None,
    strategy_class: Optional[Strategy] = None,
):
    """Running an ES loop on MNIST classification task."""
    # 1. Create placeholder env to get number of actions for policy init
    policy = MNIST_Classify_Policy(
        hidden_dims=config.model_config.num_hidden_layers
        * [config.model_config.num_hidden_units],
    )

    # 2. Define train/test task based on configs/eval settings
    train_task = MNIST_Classify_Task(
        config.env_name, config.task_config.batch_size, test=False
    )
    test_task = MNIST_Classify_Task(config.env_name, 0, test=True)

    # 3. Setup task evaluator with strategy and policy
    if strategy_class is not None:
        base_strategy = strategy_class
    else:
        base_strategy = Strategies[config.strategy_name]

    evaluator = MNIST_Classify_Evaluator(
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
    return evaluator.fitness_eval, evaluator.solution_eval
