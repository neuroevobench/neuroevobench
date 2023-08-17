from typing import Optional
import gymnax
from gymnax.wrappers.evojax import GymnaxToEvoJaxTask
from evosax import Strategy
from .policy import MinAtarPolicy
from .evaluator import MinAtarEvaluator
from ...utils import collect_strategies

Strategies = collect_strategies()


def minatar_run(
    config,
    log,
    search_iter: Optional[int] = None,
    strategy_class: Optional[Strategy] = None,
):
    """Running an ES loop on MinAtar control task."""
    # 1. Create placeholder env to get number of actions for policy init
    env, _ = gymnax.make(config.env_name)

    policy = MinAtarPolicy(
        input_dim=env.obs_shape,
        output_dim=env.num_actions,
        hidden_dims=config.model_config.num_hidden_layers
        * [config.model_config.num_hidden_units],
    )

    # 2. Define train/test task based on configs/eval settings
    train_task = GymnaxToEvoJaxTask(
        config.env_name, config.task_config.max_steps, test=False
    )
    test_task = GymnaxToEvoJaxTask(
        config.env_name, config.task_config.max_steps, test=True
    )

    # 3. Setup task evaluator with strategy and policy
    if strategy_class is not None:
        base_strategy = strategy_class
    else:
        base_strategy = Strategies[config.strategy_name]

    evaluator = MinAtarEvaluator(
        policy=policy,
        train_task=train_task,
        test_task=test_task,
        popsize=config.popsize,
        es_strategy=base_strategy,
        es_config=config.es_config,
        es_params=config.es_params,
        num_evals_per_member=config.task_config.num_evals_per_member,
        seed_id=config.seed_id,
        log=log,
        iter_id=search_iter,
    )

    # 4. Run the ES loop with logging
    evaluator.run(config.num_generations, config.eval_every_gen)

    # 5. Return mean params and final performance
    return evaluator.fitness_eval, evaluator.solution_eval
