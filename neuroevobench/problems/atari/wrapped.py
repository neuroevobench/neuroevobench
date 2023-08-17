from typing import Optional
from evosax import Strategy
from .policy import AtariPolicy
from .task import AtariTask
from .evaluator import AtariEvaluator
from ...utils import collect_strategies

Strategies = collect_strategies()

try:
    import envpool
except Exception:
    print("envpool not installed, Atari problems will not work.")


def atari_run(
    config,
    log,
    search_iter: Optional[int] = None,
    strategy_class: Optional[Strategy] = None,
):
    """Running an EO loop on ATARI task."""
    # 1. Create placeholder env to get number of actions for policy init
    env = envpool.make_gym(config.task_config.env_name, num_envs=1)
    policy = AtariPolicy(num_actions=env.action_space.n, **config.model_config)

    # 2. Define train/test task based on configs/eval settings
    train_task = AtariTask(
        policy,
        popsize=config.popsize,
        **config.task_config,
    )
    test_task = AtariTask(
        policy,
        popsize=2,
        env_name=config.task_config.env_name,
        max_steps=config.task_config.max_steps,
        num_envs_per_member=20,
    )

    # 3. Setup task evaluator with strategy and policy
    if strategy_class is not None:
        base_strategy = strategy_class
    else:
        base_strategy = Strategies[config.strategy_name]

    evaluator = AtariEvaluator(
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
    evaluator.run(
        config.num_generations,
        config.eval_every_gen,
    )

    # 5. Return mean params and final performance
    return evaluator.fitness_eval, evaluator.solution_eval
