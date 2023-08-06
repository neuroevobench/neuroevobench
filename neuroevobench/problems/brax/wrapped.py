from typing import Optional
from evosax import Strategies, Strategy
from .policy import BraxPolicy
from .task import BraxTask
from .evaluator import BraxEvaluator
from ...blines import BayesOpt

Strategies["BayesOpt"] = BayesOpt


def brax_run(
    config,
    log,
    search_iter: Optional[int] = None,
    strategy_class: Optional[Strategy] = None,
):
    """Running an ES loop on Brax task."""
    from brax.v1.envs import create

    # 1. Create placeholder env to get number of actions for policy init
    env = create(env_name=config.env_name, legacy_spring=True)
    policy = BraxPolicy(
        input_dim=env.observation_size,
        output_dim=env.action_size,
        hidden_dims=config.model_config.num_hidden_layers
        * [config.model_config.num_hidden_units],
        hidden_act_fn=config.model_config.hidden_act_fn,
    )

    # 2. Define train/test task based on configs/eval settings
    train_task = BraxTask(
        config.env_name,
        config.task_config.max_steps,
        config.task_config.noise_level,
        test=False,
    )
    test_task = BraxTask(
        config.env_name, config.task_config.max_steps, test=True
    )

    # 3. Setup task evaluator with strategy and policy
    if strategy_class is not None:
        base_strategy = strategy_class
    else:
        base_strategy = Strategies[config.strategy_name]

    evaluator = BraxEvaluator(
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
