import gymnax
from gymnax.utils.evojax_wrapper import GymnaxTask
from evosax import Strategies
from evosax_benchmark.gymnax import GymPolicy
from evosax_benchmark.gymnax import MinAtarPolicy
from evosax_benchmark.gymnax import GymnaxEvaluator


def main(config, log):
    """Running an ES loop on Brax task."""
    # 1. Create placeholder env to get number of actions for policy init
    env, _ = gymnax.make(config.env_name)

    if config.env_name in ["Pendulum-v1"]:
        policy = GymPolicy(
            input_dim=env.obs_shape,
            output_dim=env.num_actions,
            hidden_dims=config.model_config.num_hidden_layers
            * [config.model_config.num_hidden_units],
        )
    else:
        policy = MinAtarPolicy(
            input_dim=env.obs_shape,
            output_dim=env.num_actions,
            hidden_dims=config.model_config.num_hidden_layers
            * [config.model_config.num_hidden_units],
        )

    # 2. Define train/test task based on configs/eval settings
    train_task = GymnaxTask(
        config.env_name, config.task_config.max_steps, test=False
    )
    test_task = GymnaxTask(
        config.env_name, config.task_config.max_steps, test=True
    )

    # 3. Setup task evaluator with strategy and policy
    evaluator = GymnaxEvaluator(
        policy=policy,
        train_task=train_task,
        test_task=test_task,
        popsize=config.popsize,
        es_strategy=Strategies[config.strategy_name],
        es_config=config.es_config,
        es_params=config.es_params,
        num_evals_per_member=config.task_config.num_evals_per_member,
        seed_id=config.seed_id,
    )

    # 4. Run the ES loop with logging
    evaluator.run(
        config.num_generations,
        config.eval_every_gen,
        log=log,
    )


if __name__ == "__main__":
    from mle_toolbox import MLExperiment

    # Setup experiment run (visible GPUs for JAX parallelism)
    mle = MLExperiment(config_fname="configs/train.yaml")
    main(mle.train_config, mle.log)
