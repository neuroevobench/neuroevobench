from evosax import Strategies, FitnessShaper, ParameterReshaper
from evosax import ProblemMapper, NetworkMapper


def main(train_config, log):
    # Setup task.
    train_task, test_task, policy = setup_problem(config, logger)

    # Setup network.
    NetworkMapper

    # Setup Parameter Shaper.
    ParameterReshaper

    # Setup ES.
    strategy = Strategies[config.es_name](
        **config.es_config.toDict(),
        param_size=policy.num_params,
        seed=config.seed,
    )

    # Setup Fitness Shaping.
    FitnessShaper

    # Run ES Loop.
    for t in range(train_config.num_generations):
        pass
    return


if __name__ == "__main__":
    from mle_toolbox import MLExperiment

    mle = MLExperiment(config_fname="configs/base.yaml")
    main(mle.train_config, mle.log)
