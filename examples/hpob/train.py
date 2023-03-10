from evosax import Strategies
from neuroevobench.problems.hpob import HPOBEvaluator


def main(config, log):
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
    evaluator.run(
        config.num_generations,
        log=log,
    )


if __name__ == "__main__":
    from mle_toolbox import MLExperiment

    # Setup experiment run (visible GPUs for JAX parallelism)
    mle = MLExperiment(config_fname="configs/train.yaml")
    main(mle.train_config, mle.log)
