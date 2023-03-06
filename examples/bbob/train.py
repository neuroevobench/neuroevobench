from evosax import Strategies
from neuroevobench.problems.bbob import BBOBEvaluator


def main(config, log):
    """Running an ES loop on BBOB tasks."""
    # 1. Setup task evaluator with strategy
    evaluator = BBOBEvaluator(
        popsize=config.popsize,
        num_dims=config.num_dims,
        es_strategy=Strategies[config.strategy_name],
        es_config=config.es_config,
        es_params=config.es_params,
        num_eval_runs=config.num_eval_runs,
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
