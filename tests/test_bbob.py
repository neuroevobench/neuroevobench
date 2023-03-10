from evosax import Strategies
from neuroevobench.problems.bbob import BBOBEvaluator


def test_bbob():
    """Running an ES loop on BBOB tasks."""
    # 1. Setup task evaluator with strategy
    evaluator = BBOBEvaluator(
        popsize=4,
        num_dims=2,
        es_strategy=Strategies["Sep_CMA_ES"],
        es_config={
            "elite_ratio": 0.5,
        },
        es_params={
            "sigma_init": 0.5,
            "init_min": -5.0,
            "init_max": 5.0,
        },
        num_eval_runs=2,
        seed_id=0,
    )

    # 2. Run the ES loop with logging
    evaluator.run(5)
