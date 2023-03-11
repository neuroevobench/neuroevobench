from evosax import Strategies
from neuroevobench.problems.mnist import MNISTPolicy
from neuroevobench.problems.mnist import MNISTTask
from neuroevobench.problems.mnist import MNISTEvaluator


def test_mnist():
    """Running an ES loop on Brax task."""
    # 1. Create placeholder env to get number of actions for policy init
    policy = MNISTPolicy(hidden_dims=[])

    # 2. Define train/test task based on configs/eval settings
    train_task = MNISTTask("mnist", 10, test=False)
    test_task = MNISTTask("mnist", 10, test=False)

    # 3. Setup task evaluator with strategy and policy
    evaluator = MNISTEvaluator(
        policy=policy,
        train_task=train_task,
        test_task=test_task,
        popsize=12,
        es_strategy=Strategies["Sep_CMA_ES"],
        es_config={"elite_ratio": 0.5},
        es_params={"sigma_init": 0.025, "init_min": 0.0, "init_max": 0.0},
        seed_id=0,
    )

    # 4. Run the ES loop with logging
    evaluator.run(10, 4)
