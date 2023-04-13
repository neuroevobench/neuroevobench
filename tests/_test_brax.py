from brax.v1.envs import create
from evosax import Strategies
from neuroevobench.problems.brax import BraxPolicy
from neuroevobench.problems.brax import BraxTask
from neuroevobench.problems.brax import BraxEvaluator


def no_test_brax():
    """Running an ES loop on Brax task."""
    # 1. Create placeholder env to get number of actions for policy init
    env = create(env_name="ant", legacy_spring=True)
    policy = BraxPolicy(
        input_dim=env.observation_size,
        output_dim=env.action_size,
        hidden_dims=[32],
    )

    # 2. Define train/test task based on configs/eval settings
    train_task = BraxTask("ant", 100, test=False)
    test_task = BraxTask("ant", 100, test=True)

    # 3. Setup task evaluator with strategy and policy
    evaluator = BraxEvaluator(
        policy=policy,
        train_task=train_task,
        test_task=test_task,
        popsize=12,
        es_strategy=Strategies["Sep_CMA_ES"],
        es_config={"elite_ratio": 0.01},
        es_params={"sigma_init": 0.05, "init_min": 0.0, "init_max": 0.0},
        num_evals_per_member=4,
        seed_id=0,
    )

    # 4. Run the ES loop with logging
    evaluator.run(4, 2)
