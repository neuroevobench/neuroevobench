import gymnax
from gymnax.utils.evojax_wrapper import GymnaxTask
from evosax import Strategies
from neuroevobench.problems.gymnax import GymPolicy
from neuroevobench.problems.gymnax import GymnaxEvaluator


def no_test_gym():
    """Running an ES loop on Brax task."""
    # 1. Create placeholder env to get number of actions for policy init
    env, _ = gymnax.make("Pendulum-v1")

    policy = GymPolicy(
        input_dim=env.obs_shape,
        output_dim=env.num_actions,
        hidden_dims=[32],
    )

    # 2. Define train/test task based on configs/eval settings
    train_task = GymnaxTask("Pendulum-v1", 500, test=False)
    test_task = GymnaxTask("Pendulum-v1", 500, test=True)

    # 3. Setup task evaluator with strategy and policy
    evaluator = GymnaxEvaluator(
        policy=policy,
        train_task=train_task,
        test_task=test_task,
        popsize=8,
        es_strategy=Strategies["Sep_CMA_ES"],
        es_config={"elite_ratio": 0.5},
        es_params={"sigma_init": 0.05, "init_min": 0.0, "init_max": 0.0},
        num_evals_per_member=8,
        seed_id=0,
    )

    # 4. Run the ES loop with logging
    evaluator.run(5, 2)
