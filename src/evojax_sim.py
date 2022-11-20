import numpy as np
from evojax.obs_norm import ObsNormalizer
from evojax.sim_mgr import SimManager
from evosax.utils.evojax_wrapper import Evosax2JAX_Wrapper
from evojax_tasks import GymnaxTask, MinAtarPolicy, MNISTTask
from evosax import Sep_CMA_ES

from evojax.task.brax_task import BraxTask
from evojax.policy import MLPPolicy
from evojax.policy.convnet import ConvNetPolicy


def get_brax_task():
    train_task = BraxTask("ant", test=False)
    test_task = BraxTask("ant", test=True)
    policy = MLPPolicy(
        input_dim=train_task.obs_shape[0],
        output_dim=train_task.act_shape[0],
        hidden_dims=4 * [32],
    )
    return train_task, test_task, policy


def get_minatar_task():
    train_task = GymnaxTask("SpaceInvaders-MinAtar", max_steps=500, test=False)
    test_task = GymnaxTask("SpaceInvaders-MinAtar", max_steps=500, test=True)
    policy = MinAtarPolicy(
        input_dim=train_task.obs_shape,
        output_dim=train_task.num_actions,
        hidden_dim=32,
    )
    return train_task, test_task, policy


def get_mnist_task():
    # mnist, fashion_mnist, kmnist, mnist_corrupted
    train_task = MNISTTask("mnist", batch_size=1024, test=False)
    test_task = MNISTTask("mnist", batch_size=None, test=True)
    policy = ConvNetPolicy()
    return train_task, test_task, policy


def run_brax_sim():
    train_task, test_task, policy = get_brax_task()
    solver = Evosax2JAX_Wrapper(
        Sep_CMA_ES,
        param_size=policy.num_params,
        pop_size=256,
        es_config={"elite_ratio": 0.4, "maximize": True},
        es_params={"sigma_init": 0.05, "init_min": 0.0, "init_max": 0.0},
        seed=0,
    )
    obs_normalizer = ObsNormalizer(obs_shape=train_task.obs_shape, dummy=False)
    sim_mgr = SimManager(
        n_repeats=16,
        test_n_repeats=1,
        pop_size=256,
        n_evaluations=128,
        policy_net=policy,
        train_vec_task=train_task,
        valid_vec_task=test_task,
        seed=0,
        obs_normalizer=obs_normalizer,
        use_for_loop=False,
    )

    for gen_counter in range(2000):
        params = solver.ask()
        scores, _ = sim_mgr.eval_params(params=params, test=False)
        solver.tell(fitness=scores)

        if gen_counter == 0 or (gen_counter + 1) % 10 == 0:
            scores = np.array(scores)
            experiment_log = {
                "gen": gen_counter + 1,
                "pop_return": float(np.nanmean(scores)),
            }

            if gen_counter == 0 or (gen_counter + 1) % 50 == 0:
                test_scores, _ = sim_mgr.eval_params(
                    params=solver.best_params, test=True
                )
                experiment_log["test_return"] = float(np.nanmean(test_scores))
            print(experiment_log)


def run_minatar_sim():
    train_task, test_task, policy = get_minatar_task()
    solver = Evosax2JAX_Wrapper(
        Sep_CMA_ES,
        param_size=policy.num_params,
        pop_size=256,
        es_config={"elite_ratio": 0.1, "maximize": True},
        es_params={"sigma_init": 0.05, "init_min": 0.0, "init_max": 0.0},
        seed=0,
    )
    obs_normalizer = ObsNormalizer(obs_shape=train_task.obs_shape, dummy=True)
    sim_mgr = SimManager(
        n_repeats=4,
        test_n_repeats=1,
        pop_size=256,
        n_evaluations=64,
        policy_net=policy,
        train_vec_task=train_task,
        valid_vec_task=test_task,
        seed=0,
        obs_normalizer=obs_normalizer,
        use_for_loop=False,
    )

    for gen_counter in range(5000):
        params = solver.ask()
        scores, _ = sim_mgr.eval_params(params=params, test=False)
        solver.tell(fitness=scores)

        if gen_counter == 0 or (gen_counter + 1) % 10 == 0:
            scores = np.array(scores)
            experiment_log = {
                "gen": gen_counter + 1,
                "pop_return": float(np.nanmean(scores)),
            }

            if gen_counter == 0 or (gen_counter + 1) % 50 == 0:
                test_scores, _ = sim_mgr.eval_params(
                    params=solver.best_params, test=True
                )
                experiment_log["test_return"] = float(np.nanmean(test_scores))
            print(experiment_log)


def run_mnist_sim():
    train_task, test_task, policy = get_mnist_task()
    solver = Evosax2JAX_Wrapper(
        Sep_CMA_ES,
        param_size=policy.num_params,
        pop_size=128,
        es_config={"elite_ratio": 0.5, "maximize": True},
        es_params={"sigma_init": 0.025, "init_min": 0.0, "init_max": 0.0},
        seed=0,
    )
    obs_normalizer = ObsNormalizer(obs_shape=train_task.obs_shape, dummy=True)
    sim_mgr = SimManager(
        n_repeats=1,
        test_n_repeats=1,
        pop_size=128,
        n_evaluations=1,
        policy_net=policy,
        train_vec_task=train_task,
        valid_vec_task=test_task,
        seed=0,
        obs_normalizer=obs_normalizer,
        use_for_loop=False,
    )

    for gen_counter in range(4000):
        params = solver.ask()
        scores, _ = sim_mgr.eval_params(params=params, test=False)
        solver.tell(fitness=scores)

        if gen_counter == 0 or (gen_counter + 1) % 10 == 0:
            scores = np.array(scores)
            experiment_log = {
                "gen": gen_counter + 1,
                "pop_return": float(np.nanmean(scores)),
            }

            if gen_counter == 0 or (gen_counter + 1) % 50 == 0:
                test_scores, _ = sim_mgr.eval_params(
                    params=solver.best_params, test=True
                )
                experiment_log["test_return"] = float(np.nanmean(test_scores))
            print(experiment_log)


if __name__ == "__main__":
    run_brax_sim()
    # run_minatar_sim()
    # run_mnist_sim()
