from typing import Tuple, Optional, Any
import chex
import jax.numpy as jnp
from evojax.obs_norm import ObsNormalizer
from evojax.sim_mgr import SimManager
from ..neb_evaluator import NeuroevolutionEvaluator


class MNIST_Generate_Evaluator(NeuroevolutionEvaluator):
    def __init__(
        self,
        policy,
        train_task,
        test_task,
        popsize,
        es_strategy,
        es_config={},
        es_params={},
        num_evals_per_member: int = 1,
        seed_id: int = 0,
        log: Optional[Any] = None,
        time_tick_str: str = "num_gens",
        iter_id: Optional[int] = None,
    ):
        self.problem_type = "mnist_generate"
        self.num_evals_per_member = num_evals_per_member
        super().__init__(
            policy,
            train_task,
            test_task,
            popsize,
            es_strategy,
            es_config,
            es_params,
            seed_id,
            log,
            time_tick_str,
            iter_id,
            maximize_objective=True,
        )

    def setup_task(self):
        """Initialize task, strategy & policy"""
        self.obs_normalizer = ObsNormalizer(
            obs_shape=self.train_task.obs_shape, dummy=True
        )

        self.sim_mgr = SimManager(
            policy_net=self.policy,
            train_vec_task=self.train_task,
            valid_vec_task=self.test_task,
            seed=self.seed_id,
            obs_normalizer=self.obs_normalizer,
            pop_size=self.popsize,
            use_for_loop=True,
            n_repeats=1,
            test_n_repeats=1,
            n_evaluations=1,
        )

    def evaluate_pop(self, params: chex.Array) -> chex.Array:
        """Evaluate population on train task."""
        fitness, _ = self.sim_mgr.eval_params(params=params, test=False)
        return fitness

    def evaluate_perf(self) -> Tuple[chex.Array, chex.Array]:
        """Evaluate mean and best_member of test task."""
        eval_params = jnp.stack(
            [
                self.es_state.mean.reshape(-1, 1),
                self.es_state.best_member.reshape(-1, 1),
            ]
        ).squeeze()
        mean_return, _ = self.sim_mgr.eval_params(
            params=eval_params[0], test=True
        )
        best_return, _ = self.sim_mgr.eval_params(
            params=eval_params[1], test=True
        )
        return mean_return.mean(), best_return.mean()
