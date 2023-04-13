from typing import Optional, Any, Tuple
import chex
import jax
import jax.numpy as jnp
from ..neb_evaluator import NeuroevolutionEvaluator


class SMNISTEvaluator(NeuroevolutionEvaluator):
    def __init__(
        self,
        policy,
        train_task,
        test_task,
        popsize,
        es_strategy,
        es_config={},
        es_params={},
        seed_id: int = 0,
        log: Optional[Any] = None,
        time_tick_str: str = "num_gens",
        iter_id: Optional[int] = None,
    ):
        self.problem_type = "smnist"
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
        # Set apply functions for both train and test tasks
        self.train_task.set_apply_fn(
            self.policy.model.apply, self.policy.model.initialize_carry
        )
        self.test_task.set_apply_fn(
            self.policy.model.apply, self.policy.model.initialize_carry
        )

    def evaluate_pop(self, params: chex.Array) -> chex.Array:
        """Evaluate population on train task."""
        self.rng, rng_eval = jax.random.split(self.rng)
        fitness, _ = self.train_task.evaluate(rng_eval, params)
        return fitness

    def evaluate_perf(self) -> Tuple[chex.Array, chex.Array]:
        """Evaluate mean and best_member of test task."""
        self.rng, rng_test = jax.random.split(self.rng)
        eval_params = jnp.stack(
            [
                self.es_state.mean.reshape(-1, 1),
                self.es_state.best_member.reshape(-1, 1),
            ]
        ).squeeze()
        eval_params = self.strategy.param_reshaper.reshape(eval_params)
        _, test_perf = self.test_task.evaluate(rng_test, eval_params)
        return test_perf[0], test_perf[1]
