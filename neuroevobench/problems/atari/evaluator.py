from typing import Optional, Any, Tuple
import chex
import jax
import jax.numpy as jnp
from ..neb_evaluator import NeuroevolutionEvaluator


class AtariEvaluator(NeuroevolutionEvaluator):
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
        self.problem_type = "atari"
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

    def evaluate_pop(self, params: chex.Array) -> chex.Array:
        """Evaluate population on train task."""
        all_rewards = self.train_task.evaluate(params)
        fitness = jnp.mean(all_rewards, axis=1)
        return fitness

    def evaluate_perf(self) -> Tuple[chex.Array, chex.Array]:
        """Evaluate mean and best_member of test task."""
        eval_params = jnp.stack(
            [
                self.es_state.mean.reshape(-1, 1),
                self.es_state.best_member.reshape(-1, 1),
            ]
        ).squeeze()
        eval_rewards = self.test_task.evaluate(eval_params).mean(axis=1)
        return eval_rewards[0], eval_rewards[1]
