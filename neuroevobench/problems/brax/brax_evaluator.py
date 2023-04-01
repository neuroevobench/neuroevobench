from typing import Tuple, Optional, Any
import chex
import jax.numpy as jnp
from evojax.obs_norm import ObsNormalizer
from evojax.sim_mgr import SimManager
from evosax.utils.evojax_wrapper import Evosax2JAX_Wrapper


class BraxEvaluator(object):
    def __init__(
        self,
        policy,
        train_task,
        test_task,
        popsize,
        es_strategy,
        es_config={},
        es_params={},
        num_evals_per_member: int = 8,
        seed_id: int = 0,
        log: Optional[Any] = None,
        iter_id: Optional[int] = None,
    ):
        self.popsize = popsize
        self.policy = policy
        self.es_strategy = es_strategy
        self.es_config = es_config
        self.es_config["maximize"] = True
        self.es_params = es_params
        self.train_task = train_task
        self.test_task = test_task
        self.num_evals_per_member = num_evals_per_member
        self.seed_id = seed_id
        self.log = log
        self.iter_id = iter_id
        self.setup()

    def setup(self):
        """Initialize task, strategy & policy"""
        self.strategy = Evosax2JAX_Wrapper(
            self.es_strategy,
            param_size=self.policy.num_params,
            pop_size=self.popsize,
            es_config=self.es_config,
            es_params=self.es_params,
            seed=self.seed_id,
        )

        self.obs_normalizer = ObsNormalizer(
            obs_shape=self.train_task.obs_shape, dummy=False
        )

        self.sim_mgr = SimManager(
            policy_net=self.policy,
            train_vec_task=self.train_task,
            valid_vec_task=self.test_task,
            seed=self.seed_id,
            obs_normalizer=self.obs_normalizer,
            pop_size=self.popsize,
            use_for_loop=False,
            n_repeats=self.num_evals_per_member,
            test_n_repeats=1,
            n_evaluations=128,
        )

    def run(self, num_generations: int = 2000, eval_every_gen=50):
        """Run evolution loop with logging."""
        print(f"START EVOLVING {self.policy.num_params} PARAMETERS.")
        # Run very first evaluation
        mean_es_returns, best_member_returns = self.evaluate(
            self.strategy.es_state
        )
        time_tic = {"num_gens": 0}
        stats_tic = {
            "test_eval_perf": float(mean_es_returns),
            "best_eval_perf": float(best_member_returns),
        }
        self.update_log(time_tic, stats_tic)
        best_return = best_member_returns

        # Run evolution loop
        for gen_counter in range(1, num_generations + 1):
            params = self.strategy.ask()
            fitness, _ = self.sim_mgr.eval_params(params=params, test=False)
            self.strategy.tell(fitness=fitness)
            improved = best_return < fitness.max()
            best_return = (
                best_return * (1 - improved) + fitness.max() * improved
            )

            if gen_counter % eval_every_gen == 0:
                mean_es_returns, best_member_returns = self.evaluate(
                    self.strategy.es_state
                )
                time_tic = {"num_gens": gen_counter}
                if self.iter_id is not None:
                    time_tic["iter_id"] = self.iter_id
                stats_tic = {
                    "mean_pop_perf": float(fitness.mean()),
                    "max_pop_perf": float(fitness.max()),
                    "best_pop_perf": float(best_return),
                    "test_eval_perf": float(mean_es_returns),
                    "best_eval_perf": float(best_member_returns),
                }
                self.update_log(time_tic, stats_tic)

    def evaluate(self, es_state) -> Tuple[chex.Array, chex.Array]:
        """Evaluate mean and best_member of test task."""
        eval_params = jnp.stack(
            [
                es_state.mean.reshape(-1, 1),
                es_state.best_member.reshape(-1, 1),
            ]
        ).squeeze()
        mean_return, _ = self.sim_mgr.eval_params(
            params=eval_params[0], test=True
        )
        best_return, _ = self.sim_mgr.eval_params(
            params=eval_params[1], test=True
        )
        return mean_return.mean(), best_return.mean()

    def update_log(self, time_tic, stats_tic, model: Optional[Any] = None):
        """Update logger with newest data."""
        if self.log is not None:
            self.log.update(time_tic, stats_tic, save=True)
        else:
            print(time_tic, stats_tic)

    @property
    def fitness_eval(self):
        """Get latest fitness evaluation score."""
        # TODO(Robert): Hack to get the latest fitness score - CSVLogger?
        return self.log.stats_log.stats_tracked["test_eval_perf"][-1]

    @property
    def solution_eval(self):
        """Get latest solution parameters."""
        return self.strategy.solution
