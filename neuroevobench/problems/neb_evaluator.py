"""Abstract base class for all Neuroevolution evaluators."""
from typing import Optional, Any, Tuple
import chex
import jax


class NeuroevolutionEvaluator(object):
    def __init__(
        self,
        policy,
        train_task,
        test_task,
        popsize: int,
        es_strategy,
        es_config={},
        es_params={},
        seed_id: int = 0,
        log: Optional[Any] = None,
        time_tick_str: str = "num_gens",
        iter_id: Optional[int] = None,
        maximize_objective: bool = True,
    ):
        self.popsize = popsize
        self.policy = policy
        self.es_strategy = es_strategy
        self.es_config = es_config
        self.es_config["maximize"] = maximize_objective
        self.es_params = es_params
        self.train_task = train_task
        self.test_task = test_task
        self.seed_id = seed_id
        self.log = log
        self.time_tick_str = time_tick_str
        self.iter_id = iter_id
        self.maximize_objective = maximize_objective
        self.setup()

    def setup(self):
        """Initialize task, strategy & policy for downstream train loop."""
        self.rng = jax.random.PRNGKey(self.seed_id)
        # Brax evaluation is happening via Evojax, which requires flat vectors
        if self.problem_type in [
            "brax",
            "atari",
            "minatar",
            "mnist_classify",
            "mnist_generate",
        ]:
            self.strategy = self.es_strategy(
                popsize=self.popsize,
                num_dims=self.policy.num_params,
                **self.es_config,
            )
        else:
            self.strategy = self.es_strategy(
                popsize=self.popsize,
                pholder_params=self.policy.pholder_params,
                **self.es_config,
            )

        self.es_params = self.strategy.default_params.replace(**self.es_params)
        self.rng, rng_init = jax.random.split(self.rng)
        self.es_state = self.strategy.initialize(rng_init, self.es_params)

        self.setup_task()

    def setup_task(self):
        """Setup task-specific things (obs norm, simmanger, etc.)."""
        pass

    def run(self, num_generations: int, eval_every_gen: int):
        """Run evolution loop with logging."""
        print(f"START EVOLVING {self.strategy.num_dims} PARAMETERS.")
        # Run very first evaluation using self.es_state
        mean_es_perf, best_member_perf = self.evaluate_perf()
        time_tic = {self.time_tick_str: 0}
        if self.iter_id is not None:
            time_tic["iter_id"] = self.iter_id
        stats_tic = {
            "test_eval_perf": float(mean_es_perf),
            "best_eval_perf": float(best_member_perf),
        }
        self.update_log(time_tic, stats_tic)
        best_perf = best_member_perf

        # Run evolution loop
        for gen_counter in range(1, num_generations + 1):
            self.rng, rng_ask = jax.random.split(self.rng)
            params, self.es_state = self.strategy.ask(
                rng_ask, self.es_state, self.es_params
            )
            fitness = self.evaluate_pop(params)
            self.es_state = self.strategy.tell(
                params, fitness, self.es_state, self.es_params
            )

            # TODO(RobertTLange): Fix update of best_perf -> min/max objective
            improved = best_perf < fitness.max()
            best_perf = best_perf * (1 - improved) + fitness.max() * improved

            if gen_counter % eval_every_gen == 0:
                mean_es_perf, best_member_perf = self.evaluate_perf()
                time_tic = {self.time_tick_str: gen_counter}
                if self.iter_id is not None:
                    time_tic["iter_id"] = self.iter_id
                stats_tic = {
                    "mean_pop_perf": float(fitness.mean()),
                    "max_pop_perf": float(fitness.max()),
                    "min_pop_perf": float(fitness.max()),
                    "best_pop_perf": float(best_perf),
                    "test_eval_perf": float(mean_es_perf),
                    "best_eval_perf": float(best_member_perf),
                }
                self.update_log(time_tic, stats_tic)

    def evaluate_pop(self, params: chex.Array) -> chex.Array:
        """Evaluate a population of policy parameters on train task."""
        raise NotImplementedError

    def evaluate_perf(self) -> Tuple[chex.Array, chex.Array]:
        """Evaluate mean and best_member on test task."""
        raise NotImplementedError

    def update_log(self, time_tic, stats_tic, model: Optional[Any] = None):
        """Update logger with newest data."""
        if self.log is not None:
            self.log.update(time_tic, stats_tic, model=model, save=True)
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
        return self.es_state.mean
