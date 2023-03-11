import jax
import jax.numpy as jnp


class AdditionEvaluator(object):
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
    ):
        self.popsize = popsize
        self.policy = policy
        self.es_strategy = es_strategy
        self.es_config = es_config
        self.es_config["maximize"] = True
        self.es_params = es_params
        self.train_task = train_task
        self.test_task = test_task
        self.seed_id = seed_id
        self.setup()

    def setup(self):
        """Initialize task, strategy & policy"""
        self.strategy = self.es_strategy(
            popsize=self.popsize,
            pholder_params=self.policy.pholder_params,
            **self.es_config,
        )
        self.es_params = self.strategy.default_params.replace(**self.es_params)
        # Set apply functions for both train and test tasks
        self.train_task.set_apply_fn(
            self.policy.model.apply, self.policy.model.initialize_carry
        )
        self.test_task.set_apply_fn(
            self.policy.model.apply, self.policy.model.initialize_carry
        )

    def run(self, num_generations: int, eval_every_gen: int = 10, log=None):
        """Run evolution loop with logging."""
        print(f"START EVOLVING {self.strategy.num_dims} PARAMETERS.")
        best_perf = jnp.finfo(jnp.float32).max
        rng = jax.random.PRNGKey(self.seed_id)
        rng, rng_init = jax.random.split(rng)
        es_state = self.strategy.initialize(rng_init, self.es_params)

        # Run evolution loop for number of generations
        for gen_counter in range(1, num_generations + 1):
            rng, rng_ask, rng_eval = jax.random.split(rng, 3)
            params, es_state = self.strategy.ask(
                rng_ask, es_state, self.es_params
            )
            fitness, _ = self.train_task.evaluate(rng_eval, params)
            es_state = self.strategy.tell(
                params, fitness, es_state, self.es_params
            )
            improved = best_perf > fitness.max()
            best_perf = best_perf * (1 - improved) + fitness.max() * improved

            if gen_counter % eval_every_gen == 0:
                rng, rng_test = jax.random.split(rng)
                eval_params = jnp.stack(
                    [
                        es_state.mean.reshape(-1, 1),
                        es_state.best_member.reshape(-1, 1),
                    ]
                ).squeeze()
                eval_params = self.strategy.param_reshaper.reshape(eval_params)
                _, test_acc = self.test_task.evaluate(rng_test, eval_params)
                time_tic = {"num_gens": gen_counter}
                stats_tic = {
                    "mean_pop_perf": float(fitness.mean()),
                    "max_pop_perf": float(fitness.max()),
                    "best_pop_perf": float(best_perf),
                    "test_eval_perf": float(test_acc[0]),
                    "best_eval_perf": float(test_acc[1]),
                }
                if log is not None:
                    log.update(time_tic, stats_tic, save=True)
                else:
                    print(time_tic, stats_tic)
