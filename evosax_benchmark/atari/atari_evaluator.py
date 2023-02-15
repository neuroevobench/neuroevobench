import jax
import jax.numpy as jnp
from .atari_task import AtariTask


class AtariEvaluator(object):
    def __init__(
        self,
        policy,
        popsize,
        es_strategy,
        es_config={},
        es_params={},
        task_config={},
    ):
        self.popsize = popsize
        self.policy = policy
        self.es_strategy = es_strategy
        self.es_config = es_config
        self.es_params = es_params
        self.train_task_config = dict(task_config)
        self.test_task_config = dict(task_config)
        self.train_task_config["popsize"] = popsize
        self.test_task_config["popsize"] = 2  # best and mean
        self.test_task_config["num_envs_per_member"] = 20
        self.setup()

    def setup(self):
        """Initialize task, strategy & policy"""
        self.train_task = AtariTask(self.policy, **self.train_task_config)
        self.test_task = AtariTask(self.policy, **self.test_task_config)
        self.strategy = self.es_strategy(
            popsize=self.popsize,
            num_dims=self.train_task.num_dims,
            **self.es_config,
        )
        self.es_params = self.strategy.default_params.replace(**self.es_params)

    def run(self, seed_id, num_generations, eval_every_gen=10, log=None):
        """Run evolution loop with logging."""
        print(f"START EVOLVING {self.train_task.num_dims} PARAMETERS.")

        rng = jax.random.PRNGKey(seed_id)
        es_state = self.strategy.initialize(rng)
        best_return = -jnp.finfo(jnp.float32).max

        for gen_counter in range(num_generations):
            rng, rng_ask = jax.random.split(rng)
            params, es_state = self.strategy.ask(rng_ask, es_state)
            all_rewards = self.train_task.evaluate(params)
            fit_re = jnp.mean(all_rewards, axis=1)
            es_state = self.strategy.tell(params, fit_re, es_state)
            improved = best_return < fit_re.max()
            best_return = best_return * (1 - improved) + fit_re.max() * improved

            if gen_counter % eval_every_gen == 0:
                eval_params = jnp.stack(
                    [
                        es_state.mean.reshape(-1, 1),
                        es_state.best_member.reshape(-1, 1),
                    ]
                ).squeeze()
                eval_rewards = self.test_task.evaluate(eval_params).mean(axis=1)
                time_tic = {"num_gens": gen_counter + 1}
                stats_tic = {
                    "mean_pop_perf": float(fit_re.mean()),
                    "max_pop_perf": float(fit_re.max()),
                    "best_pop_perf": float(best_return),
                    "test_eval_perf": float(eval_rewards[0]),
                    "best_eval_perf": float(eval_rewards[1]),
                }
                if log is not None:
                    log.update(time_tic, stats_tic, save=True)
                else:
                    print(time_tic, stats_tic)
