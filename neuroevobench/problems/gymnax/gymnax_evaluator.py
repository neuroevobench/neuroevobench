import jax.numpy as jnp
from evojax.obs_norm import ObsNormalizer
from evojax.sim_mgr import SimManager
from evosax.utils.evojax_wrapper import Evosax2JAX_Wrapper


class GymnaxEvaluator(object):
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
    ):
        self.popsize = popsize
        self.policy = policy
        self.es_strategy = es_strategy
        self.es_config = es_config
        self.es_params = es_params
        self.train_task = train_task
        self.test_task = test_task
        self.num_evals_per_member = num_evals_per_member
        self.seed_id = seed_id
        self.setup()

    def setup(self):
        """Initialize task, strategy & policy"""
        self.strategy = self.es_strategy(
            popsize=self.popsize,
            num_dims=self.policy.num_params,
            maximize=True,
            **self.es_config,
        )
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

    def run(self, num_generations, eval_every_gen=10, log=None):
        """Run evolution loop with logging."""
        print(f"START EVOLVING {self.policy.num_params} PARAMETERS.")
        best_return = -jnp.finfo(jnp.float32).max

        for gen_counter in range(1, num_generations + 1):
            params = self.strategy.ask()
            fit_re, _ = self.sim_mgr.eval_params(params=params, test=False)
            self.strategy.tell(fitness=fit_re)
            improved = best_return < fit_re.max()
            best_return = best_return * (1 - improved) + fit_re.max() * improved

            if gen_counter % eval_every_gen == 0:
                eval_params = jnp.stack(
                    [
                        self.strategy.es_state.mean.reshape(-1, 1),
                        self.strategy.es_state.best_member.reshape(-1, 1),
                    ]
                ).squeeze()
                mean_rewards, _ = self.sim_mgr.eval_params(
                    params=eval_params[0], test=True
                )
                best_rewards, _ = self.sim_mgr.eval_params(
                    params=eval_params[1], test=True
                )
                time_tic = {"num_gens": gen_counter}
                stats_tic = {
                    "mean_pop_perf": float(fit_re.mean()),
                    "max_pop_perf": float(fit_re.max()),
                    "best_pop_perf": float(best_return),
                    "test_eval_perf": float(mean_rewards.mean()),
                    "best_eval_perf": float(best_rewards.mean()),
                }
                if log is not None:
                    log.update(time_tic, stats_tic, save=True)
                else:
                    print(time_tic, stats_tic)
