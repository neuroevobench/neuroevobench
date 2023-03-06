import jax
from evosax.problems import BBOBFitness


bbob_fns = [
    "Sphere",
    "RosenbrockRotated",
    "Discus",
    "RastriginRotated",
    "Schwefel",
    "BuecheRastrigin",
    "AttractiveSector",
    "Weierstrass",
    "SchaffersF7",
    "GriewankRosenbrock",
    # Part 1: Separable functions
    "EllipsoidalOriginal",
    "RastriginOriginal",
    "LinearSlope",
    # Part 2: Functions with low or moderate conditions
    "StepEllipsoidal",
    "RosenbrockOriginal",
    # Part 3: Functions with high conditioning and unimodal
    "EllipsoidalRotated",
    "BentCigar",
    "SharpRidge",
    "DifferentPowers",
    # Part 4: Multi-modal functions with adequate global structure
    "SchaffersF7IllConditioned",
    # Part 5: Multi-modal functions with weak global structure
    "Lunacek",
    "Gallagher101Me",
    "Gallagher21Hi",
]


class BBOBEvaluator(object):
    def __init__(
        self,
        popsize: int,
        num_dims: int,
        es_strategy,
        es_config={},
        es_params={},
        num_eval_runs: int = 50,
        seed_id: int = 0,
    ):
        self.popsize = popsize
        self.num_dims = num_dims
        self.es_strategy = es_strategy
        self.es_config = es_config
        self.es_params = es_params
        self.num_eval_runs = num_eval_runs
        self.seed_id = seed_id
        self.setup()

    def setup(self):
        """Initialize task, strategy & policy"""
        self.strategy = self.es_strategy(
            popsize=self.popsize,
            num_dims=self.num_dims,
            **self.es_config,
        )
        self.es_params = self.strategy.default_params.replace(**self.es_params)

    def run(self, num_generations, log=None):
        """Run evolution loop with logging."""
        print(f"START EVOLVING {self.num_dims} PARAMETERS.")
        mean_perf, best_perf = eval_bbob_sweep(
            self.strategy,
            self.num_dims,
            num_generations,
            self.num_eval_runs,
            self.es_params,
            self.seed_id,
        )
        if log is not None:
            log.update(
                {"num_gens": 1},
                {**mean_perf, **best_perf},
                save=True,
            )
        else:
            print(mean_perf, best_perf)


def eval_bbob_sweep(
    strategy,
    num_dims: int,
    num_gens: int,
    num_evals: int,
    es_params,
    seed_id: int,
):
    """Runs BBO evaluation on all BBOB tasks."""
    fn_mean, fn_best = {}, {}
    rng = jax.random.PRNGKey(seed_id)
    for fn in bbob_fns:
        rng, rng_fn = jax.random.split(rng)
        evaluator = BBOBFitness(fn, num_dims)

        def run_es_loop(rng, num_gens, es_params):
            """Run ES loop on single BBOB function."""
            rng, rng_init = jax.random.split(rng)
            es_state = strategy.initialize(rng_init, es_params)
            for _ in range(num_gens):
                rng, rng_ask, rng_eval = jax.random.split(rng, 3)
                x, es_state = strategy.ask(rng_ask, es_state, es_params)
                fitness = evaluator.rollout(rng_eval, x)
                es_state = strategy.tell(x, fitness, es_state, es_params)
            rng, rng_final = jax.random.split(rng)
            final_perf = evaluator.rollout(
                rng_final, es_state.mean.reshape(1, -1)
            )
            return final_perf, es_state.best_fitness

        batch_rng = jax.random.split(rng_fn, num_evals)
        final_perf, best_perf = jax.vmap(run_es_loop, in_axes=(0, None, None))(
            batch_rng, num_gens, es_params
        )
        fn_mean[fn + "_mean"] = final_perf.mean()
        fn_best[fn + "_best"] = best_perf.mean()
    return fn_mean, fn_best
