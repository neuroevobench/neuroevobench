from typing import List
import chex
import jax
from evosax import Strategies
from evosax.problems import BBOBFitness


small = ["Sphere"]
medium = [
    "Sphere",
    "RosenbrockOriginal",
    "Discus",
    "RastriginRotated",
    "Schwefel",
]
large = [
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
]
rest = [
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


def main(config, log):
    """Running an ES loop on HPO task."""
    # Setup task & network apply function & ES.
    strategy = Strategies[config.strategy_name](
        popsize=config.popsize,
        num_dims=config.num_dims,
        **config.es_config,
    )
    es_params = strategy.default_params.replace(**config.es_params)

    fns_to_eval = large + rest
    # Run ES Loop on HPO tasks -> Return mean timeseries across all tasks
    rng = jax.random.PRNGKey(config.seed_id)
    mean_perf, best_perf = eval_bbob_sweep(
        rng,
        strategy,
        config.num_dims,
        config.num_generations,
        config.num_evals,
        fns_to_eval,
        es_params,
    )

    # Loop over all mean results and return performance
    log.update(
        {"num_gens": 1},
        {**mean_perf, **best_perf},
        save=True,
    )


def eval_bbob_sweep(
    rng: chex.PRNGKey,
    strategy,
    num_dims: int,
    num_gens: int,
    num_evals: int,
    fns_to_eval: List[str],
    es_params,
):
    """Runs BBO evaluation on all BBOB tasks."""
    fn_mean, fn_best = {}, {}
    for fn in fns_to_eval:
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


if __name__ == "__main__":
    from mle_toolbox import MLExperiment

    # Setup experiment run (visible GPUs for JAX parallelism)
    mle = MLExperiment(config_fname="configs/Sep_CMA_ES/train/bbob/bbob.yaml")
    main(mle.train_config, mle.log)
