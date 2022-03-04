import jax
from evosax import ProblemMapper


def setup_es_problem(config):
    rng = jax.random.PRNGKey(config.seed_id)
    train_eval = ProblemMapper[config.problem_name](
        **config.problem_config.toDict()
    )
    test_eval = ProblemMapper[config.problem_name](
        **config.problem_config.toDict()
    )
    return rng, train_eval, test_eval
