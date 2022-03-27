import jax
import jax.numpy as jnp
from evosax import ProblemMapper, NetworkMapper, ParameterReshaper, ARS


def main(problem_name="Sequence"):
    evaluator = ProblemMapper[problem_name](task_name="Addition", test=False)

    network = NetworkMapper["LSTM"](
        num_hidden_units=100,
        num_output_units=evaluator.action_shape,
    )

    print(evaluator.input_shape[0], evaluator.input_shape)

    rng = jax.random.PRNGKey(0)
    params = network.init(
        rng,
        x=jnp.ones([1, evaluator.input_shape[0]]),
        carry=network.initialize_carry(),
        rng=rng,
    )
    param_reshaper = ParameterReshaper(params["params"])
    evaluator.set_apply_fn(
        param_reshaper.vmap_dict,
        network.apply,
        network.initialize_carry,
    )

    strategy = ARS(param_reshaper.total_params, 20)
    print(param_reshaper.total_params)
    es_params = strategy.default_params
    es_state = strategy.initialize(rng, es_params)

    x, es_state = strategy.ask(rng, es_state, es_params)
    reshaped_params = param_reshaper.reshape(x)
    # return
    # Rollout population performance, reshape fitness & update strategy.
    fitness = evaluator.rollout(rng, reshaped_params)
    print(fitness[0].shape)
    print(fitness[1].shape)
    print(fitness[0])
    print(fitness[1])


if __name__ == "__main__":
    main()
