"""Evosax strategy wrapper for HPO-B tasks."""
import jax
import jax.numpy as jnp
import numpy as np
from evosax import Strategy


class Evosax2HPO_Wrapper:
    def __init__(
        self,
        evosax_strategy: Strategy,
        popsize: int,
        es_config: dict = {},
        es_params: dict = {},
        seed: int = 42,
    ):
        self.evosax_strategy = evosax_strategy
        self.popsize = popsize
        self.es_config = es_config
        self.es_params = es_params
        self.rng = jax.random.PRNGKey(seed)

    def observe_and_suggest(self, X_obs, y_obs):
        """tell-ask step - reverse order in order to accomodate init."""
        self.rng, rng_ask = jax.random.split(self.rng)
        # Skip first init update - since many ES have params based on popsize
        if self.es_state.gen_counter > 0:
            self.es_state = self.strategy.tell(
                X_obs, y_obs.squeeze(), self.es_state, self.bbo_params
            )
        x_new, self.es_state = self.strategy.ask(
            rng_ask, self.es_state, self.bbo_params
        )
        x_new = jnp.clip(x_new, 0, 1)
        dim = len(X_obs[0])
        x_new = np.array(x_new).reshape(-1, dim)
        return x_new

    def init_bbo(self, num_dims: int):
        """Setup strategy for new HPO-B task with correct number of dims."""
        self.rng, rng_init = jax.random.split(self.rng)
        self.strategy = self.evosax_strategy(
            popsize=self.popsize,
            num_dims=num_dims,
            **self.es_config,
        )
        self.bbo_params = self.strategy.default_params.replace(**self.es_params)
        self.es_state = self.strategy.initialize(rng_init, self.bbo_params)
