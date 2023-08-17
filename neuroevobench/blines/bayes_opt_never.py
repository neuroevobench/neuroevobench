"""Bayesian Optimization Wrapper to benchmark agains - not JAX compatible."""
from typing import Optional, Union, Tuple
import chex
import numpy as np
import jax
import jax.numpy as jnp
from flax import struct
from evosax.core import ParameterReshaper, FitnessShaper
from evosax.utils import get_best_fitness_member


@struct.dataclass
class EvoState:
    mean: chex.Array
    best_member: chex.Array
    best_fitness: float = jnp.finfo(jnp.float32).max
    gen_counter: int = 0


@struct.dataclass
class EvoParams:
    init_min: float = -1.0  # going to be ignored
    init_max: float = 1.0  # going to be ignored
    search_min: float = -3.0
    search_max: float = 3.0
    clip_min: float = -jnp.finfo(jnp.float32).max
    clip_max: float = jnp.finfo(jnp.float32).max


class BayesOptNevergrad(object):
    def __init__(
        self,
        popsize: int,
        num_dims: Optional[int] = None,
        pholder_params: Optional[Union[chex.ArrayTree, chex.Array]] = None,
        n_pca_components: float = 0.95,
        param_range: float = 2.0,
        sigma_init: float = 0.1,  # going to be ignored
        n_devices: Optional[int] = None,
        **fitness_kwargs: Union[bool, int, float]
    ):
        self.popsize = popsize

        # Setup optional parameter reshaper
        self.use_param_reshaper = pholder_params is not None
        if self.use_param_reshaper:
            self.param_reshaper = ParameterReshaper(pholder_params, n_devices)
            self.num_dims = self.param_reshaper.total_params
        else:
            self.num_dims = num_dims
        assert (
            self.num_dims is not None
        ), "Provide either num_dims or pholder_params to strategy."

        # Set default hyperparameters for SMBO
        self.n_pca_components = n_pca_components
        self.param_range = param_range

        # Setup optional fitness shaper
        self.fitness_shaper = FitnessShaper(**fitness_kwargs)

    @property
    def default_params(self) -> EvoParams:
        """Return default parameters of evolution strategy."""
        return EvoParams(
            search_min=-self.param_range, search_max=self.param_range
        )

    def initialize(
        self,
        rng: chex.PRNGKey,
        params: Optional[EvoParams] = None,
    ):
        # Use default hyperparameters if no other settings provided
        if params is None:
            params = self.default_params

        try:
            import nevergrad as ng
            from nevergrad.optimization.optimizerlib import BayesOptim

        except ImportError:
            raise ImportError(
                "You need to install `nevergrad` to use the BO search strategy."
            )

        # Turn off logging for nevergrad
        import logging

        logger = logging.getLogger("nevergrad")
        logger.propagate = False
        dimensions = ng.p.Instrumentation(
            opt_params=ng.p.Array(
                shape=(self.num_dims,),
                lower=params.search_min,
                upper=params.search_max,
            )
        )
        strategy_cls = BayesOptim(pca=True, n_components=self.n_pca_components)
        self.hyper_optimizer = strategy_cls(parametrization=dimensions)
        initialization = jax.random.uniform(
            rng,
            (self.num_dims,),
            minval=params.search_min,
            maxval=params.search_max,
        )
        state = EvoState(
            mean=initialization,
            best_member=initialization,
        )
        return state

    def ask(
        self, rng: chex.PRNGKey, state: EvoState, params: EvoParams
    ) -> Tuple[chex.Array, EvoState]:
        x = jnp.array(
            [
                self.hyper_optimizer.ask().value[1]["opt_params"]
                for _ in range(self.popsize)
            ]
        )
        x_clipped = jnp.clip(x, params.clip_min, params.clip_max)

        # Reshape parameters into pytrees
        if self.use_param_reshaper:
            x_out = self.param_reshaper.reshape(x_clipped)
        else:
            x_out = x_clipped
        return x_out, state

    def tell(
        self,
        x: Union[chex.Array, chex.ArrayTree],
        fitness: chex.Array,
        state: EvoState,
        params: Optional[EvoParams] = None,
    ):
        # Use default hyperparameters if no other settings provided
        if params is None:
            params = self.default_params

        # Flatten params if using param reshaper for ES update
        if self.use_param_reshaper:
            x = self.param_reshaper.flatten(x)

        # Perform fitness reshaping inside of strategy tell call (if desired)
        fitness_re = self.fitness_shaper.apply(x, fitness)

        # Check if there is a new best member & update trackers
        best_member, best_fitness = get_best_fitness_member(
            x, fitness, state, self.fitness_shaper.maximize
        )

        for i in range(self.popsize):
            x_tell = {"opt_params": np.array(x[i])}
            x_tell_p = self.hyper_optimizer.parametrization.spawn_child(
                new_value=((), x_tell)
            )
            self.hyper_optimizer.tell(x_tell_p, float(fitness_re[i]))
        return state.replace(
            mean=best_member,
            best_member=best_member,
            best_fitness=best_fitness,
            gen_counter=state.gen_counter + 1,
        )
