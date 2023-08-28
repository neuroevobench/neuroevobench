"""Bayesian Optimization Wrapper to benchmark agains - not JAX compatible."""
from typing import Any, Optional, Union, Tuple
import chex
import jax
import jax.numpy as jnp
from flax import struct
from evosax.core import ParameterReshaper, FitnessShaper
from evosax.utils import get_best_fitness_member
from .bo_utils import (
    ACQ,
    select_acq,
    DataTypes,
    GParameters,
    train,
    suggest_next,
)

Array = Any
dtypes = DataTypes(integers=[])


@struct.dataclass
class EvoState:
    gp_params: chex.ArrayTree
    momentum: chex.ArrayTree
    scales: chex.ArrayTree
    X: chex.Array
    Y: chex.Array
    bounds: chex.Array
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


class BayesOptJAX(object):
    def __init__(
        self,
        popsize: int,
        num_dims: Optional[int] = None,
        pholder_params: Optional[Union[chex.ArrayTree, chex.Array]] = None,
        param_range: float = 2.0,
        acq_fn_name: str = "UCB",
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
        self.param_range = param_range

        # Setup optional fitness shaper
        self.fitness_shaper = FitnessShaper(**fitness_kwargs)

        # Setup acquisiton function for BO
        if acq_fn_name == "EI":
            acq = ACQ.EI
        elif acq_fn_name == "POI":
            acq = ACQ.POI
        elif acq_fn_name == "UCB":
            acq = ACQ.UCB
        elif acq_fn_name == "LCB":
            acq = ACQ.LCB
        else:
            raise ValueError("Specify correct acquisition fn.")
        self.acq_fn = select_acq(acq, acq_params={})

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
        # No integer-valued variables to optimize
        _sorted_constrains = {
            k: (params.search_min, params.search_max)
            for k in range(self.num_dims)
        }
        bounds = jnp.asarray(list(_sorted_constrains.values()))
        rng, rng_init = jax.random.split(rng)
        initialization = jax.random.uniform(
            rng_init,
            (self.num_dims,),
            minval=params.search_min,
            maxval=params.search_max,
        )

        # GP Specific initializations
        X = jax.random.uniform(
            rng,
            (self.popsize, self.num_dims),
            minval=params.search_min,
            maxval=params.search_max,
        )
        Y = jnp.zeros(self.popsize)
        gp_params = GParameters(
            noise=jnp.zeros((1, 1)) - 5.0,
            amplitude=jnp.zeros((1, 1)),
            lengthscale=jnp.zeros((1, self.num_dims)),
        )
        momentum = jax.tree_map(lambda x: x * 0, gp_params)
        scales = jax.tree_map(lambda x: x * 0 + 1, gp_params)

        state = EvoState(
            gp_params=gp_params,
            momentum=momentum,
            scales=scales,
            X=X,
            Y=Y,
            bounds=bounds,
            mean=initialization,
            best_member=initialization,
        )
        return state

    def ask(
        self, rng: chex.PRNGKey, state: EvoState, params: EvoParams
    ) -> Tuple[chex.Array, EvoState]:
        if state.gen_counter == 0:
            x = state.X
        else:
            x = suggest_next(
                rng,
                state.gp_params,
                state.X,
                state.Y,
                state.bounds,
                dtypes,
                self.acq_fn,
                self.popsize,
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

        # TODO(RobertTLange): Train the GP - always maximize!
        Y = -1 * fitness_re
        gp_params, momentum, scales = train(
            x, Y, state.gp_params, state.momentum, state.scales, dtypes
        )

        return state.replace(
            X=x,
            Y=Y,
            gp_params=gp_params,
            momentum=momentum,
            scales=scales,
            mean=best_member,
            best_member=best_member,
            best_fitness=best_fitness,
            gen_counter=state.gen_counter + 1,
        )
