"""Adapted from github.com/alonfnt/bayex/ BayesOpt package."""
from functools import partial
from typing import Any, Callable, NamedTuple, Union

import jax.numpy as jnp
from jax import jacrev, jit, lax, random, vmap
from .bo_gp import DataTypes, GParameters, round_integers

Array = Any


class OptimizerParameters(NamedTuple):
    """
    Object holding the results of the optimization.
    """

    target: Union[Array, float]
    params: Array
    f: Callable
    params_all: Array
    target_all: Array


def jacobian(f: Callable) -> Callable:
    return jit(jacrev(f))


def replace_nan_values(arr: Array) -> Array:
    """
    Replaces the NaN values (if any) in arr with 0.

    Parameters:
    -----------
    arr: The array where NaN is removed from.

    Returns:
    --------
    The array with all the NaN elements replaced with 0.
    """
    # todo(alonfnt): Find a more robust solution.
    return jnp.where(jnp.isnan(arr), 0, arr)


@partial(jit, static_argnums=(6, 7))
def suggest_next(
    key: Array,
    params: GParameters,
    x: Array,
    y: Array,
    bounds: Array,
    dtypes: DataTypes,
    acq: Callable,
    popsize: int = 1,
    n_seed: int = 1000,
    lr: float = 0.1,
    n_epochs: int = 150,
) -> Array:
    """
    Suggests the new point to sample by optimizing the acquisition function.

    Parameters:
    -----------
    key: The pseudo-random generator key used for jax random functions.
    params: Hyperparameters of the Gaussian Process Regressor.
    x: Sampled points.
    y: Sampled targets.
    bounds: Array of (2, dim) shape with the lower and upper bounds of the
            variables.y_max: The current maximum value of the target values Y.
    dtypes: The type of non-real variables in the target function.
    n_seed (optional): the number of points to probe and minimize until
            finding the one that maximizes the acquisition functions.
    lr (optional): The step size of the gradient descent.
    n_epochs (optional): The number of steps done on the descent to minimize
            the seeds.


    Returns:
    --------
    A tuple with the parameters that maximize the acquisition function and a
    jax PRGKey to be used in the next sampling.
    """

    dim = bounds.shape[0]

    key_d, key_r = random.split(key)
    domain = random.uniform(
        key_d, shape=(n_seed, dim), minval=bounds[:, 0], maxval=bounds[:, 1]
    )

    _acq = partial(acq, params=params, x=x, y=y, dtypes=dtypes)

    J = jacobian(lambda x: _acq(x.reshape(-1, dim)).reshape())
    HS = vmap(lambda x: x + lr * J(x))

    domain = lax.fori_loop(0, n_epochs, lambda _, d: HS(d), domain)
    domain = jnp.clip(
        domain.reshape(-1, dim), a_min=bounds[:, 0], a_max=bounds[:, 1]
    )
    domain = replace_nan_values(domain)
    domain = round_integers(domain, dtypes)

    ys = _acq(domain)
    # Pick the first point based on optimized acquisition fn
    best_X = domain[ys.argmax()].reshape(1, dim)
    # Add remaining half based on random domain samples
    random_X = random.uniform(
        key_r,
        shape=(popsize - 1, dim),
        minval=bounds[:, 0],
        maxval=bounds[:, 1],
    )
    next_X = jnp.concatenate([best_X, random_X], axis=0)
    return next_X
