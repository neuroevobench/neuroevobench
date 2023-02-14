from typing import Sequence
import jax
import jax.numpy as jnp
from flax import linen as nn
from evojax.util import get_params_format_fn


class AtariCNN(nn.Module):
    """DQN-style network."""

    out_dim: int
    hidden_dims: Sequence[int]

    @nn.compact
    def __call__(self, x):
        x = nn.relu(nn.Conv(features=32, kernel_size=(8, 8), strides=(4, 4))(x))
        x = nn.relu(nn.Conv(features=64, kernel_size=(4, 4), strides=(2, 2))(x))
        x = nn.relu(nn.Conv(features=64, kernel_size=(3, 3), strides=(1, 1))(x))
        x = x.reshape((x.shape[0], -1))
        for hidden_dim in self.hidden_dims:
            x = nn.relu(nn.Dense(hidden_dim)(x))
        x = nn.Dense(features=self.out_dim)(x)
        return x


class AtariPolicy(object):
    def __init__(
        self, hidden_dims: Sequence[int] = [64], num_actions: int = 10
    ):
        self.input_dim = [1, 84, 84, 4]
        self.model = AtariCNN(num_actions, hidden_dims)
        self.params = self.model.init(
            jax.random.PRNGKey(0), jnp.zeros(self.input_dim)
        )
        self.num_params, self.format_params_fn = get_params_format_fn(
            self.params
        )
        self._format_params_fn = jax.vmap(self.format_params_fn)
        self._forward_fn = jax.vmap(self.model.apply)

    def get_actions(self, params: jnp.ndarray, obs: jnp.ndarray) -> jnp.ndarray:
        params = self._format_params_fn(params)
        activations = jax.vmap(self.model.apply, in_axes=(0, 0))(params, obs)
        action = jnp.argmax(activations, axis=-1).squeeze()
        return action

    def get_actions_single(
        self, params: jnp.ndarray, obs: jnp.ndarray
    ) -> jnp.ndarray:
        """Actions for single network, multiple states/obs.
        Obs shape: (batch, 84, 84, stacked_frames)
        """
        params = self.format_params_fn(params)
        activations = self.model.apply(params, obs)
        action = jnp.argmax(activations, axis=1).squeeze()
        return action
