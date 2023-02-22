"""EvoJAX MinAtar compatible policies."""
from typing import Sequence, Tuple
import chex
import jax
import jax.numpy as jnp
from flax import linen as nn
from evojax.policy.base import PolicyNetwork
from evojax.policy.base import PolicyState
from evojax.util import get_params_format_fn
from evojax.task.base import TaskState


def default_mlp_init(scale=0.05):
    return nn.initializers.uniform(scale)


class MinAtarCNN(nn.Module):
    """A general purpose conv net model."""

    hidden_dims: Sequence[int]
    out_dim: int

    @nn.compact
    def __call__(self, x: chex.Array) -> chex.Array:
        x = nn.Conv(
            features=16,
            kernel_size=(3, 3),
            padding="SAME",
            strides=1,
            bias_init=default_mlp_init(),
        )(x)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))
        for hidden_dim in self.hidden_dims:
            x = nn.relu(
                nn.Dense(
                    features=hidden_dim,
                    bias_init=default_mlp_init(),
                )(x)
            )
        x = nn.Dense(features=self.out_dim)(x)
        return x


class MinAtarPolicy(PolicyNetwork):
    """Deterministic CNN policy - greedy action selection."""

    def __init__(
        self,
        input_dim: Sequence[int],
        hidden_dims: Sequence[int],
        output_dim: int,
    ):
        self.input_dim = [1, *input_dim]
        self.model = MinAtarCNN(hidden_dims=hidden_dims, out_dim=output_dim)
        self.params = self.model.init(
            jax.random.PRNGKey(0), jnp.ones(self.input_dim)
        )
        self.num_params, format_params_fn = get_params_format_fn(self.params)
        self._format_params_fn = jax.vmap(format_params_fn)
        self._forward_fn = jax.vmap(self.model.apply)

    def get_actions(
        self, t_states: TaskState, params: chex.Array, p_states: PolicyState
    ) -> Tuple[chex.Array, PolicyState]:
        params = self._format_params_fn(params)
        obs = jnp.expand_dims(t_states.obs, axis=1)
        activations = self._forward_fn(params, obs)
        action = jnp.argmax(activations, axis=2).squeeze()
        return action, p_states
