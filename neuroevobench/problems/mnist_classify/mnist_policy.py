"""EvoJAX MNIST compatible policies."""
from typing import Sequence, Tuple
import jax
import jax.numpy as jnp
from flax import linen as nn
from evojax.policy.base import PolicyNetwork
from evojax.policy.base import PolicyState
from evojax.util import get_params_format_fn
from evojax.task.base import TaskState


def default_mlp_init(scale=0.05):
    return nn.initializers.uniform(scale)


class MNIST_CNN(nn.Module):
    """CNN for MNIST."""

    hidden_dims: Sequence[int]
    out_dim: int = 10

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(
            features=8,
            kernel_size=(5, 5),
            padding="SAME",
            bias_init=default_mlp_init(),
        )(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Conv(
            features=16,
            kernel_size=(5, 5),
            padding="SAME",
            bias_init=default_mlp_init(),
        )(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))  # flatten
        for hidden_dim in self.hidden_dims:
            x = nn.relu(
                nn.Dense(
                    features=hidden_dim,
                    bias_init=default_mlp_init(),
                )(x)
            )
        x = nn.Dense(features=self.out_dim)(x)
        x = nn.log_softmax(x)
        return x


class MNISTPolicy(PolicyNetwork):
    """A convolutional neural network for the MNIST classification task."""

    def __init__(
        self,
        hidden_dims: Sequence[int],
    ):
        self.input_dim = [1, 28, 28, 1]
        self.model = MNIST_CNN(hidden_dims)
        self.params = self.model.init(
            jax.random.PRNGKey(0), jnp.zeros(self.input_dim)
        )
        self.num_params, format_params_fn = get_params_format_fn(self.params)
        self._format_params_fn = jax.vmap(format_params_fn)
        self._forward_fn = jax.vmap(self.model.apply)

    def get_actions(
        self, t_states: TaskState, params: jnp.ndarray, p_states: PolicyState
    ) -> Tuple[jnp.ndarray, PolicyState]:
        params = self._format_params_fn(params)
        return self._forward_fn(params, t_states.obs), p_states
