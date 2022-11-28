"""EvoJAX compatible policies."""
from typing import Sequence, Tuple
import chex
import jax
import jax.numpy as jnp
from flax import linen as nn
from evojax.policy.base import PolicyNetwork
from evojax.policy.base import PolicyState
from evojax.util import get_params_format_fn
from evojax.task.base import TaskState
from evojax.policy.mlp import PolicyNetwork, MLP


class MLPPolicy(PolicyNetwork):
    """A general purpose multi-layer perceptron model."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        output_dim: int,
        output_act_fn: str = "tanh",
    ):
        model = MLP(
            feat_dims=hidden_dims, out_dim=output_dim, out_fn=output_act_fn
        )
        self.params = model.init(
            jax.random.PRNGKey(0), jnp.ones([1, input_dim])
        )
        self.num_params, format_params_fn = get_params_format_fn(self.params)
        self._format_params_fn = jax.vmap(format_params_fn)
        self._forward_fn = jax.vmap(model.apply)

    def get_actions(
        self, t_states: TaskState, params: jnp.ndarray, p_states: PolicyState
    ) -> Tuple[jnp.ndarray, PolicyState]:
        params = self._format_params_fn(params)
        return self._forward_fn(params, t_states.obs), p_states


class MinAtarCNN(nn.Module):
    """A general purpose conv net model."""

    hidden_dims: Sequence[int]
    out_dim: int

    @nn.compact
    def __call__(self, x: chex.Array) -> chex.Array:
        x = nn.Conv(features=16, kernel_size=(3, 3), padding="SAME", strides=1)(
            x
        )
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))
        for hidden_dim in self.hidden_dims:
            x = nn.relu(nn.Dense(features=hidden_dim)(x))
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
        model = MinAtarCNN(hidden_dims=hidden_dims, out_dim=output_dim)
        self.params = model.init(
            jax.random.PRNGKey(0), jnp.ones([1, *input_dim])
        )
        self.num_params, format_params_fn = get_params_format_fn(self.params)
        self._format_params_fn = jax.vmap(format_params_fn)
        self._forward_fn = jax.vmap(model.apply)

    def get_actions(
        self, t_states: TaskState, params: chex.Array, p_states: PolicyState
    ) -> Tuple[chex.Array, PolicyState]:
        params = self._format_params_fn(params)
        obs = jnp.expand_dims(t_states.obs, axis=1)
        activations = self._forward_fn(params, obs)
        action = jnp.argmax(activations, axis=2).squeeze()
        return action, p_states


class MNIST_CNN(nn.Module):
    """CNN for MNIST."""

    hidden_dims: Sequence[int]
    out_dim: int = 10

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=8, kernel_size=(5, 5), padding="SAME")(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Conv(features=16, kernel_size=(5, 5), padding="SAME")(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))  # flatten
        for hidden_dim in self.hidden_dims:
            x = nn.relu(nn.Dense(features=hidden_dim)(x))
        x = nn.Dense(features=self.out_dim)(x)
        x = nn.log_softmax(x)
        return x


class MNISTPolicy(PolicyNetwork):
    """A convolutional neural network for the MNIST classification task."""

    def __init__(
        self,
        hidden_dims: Sequence[int],
    ):
        model = MNIST_CNN(hidden_dims)
        self.params = model.init(
            jax.random.PRNGKey(0), jnp.zeros([1, 28, 28, 1])
        )
        self.num_params, format_params_fn = get_params_format_fn(self.params)
        self._format_params_fn = jax.vmap(format_params_fn)
        self._forward_fn = jax.vmap(model.apply)

    def get_actions(
        self, t_states: TaskState, params: jnp.ndarray, p_states: PolicyState
    ) -> Tuple[jnp.ndarray, PolicyState]:
        params = self._format_params_fn(params)
        return self._forward_fn(params, t_states.obs), p_states
