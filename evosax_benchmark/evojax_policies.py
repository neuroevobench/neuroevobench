"""EvoJAX compatible policies."""
from typing import Sequence, Tuple, Optional
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


class BraxMLP(nn.Module):
    feat_dims: Sequence[int]
    out_dim: int
    out_fn: str

    @nn.compact
    def __call__(self, x):
        for hidden_dim in self.feat_dims:
            x = nn.tanh(
                nn.Dense(
                    hidden_dim,
                    bias_init=default_mlp_init(),
                )(x)
            )
        x = nn.Dense(
            self.out_dim,
            bias_init=default_mlp_init(),
        )(x)
        if self.out_fn == "tanh":
            x = nn.tanh(x)
        elif self.out_fn == "softmax":
            x = nn.softmax(x, axis=-1)
        else:
            raise ValueError(
                "Unsupported output activation: {}".format(self.out_fn)
            )
        return x


class BraxPolicy(PolicyNetwork):
    """A general purpose multi-layer perceptron model."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        output_dim: int,
        output_act_fn: str = "tanh",
    ):
        self.input_dim = [1, input_dim]
        self.model = BraxMLP(
            feat_dims=hidden_dims, out_dim=output_dim, out_fn=output_act_fn
        )
        self.params = self.model.init(
            jax.random.PRNGKey(0), jnp.ones(self.input_dim)
        )
        self.num_params, format_params_fn = get_params_format_fn(self.params)
        self._format_params_fn = jax.vmap(format_params_fn)
        self._forward_fn = jax.vmap(self.model.apply)

    def get_actions(
        self, t_states: TaskState, params: jnp.ndarray, p_states: PolicyState
    ) -> Tuple[jnp.ndarray, PolicyState]:
        params = self._format_params_fn(params)
        return self._forward_fn(params, t_states.obs), p_states


class GymMLP(nn.Module):
    feat_dims: Sequence[int]
    out_dim: int
    out_fn: str

    @nn.compact
    def __call__(self, x):
        for hidden_dim in self.feat_dims:
            x = nn.tanh(
                nn.Dense(
                    hidden_dim,
                    bias_init=default_mlp_init(),
                )(x)
            )
        x = nn.Dense(
            self.out_dim,
            bias_init=default_mlp_init(),
        )(x)
        if self.out_fn == "tanh":
            x = nn.tanh(x)
        elif self.out_fn == "softmax":
            x = nn.softmax(x, axis=-1)
        return x


class GymPolicy(PolicyNetwork):
    """A general purpose multi-layer perceptron model."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        output_dim: int,
        output_act_fn: Optional[str] = None,
    ):
        self.input_dim = [1, *input_dim]
        self.model = GymMLP(
            feat_dims=hidden_dims, out_dim=output_dim, out_fn=None
        )
        self.params = self.model.init(
            jax.random.PRNGKey(0), jnp.ones(self.input_dim)
        )
        self.output_act_fn = output_act_fn
        self.num_params, format_params_fn = get_params_format_fn(self.params)
        self._format_params_fn = jax.vmap(format_params_fn)
        self._forward_fn = jax.vmap(self.model.apply)

    def get_actions(
        self, t_states: TaskState, params: jnp.ndarray, p_states: PolicyState
    ) -> Tuple[jnp.ndarray, PolicyState]:
        params = self._format_params_fn(params)
        obs = jnp.expand_dims(t_states.obs, axis=1)
        activations = self._forward_fn(params, obs)
        if self.output_act_fn == "argmax":
            action = jnp.argmax(activations, axis=2).squeeze()
        else:
            # Pendulum reshape
            action = activations.squeeze()
        return action, p_states


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
