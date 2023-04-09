"""EvoJAX Brax compatible policies."""
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


class BraxMLP(nn.Module):
    feat_dims: Sequence[int]
    out_dim: int
    hidden_fn: str
    out_fn: str

    @nn.compact
    def __call__(self, x):
        for hidden_dim in self.feat_dims:
            x = nn.Dense(
                hidden_dim,
                bias_init=default_mlp_init(),
            )(x)
            if self.hidden_fn == "tanh":
                x = nn.tanh(x)
            elif self.hidden_fn == "relu":
                x = nn.relu(x)
            elif self.hidden_fn == "swish":
                x = nn.swish(x)
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
        hidden_act_fn: str = "tanh",
        output_act_fn: str = "tanh",
    ):
        self.input_dim = [1, input_dim]
        self.model = BraxMLP(
            feat_dims=hidden_dims,
            out_dim=output_dim,
            hidden_fn=hidden_act_fn,
            out_fn=output_act_fn,
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
