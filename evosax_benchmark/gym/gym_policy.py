"""EvoJAX Gym compatible policies."""
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
