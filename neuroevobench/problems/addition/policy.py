"""Addition compatible policies."""
from typing import Tuple
import chex
import jax
import jax.numpy as jnp
from flax import linen as nn


class Addition_LSTM(nn.Module):
    """LSTM for Addition regression task."""

    hidden_dims: int
    out_dim: int = 1

    @nn.compact
    def __call__(
        self,
        x: chex.Array,
        carry: chex.ArrayTree,
    ) -> Tuple[Tuple[chex.ArrayTree, chex.ArrayTree], chex.Array]:
        lstm_state, x = nn.LSTMCell()(carry, x)
        x = nn.Dense(features=self.out_dim)(x)
        return lstm_state, x

    def initialize_carry(self) -> Tuple[chex.ArrayTree, chex.ArrayTree]:
        """Initialize hidden state of LSTM."""
        return nn.LSTMCell.initialize_carry(
            jax.random.PRNGKey(0), (), self.hidden_dims
        )


class AdditionPolicy(object):
    """An LSTM for the Addition regression task."""

    def __init__(self, hidden_dims: int):
        self.input_dim = [2]
        self.model = Addition_LSTM(hidden_dims)
        carry = self.model.initialize_carry()
        self.pholder_params = self.model.init(
            jax.random.PRNGKey(0), jnp.zeros(self.input_dim), carry
        )
