"""Sequential MNIST compatible policies."""
from typing import Tuple
import chex
import jax
import jax.numpy as jnp
from flax import linen as nn


class SMNIST_LSTM(nn.Module):
    """LSTM for Sequential MNIST."""

    hidden_dims: int
    out_dim: int = 10

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


class SMNISTPolicy(object):
    """An LSTM for the Sequential MNIST classification task."""

    def __init__(self, hidden_dims: int):
        self.input_dim = [1]
        self.model = SMNIST_LSTM(hidden_dims)
        carry = self.model.initialize_carry()
        self.pholder_params = self.model.init(
            jax.random.PRNGKey(0), jnp.zeros(self.input_dim), carry
        )
