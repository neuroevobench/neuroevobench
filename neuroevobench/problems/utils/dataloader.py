from typing import Tuple
import chex
import jax
import jax.numpy as jnp


class BatchLoader:
    def __init__(
        self,
        X: chex.Array,
        y: chex.Array,
        batch_size: int,
    ):
        self.X = X
        self.y = y
        self.data_shape = self.X.shape[1:][::-1]
        self.num_train_samples = X.shape[0]
        self.batch_size = batch_size

    def sample(self, key: chex.PRNGKey) -> Tuple[chex.Array, chex.Array]:
        """Sample a single batch of X, y data."""
        sample_idx = jax.random.choice(
            key,
            jnp.arange(self.num_train_samples),
            (self.batch_size,),
            replace=False,
        )
        return (
            jnp.take(self.X, sample_idx, axis=0),
            jnp.take(self.y, sample_idx, axis=0),
        )
