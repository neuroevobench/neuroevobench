from typing import Tuple
import jax
import jax.numpy as jnp
import chex
from ..utils import BatchLoader


class AdditionTask(object):
    def __init__(
        self,
        batch_size: int = 128,
        seq_length: int = 150,  # Sequence length in addition task
        seed_id: int = 0,
        test: bool = False,
    ):
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.seed_id = seed_id
        self.test = test

        # Setup task-specific input/output shapes and loss fn
        self.action_shape = 1
        self.loss_fn = loss_and_mae

        data = get_addition_data(self.seed_id, self.seq_length, self.test)
        self.dataloader = BatchLoader(*data, batch_size=self.batch_size)
        self.num_rnn_steps = self.dataloader.data_shape[1]

    def set_apply_fn(self, network_apply_fn, init_carry_fn):
        """Set the network forward function."""
        self.network_apply_fn = network_apply_fn
        self.init_carry_fn = init_carry_fn
        # Map batch evaluation over different policy parameters
        # NOTE: All population members are evaluated on same batch
        self.evaluate = jax.jit(jax.vmap(self.rollout_rnn, in_axes=(None, 0)))

    def rollout_rnn(
        self, rng_input: chex.PRNGKey, network_params: chex.ArrayTree
    ) -> Tuple[float, float]:
        """Evaluate a network on a supervised learning task."""
        X, y = self.dataloader.sample(rng_input)
        # Map over sequence batch dimension
        y_pred = jax.vmap(self.rollout_single, in_axes=(None, 0))(
            network_params, X
        )
        loss, perf = self.loss_fn(y_pred, y)
        # Return negative loss to maximize!
        return -1 * loss, perf

    def rollout_single(
        self, network_params: chex.ArrayTree, X_single: chex.Array
    ):
        """Rollout RNN on a single sequence."""
        # Reset the network hidden state at beginning of sequence
        hidden = self.init_carry_fn()

        def rnn_step(state_input, unused):
            """lax.scan compatible step transition in jax env."""
            hidden, t = state_input
            hidden, pred = self.network_apply_fn(
                network_params,
                X_single[t],
                hidden,
            )
            carry = [hidden, t + 1]
            return carry, pred

        # Scan over image length (784)/sequence
        _, scan_out = jax.lax.scan(
            rnn_step, [hidden, 0], (), self.num_rnn_steps
        )
        y_pred = scan_out[-1]
        return y_pred

    @property
    def input_shape(self) -> Tuple[int]:
        """Get the shape of the observation."""
        return self.dataloader.data_shape


def loss_and_mae(
    y_pred: chex.Array, y_true: chex.Array
) -> Tuple[chex.Array, chex.Array]:
    """Compute mean squared error loss and mean absolute error."""
    loss = jnp.mean((y_pred.squeeze() - y_true) ** 2)
    mae = jnp.mean(jnp.abs(y_pred.squeeze() - y_true))
    return loss, mae


def get_addition_data(seed_id: int = 0, T: int = 150, test: bool = False):
    """
    Sample a mask, [0, 1] samples and sum of targets for len T.
    Reference:  Martens & Sutskever. ICML, 2011.
    """
    rng = jax.random.PRNGKey(seed_id)
    bs = 100000 if test else 10000

    def get_single_addition(rng, T):
        rng_numb, rng_mask = jax.random.split(rng)
        numbers = jax.random.uniform(rng_numb, (T,), minval=0, maxval=1)
        mask_ids = jax.random.choice(
            rng_mask, jnp.arange(T), (2,), replace=False
        )
        mask = jnp.zeros(T).at[mask_ids].set(1)
        target = jnp.sum(mask * numbers)
        return jnp.stack([numbers, mask], axis=1), target

    batch_seq_gen = jax.vmap(get_single_addition, in_axes=(0, None))
    data, target = batch_seq_gen(jax.random.split(rng, bs), T)
    return jnp.array(data), jnp.array(target)
