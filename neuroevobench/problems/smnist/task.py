from typing import Tuple
import jax
import jax.numpy as jnp
import chex
from functools import partial
import torch

try:
    from torchvision import datasets, transforms
except:
    print("You need to install torchvision for SMNIST tasks:")
    print("  pip install torchvision")
    # sys.exit(1)
from ..utils import BatchLoader


class SMNISTTask(object):
    def __init__(
        self,
        batch_size: int = 128,
        permute_seq: bool = False,  # Permuted S-MNIST task option
        seed_id: int = 0,
        test: bool = False,
    ):
        self.batch_size = batch_size
        self.permute_seq = permute_seq  # Whether to permute timeseries
        self.seed_id = seed_id
        self.test = test

        # Setup task-specific input/output shapes and loss fn
        self.action_shape = 10
        self.seq_length = 784
        self.loss_fn = partial(loss_and_acc, num_classes=10)

        data = get_smnist_data(self.seed_id, self.permute_seq, self.test)
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
                network_params, X_single[t], hidden
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


def loss_and_acc(
    y_pred: chex.Array, y_true: chex.Array, num_classes: int
) -> Tuple[chex.Array, chex.Array]:
    """Compute cross-entropy loss and accuracy."""
    acc = jnp.mean(jnp.argmax(y_pred, axis=-1) == y_true)
    labels = jax.nn.one_hot(y_true, num_classes)
    loss = -jnp.sum(labels * jax.nn.log_softmax(y_pred))
    loss /= labels.shape[0]
    return loss, acc


def get_smnist_data(
    seed_id: int = 0, permute_seq: bool = False, test: bool = False
):
    """Load MNIST data from torch, flatten and permute if desired."""
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.Lambda(lambda x: torch.flatten(x)),
            transforms.Lambda(lambda x: torch.unsqueeze(x, -1)),
        ]
    )
    bs = 10000 if test else 60000
    loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "~/data", download=True, train=not test, transform=transform
        ),
        batch_size=bs,
        shuffle=False,
    )

    for _, (data, target) in enumerate(loader):
        break
    data, target = jnp.array(data), jnp.array(target)

    # Permute the sequence of the pixels if desired.
    if permute_seq:  # bs, T - fix permutation by seed
        rng = jax.random.PRNGKey(seed_id)
        idx = jnp.arange(784)
        idx_perm = jax.random.permutation(rng, idx)
        data = data.at[:].set(data[:, idx_perm])
    return data, target
