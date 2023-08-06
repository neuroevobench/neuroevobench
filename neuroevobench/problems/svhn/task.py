from typing import Tuple
import chex
import jax
import jax.numpy as jnp
from functools import partial
import jax
import torch

try:
    from torchvision import datasets, transforms
except:
    print("You need to install torchvision for SVHN tasks:")
    print("  pip install torchvision")
    # sys.exit(1)
from torch.utils.data import random_split


class SVHNTask(object):
    def __init__(
        self,
        batch_size: int = 128,
        seed_id: int = 0,  # fix seed for SVHN dataset split
        test: bool = False,
    ):
        self.batch_size = batch_size
        self.seed_id = seed_id
        self.test = test

        # Setup task-specific input/output shapes and loss fn
        self.action_shape = 10
        self.loss_fn = partial(loss_and_acc, num_classes=10)
        torch.manual_seed(seed_id)
        self.data_loader = get_svhn_data(self.batch_size, self.test)

    def set_apply_fn(self, network_apply_fn):
        """Set the network forward function."""
        self.network_apply_fn = network_apply_fn

    def evaluate(self, rng_input: chex.PRNGKey, network_params: chex.ArrayTree):
        """Sample batch from dataloader and evaluate full population."""
        # Sample batch from torch loader and put on device
        X, labels = next(iter(self.data_loader))
        X, labels = jnp.array(X), jnp.array(labels)
        # Map batch evaluation over different policy parameters
        # NOTE: All population members are evaluated on same batch
        loss, acc = jax.jit(jax.vmap(self.rollout, in_axes=(0, None, None)))(
            network_params, X, labels
        )
        return -1 * loss, acc

    def rollout(
        self, network_params: chex.ArrayTree, X: chex.Array, labels: chex.Array
    ):
        """Rollout CNN on a single batch."""
        y_pred = self.network_apply_fn(network_params, X)
        loss, acc = self.loss_fn(y_pred, labels)
        return loss, acc


def loss_and_acc(
    y_pred: chex.Array, y_true: chex.Array, num_classes: int
) -> Tuple[chex.Array, chex.Array]:
    """Compute cross-entropy loss and accuracy."""
    acc = jnp.mean(jnp.argmax(y_pred, axis=-1) == y_true)
    labels = jax.nn.one_hot(y_true, num_classes)
    loss = -jnp.sum(labels * jax.nn.log_softmax(y_pred))
    loss /= labels.shape[0]
    return loss, acc


def normalize(data_tensor):
    """re-scale image values to [-1, 1]"""
    return (data_tensor / 255.0) * 2.0 - 1.0


def get_svhn_data(batch_size: int, test: bool = False):
    """Get PyTorch Data Loaders for SVHN."""
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(lambda x: normalize(x)),
            transforms.Lambda(lambda x: x.permute(1, 2, 0)),
        ]
    )
    dataset = datasets.SVHN(root="~/data", download=True, transform=transform)
    val_size = 12000
    train_size = len(dataset) - val_size

    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    if test:
        return torch.utils.data.DataLoader(
            val_ds, batch_size, num_workers=4, pin_memory=False
        )
    else:
        return torch.utils.data.DataLoader(
            train_ds, batch_size, shuffle=True, num_workers=4, pin_memory=False
        )
