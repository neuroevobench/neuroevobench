from typing import Tuple
import chex
import jax
import jax.numpy as jnp
from functools import partial
import jax
import torch
from torchvision import datasets, transforms


class CifarTask(object):
    def __init__(
        self,
        batch_size: int = 128,
        test: bool = False,
    ):
        self.batch_size = batch_size
        self.test = test

        # Setup task-specific input/output shapes and loss fn
        self.action_shape = 10
        self.loss_fn = partial(loss_and_acc, num_classes=10)
        self.data_loader = get_cifar_data(self.batch_size, self.test)

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


def get_cifar_data(batch_size: int, test: bool = False):
    """Get PyTorch Data Loaders for CIFAR-10."""
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
            ),
            transforms.Lambda(lambda x: x.permute(1, 2, 0)),
        ]
    )

    loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(
            root="~/data",
            train=not test,
            download=True,
            transform=transform,
        ),
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
    )
    return loader
