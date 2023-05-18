from typing import Tuple
import chex
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.struct import dataclass
from evojax.task.base import VectorizedTask
from evojax.task.base import TaskState

import tensorflow as tf
import tensorflow_datasets as tfds
from ..utils import BatchLoader


@jax.vmap
def kl_divergence(mean, logvar):
    return -0.5 * jnp.sum(1 + logvar - jnp.square(mean) - jnp.exp(logvar))


@jax.vmap
def binary_cross_entropy_with_logits(logits, labels):
    logits = nn.log_sigmoid(logits)
    return -jnp.sum(
        labels * logits + (1.0 - labels) * jnp.log(-jnp.expm1(logits))
    )


def loss_fn(recon_x, mean, logvar, batch):
    bce_loss = binary_cross_entropy_with_logits(recon_x, batch).mean()
    kld_loss = kl_divergence(mean, logvar).mean()
    loss = bce_loss + kld_loss
    return loss


@dataclass
class MNISTState(TaskState):
    obs: chex.Array
    rng: chex.PRNGKey


class MNIST_Generate_Task(VectorizedTask):
    def __init__(
        self,
        env_name: str = "mnist",
        batch_size: int = 1024,
        test: bool = False,
    ):
        self.max_steps = 1
        self.obs_shape = tuple([784])
        data = get_binarized_mnist_data(test)
        dataloader = BatchLoader(data, None, batch_size)

        def reset_fn(key: chex.PRNGKey) -> MNISTState:
            key, key_gen = jax.random.split(key)
            if test:
                # Use entire test set for evaluation
                batch_data = data
            else:
                # Sample batch from dataloader
                batch_data = dataloader.sample(key)
            return MNISTState(obs=batch_data, rng=key_gen)

        self._reset_fn = jax.jit(jax.vmap(reset_fn))

        def step_fn(
            state: MNISTState, action: chex.Array
        ) -> Tuple[MNISTState, chex.Array, chex.Array]:
            recon_x, mean, logvar = action[0], action[1], action[2]
            reward = loss_fn(recon_x, mean, logvar, state.obs)
            return state, -1 * reward, jnp.ones(())

        self._step_fn = jax.jit(jax.vmap(step_fn))

    def reset(self, key: chex.Array) -> MNISTState:
        return self._reset_fn(key)

    def step(
        self, state: MNISTState, action: chex.Array
    ) -> Tuple[MNISTState, chex.Array, chex.Array]:
        return self._step_fn(state, action)


def prepare_image(x):
    x = tf.cast(x["image"], tf.float32)
    x = tf.reshape(x, (-1,))
    return x


def get_binarized_mnist_data(test: bool = False):
    ds_builder = tfds.builder("binarized_mnist")
    ds_builder.download_and_prepare()
    if not test:
        train_ds = ds_builder.as_dataset(split=tfds.Split.TRAIN)
        train_ds = train_ds.map(prepare_image).batch(50000)
        train_ds = jnp.array(list(train_ds)[0])
        train_ds = jax.device_put(train_ds)
        return train_ds
    else:
        test_ds = ds_builder.as_dataset(split=tfds.Split.TEST)
        test_ds = test_ds.map(prepare_image).batch(10000)
        test_ds = jnp.array(list(test_ds)[0])
        test_ds = jax.device_put(test_ds)
        return test_ds
