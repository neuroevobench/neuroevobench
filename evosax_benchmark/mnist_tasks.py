from typing import Tuple
import chex
import jax
import jax.numpy as jnp
from flax.struct import dataclass
from evojax.task.base import VectorizedTask
from evojax.task.base import TaskState


@dataclass
class MNISTState(TaskState):
    obs: chex.Array
    labels: chex.Array


def sample_batch(
    key: chex.PRNGKey, data: chex.Array, labels: chex.Array, batch_size: int
):
    idx = jax.random.choice(
        key, a=data.shape[0], shape=(batch_size,), replace=False
    )
    data_sub = jnp.take(data, indices=idx, axis=0)
    labels_sub = jnp.take(labels, indices=idx, axis=0)
    return data_sub, labels_sub


def loss(prediction: chex.Array, target: chex.Array) -> chex.Array:
    target = jax.nn.one_hot(target, 10)
    return -jnp.mean(jnp.sum(prediction * target, axis=1))


def accuracy(prediction: chex.Array, target: chex.Array) -> chex.Array:
    predicted_class = jnp.argmax(prediction, axis=1)
    return jnp.mean(predicted_class == target)


class MNISTTask(VectorizedTask):
    def __init__(
        self,
        env_name: str = "mnist",
        batch_size: int = 1024,
        test: bool = False,
    ):
        import tensorflow_datasets as tfds

        self.max_steps = 1
        self.obs_shape = tuple([28, 28, 1])
        self.act_shape = tuple([10])
        image, labels = tfds.as_numpy(
            tfds.load(
                env_name,
                split="test" if test else "train",
                batch_size=-1,
                as_supervised=True,
            )
        )
        data = image / 255.0

        def reset_fn(key: chex.PRNGKey) -> MNISTState:
            if test:
                batch_data, batch_labels = data, labels
            else:
                batch_data, batch_labels = sample_batch(
                    key, data, labels, batch_size
                )
            return MNISTState(obs=batch_data, labels=batch_labels)

        self._reset_fn = jax.jit(jax.vmap(reset_fn))

        def step_fn(
            state: MNISTState, action: chex.Array
        ) -> Tuple[MNISTState, chex.Array, chex.Array]:
            if test:
                reward = accuracy(action, state.labels)
            else:
                reward = -loss(action, state.labels)
            return state, reward, jnp.ones(())

        self._step_fn = jax.jit(jax.vmap(step_fn))

    def reset(self, key: chex.Array) -> MNISTState:
        return self._reset_fn(key)

    def step(
        self, state: MNISTState, action: chex.Array
    ) -> Tuple[MNISTState, chex.Array, chex.Array]:
        return self._step_fn(state, action)
