"""Utilities to port gymnax/mnist env to evoJAX tasks."""
from typing import Tuple, Sequence, Optional
import chex
import jax
import jax.numpy as jnp
from flax.struct import dataclass

import gymnax
from gymnax import EnvState
from evojax.task.base import VectorizedTask
from evojax.task.base import TaskState
from evojax.task.brax_task import BraxTask
from .evojax_policies import MLPPolicy, MNISTPolicy, MinAtarPolicy


def get_evojax_task(
    env_name: str,
    hidden_layers: int = 4,
    hidden_dims: int = 32,
    max_steps: Optional[int] = 1000,
    batch_size: Optional[int] = 1024,
):
    """Return EvoJAX conform task wrapper."""
    # Create list of hidden dims per dense layer
    hidden_dims_evo = hidden_layers * [hidden_dims]
    if env_name in [
        "ant",
        "fetch",
        "grasp",
        "halfcheetah",
        "hopper",
        "humanoid",
        "reacher",
        "ur5e",
        "walker2d",
    ]:
        train_task, test_task, policy = get_brax_task(
            env_name, hidden_dims_evo, max_steps
        )
    elif env_name in [
        "Asterix-MinAtar",
        "Breakout-MinAtar",
        "Freeway-MinAtar",
        "SpaceInvaders-MinAtar",
        "Seaquest-MinAtar",
    ]:
        train_task, test_task, policy = get_minatar_task(
            env_name, hidden_dims_evo, max_steps
        )
    elif env_name in ["mnist", "fashion_mnist", "kmnist", "mnist_corrupted"]:
        train_task, test_task, policy = get_mnist_task(
            env_name, hidden_dims_evo, batch_size
        )
    return train_task, test_task, policy


def get_brax_task(
    env_name: str = "ant",
    hidden_dims: Sequence[int] = [32, 32, 32, 32],
    max_steps: int = 1000,
):
    train_task = BraxTask(env_name, max_steps=max_steps, test=False)
    test_task = BraxTask(env_name, max_steps=max_steps, test=True)
    policy = MLPPolicy(
        input_dim=train_task.obs_shape[0],
        output_dim=train_task.act_shape[0],
        hidden_dims=hidden_dims,
    )
    return train_task, test_task, policy


def get_minatar_task(
    env_name: str = "SpaceInvaders-MinAtar",
    hidden_dims: Sequence[int] = [32],
    max_steps: int = 500,
):
    train_task = GymnaxTask(env_name, max_steps=max_steps, test=False)
    test_task = GymnaxTask(env_name, max_steps=max_steps, test=True)
    policy = MinAtarPolicy(
        input_dim=train_task.obs_shape,
        output_dim=train_task.num_actions,
        hidden_dims=hidden_dims,
    )
    return train_task, test_task, policy


def get_mnist_task(
    env_name: str = "mnist",
    hidden_dims: Sequence[int] = [],
    batch_size: int = 1024,
):
    # mnist, fashion_mnist, kmnist, mnist_corrupted
    train_task = MNISTTask(env_name, batch_size=batch_size, test=False)
    test_task = MNISTTask(env_name, batch_size=None, test=True)
    policy = MNISTPolicy(hidden_dims)
    return train_task, test_task, policy


@dataclass
class GymState(TaskState):
    state: EnvState
    obs: chex.Array
    rng: chex.PRNGKey


class GymnaxTask(VectorizedTask):
    """Task wrapper for gymnax environments."""

    def __init__(
        self, env_name: str, max_steps: int = 1000, test: bool = False
    ):
        self.max_steps = max_steps
        self.test = test
        env, env_params = gymnax.make(env_name)
        env_params = env_params.replace(max_steps_in_episode=max_steps)
        self.obs_shape = env.obs_shape
        self.act_shape = env.num_actions
        self.num_actions = env.num_actions

        def reset_fn(key: chex.PRNGKey) -> GymState:
            key_re, key_ep = jax.random.split(key)
            obs, state = env.reset(key_re, env_params)
            state = GymState(state=state, obs=obs, rng=key_ep)
            return state

        self._reset_fn = jax.jit(jax.vmap(reset_fn))

        def step_fn(
            state: GymState, action: chex.Array
        ) -> Tuple[GymState, chex.Array, chex.Array]:
            key_st, key_ep = jax.random.split(state.rng)
            obs, env_state, reward, done, _ = env.step(
                key_st, state.state, action, env_params
            )
            state = state.replace(rng=key_ep, state=env_state, obs=obs)
            return state, reward, done

        self._step_fn = jax.jit(jax.vmap(step_fn))

    def reset(self, key: chex.PRNGKey) -> GymState:
        return self._reset_fn(key)

    def step(
        self, state: GymState, action: chex.Array
    ) -> Tuple[GymState, chex.Array, chex.Array]:
        return self._step_fn(state, action)


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
