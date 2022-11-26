"""Utilities to port gymnax env to evoJAX tasks."""
from typing import Tuple, Sequence
import chex
import jax
import jax.numpy as jnp
from flax.struct import dataclass
from flax import linen as nn

import gymnax
from gymnax import EnvState
from evojax.task.base import VectorizedTask
from evojax.task.base import TaskState
from evojax.policy.base import PolicyNetwork
from evojax.policy.base import PolicyState
from evojax.util import get_params_format_fn
import tensorflow_datasets as tfds
from evojax.policy.convnet import ConvNetPolicy
from evojax.task.brax_task import BraxTask
from evojax.policy import MLPPolicy


def get_evojax_task(env_name: str):
    if env_name in ["ant", "humanoid"]:
        return get_brax_task(env_name)
    elif env_name in ["SpaceInvaders-MinAtar", "Breakout-MinAtar"]:
        return get_minatar_task(env_name)
    elif env_name in ["mnist", "fashion_mnist"]:
        return get_mnist_task(env_name)


def get_brax_task(env_name: str = "ant"):
    train_task = BraxTask(env_name, test=False)
    test_task = BraxTask(env_name, test=True)
    policy = MLPPolicy(
        input_dim=train_task.obs_shape[0],
        output_dim=train_task.act_shape[0],
        hidden_dims=4 * [32],
    )
    return train_task, test_task, policy


def get_minatar_task(env_name: str = "SpaceInvaders-MinAtar"):
    train_task = GymnaxTask(env_name, max_steps=500, test=False)
    test_task = GymnaxTask(env_name, max_steps=500, test=True)
    policy = MinAtarPolicy(
        input_dim=train_task.obs_shape,
        output_dim=train_task.num_actions,
        hidden_dim=32,
    )
    return train_task, test_task, policy


def get_mnist_task(env_name: str = "mnist"):
    # mnist, fashion_mnist, kmnist, mnist_corrupted
    train_task = MNISTTask(env_name, batch_size=1024, test=False)
    test_task = MNISTTask(env_name, batch_size=None, test=True)
    policy = ConvNetPolicy()
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


class CNN(nn.Module):
    feat_dim: int
    out_dim: int

    @nn.compact
    def __call__(self, x: chex.Array) -> chex.Array:
        x = nn.Conv(features=16, kernel_size=(3, 3), padding="SAME", strides=1)(
            x
        )
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(features=self.feat_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.out_dim)(x)
        return x


class MinAtarPolicy(PolicyNetwork):
    """Deterministic CNN policy - greedy action selection."""

    def __init__(
        self, input_dim: Sequence[int], hidden_dim: int, output_dim: int
    ):
        model = CNN(feat_dim=hidden_dim, out_dim=output_dim)
        params = model.init(jax.random.PRNGKey(0), jnp.ones([1, *input_dim]))
        self.num_params, format_params_fn = get_params_format_fn(params)
        self._format_params_fn = jax.vmap(format_params_fn)
        self._forward_fn = jax.vmap(model.apply)

    def get_actions(
        self, t_states: TaskState, params: chex.Array, p_states: PolicyState
    ) -> Tuple[chex.Array, PolicyState]:
        params = self._format_params_fn(params)
        obs = jnp.expand_dims(t_states.obs, axis=1)
        activations = self._forward_fn(params, obs)
        action = jnp.argmax(activations, axis=2).squeeze()
        return action, p_states


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
