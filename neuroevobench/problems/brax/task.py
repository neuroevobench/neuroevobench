"""Adapted from https://github.com/google/evojax/blob/main/evojax/task/brax_task.py"""
import sys
from typing import Tuple, Any

import jax
import jax.numpy as jnp
from flax.struct import dataclass

from evojax.task.base import VectorizedTask
from evojax.task.base import TaskState

try:
    from brax.v1.envs import create
    from brax.v1.envs import State as BraxState
    from .modified_ant import create_modified_ant_env
except Exception:
    print("You need to install brax for Brax tasks:")
    print("  pip install git+https://github.com/google/brax.git@main")
    # sys.exit(1)


@dataclass
class State(TaskState):
    state: Any
    obs: jnp.ndarray


def get_state_dataclass(**states):
    return State(**states)


class BraxTask(VectorizedTask):
    """Tasks from the Brax simulator."""

    def __init__(
        self,
        env_name: str,
        max_steps: int = 1000,
        legacy_spring: bool = True,
        test: bool = False,
    ):
        self.max_steps = max_steps
        self.test = test
        if env_name == "ant_modified":
            brax_env = create_modified_ant_env(episode_length=max_steps)
        else:
            brax_env = create(
                env_name=env_name,
                episode_length=max_steps,
                legacy_spring=legacy_spring,
            )

        self.obs_shape = tuple(
            [
                brax_env.observation_size,
            ]
        )
        self.act_shape = tuple(
            [
                brax_env.action_size,
            ]
        )

        def reset_fn(key):
            state = brax_env.reset(key)
            state = get_state_dataclass(state=state, obs=state.obs)
            return state

        self._reset_fn = jax.jit(jax.vmap(reset_fn))

        def step_fn(state, action):
            brax_state = brax_env.step(state.state, action)
            state = state.replace(state=brax_state, obs=brax_state.obs)
            return state, brax_state.reward, brax_state.done

        self._step_fn = jax.jit(jax.vmap(step_fn))

    def reset(self, key: jnp.ndarray) -> State:
        return self._reset_fn(key)

    def step(
        self, state: State, action: jnp.ndarray
    ) -> Tuple[State, jnp.ndarray, jnp.ndarray]:
        return self._step_fn(state, action)
