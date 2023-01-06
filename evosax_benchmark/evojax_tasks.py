"""Utilities to port gymnax/mnist env to evoJAX tasks."""
from typing import Sequence, Optional
from gymnax.utils.evojax_wrapper import GymnaxTask
from evojax.task.brax_task import BraxTask
from .evojax_policies import BraxPolicy, MNISTPolicy, MinAtarPolicy, GymPolicy

# from .mnist_tasks import MNISTTask
from .mnist_tasks_torch import MNISTTask


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
    elif env_name in [
        "CartPole-v1",
        "Acrobot-v1",
        "Pendulum-v1",
    ]:
        train_task, test_task, policy = get_gym_task(
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
    policy = BraxPolicy(
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


def get_gym_task(
    env_name: str = "CartPole-v1",
    hidden_dims: Sequence[int] = [32],
    max_steps: int = 500,
):
    train_task = GymnaxTask(env_name, max_steps=max_steps, test=False)
    test_task = GymnaxTask(env_name, max_steps=max_steps, test=True)
    policy = GymPolicy(
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
