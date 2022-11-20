from .evojax_tasks import GymnaxTask, MinAtarPolicy, MNISTTask
from evojax.policy.convnet import ConvNetPolicy
from evojax.task.brax_task import BraxTask
from evojax.policy import MLPPolicy


def get_task(env_name: str):
    if env_name in ["ant"]:
        return get_brax_task(env_name)
    elif env_name in ["SpaceInvaders-MinAtar"]:
        return get_minatar_task(env_name)
    elif env_name in ["mnist"]:
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
