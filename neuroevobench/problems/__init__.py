from .neb_evaluator import NeuroevolutionEvaluator
from .addition import addition_run
from .atari import atari_run
from .brax import brax_run
from .cifar import cifar_run
from .minatar import minatar_run
from .mnist_classify import mnist_classify_run

neb_eval_loops = {
    "addition": addition_run,
    "atari": atari_run,
    "brax": brax_run,
    "cifar": cifar_run,
    "minatar": minatar_run,
    "mnist_classify": mnist_classify_run,
}


__all__ = [
    "NeuroevolutionEvaluator",
    "addition_run",
    "atari_run",
    "brax_run",
    "cifar_run",
    "minatar_run",
    "mnist_classify_run",
    "neb_eval_loops",
]
