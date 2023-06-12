from .neb_evaluator import NeuroevolutionEvaluator
from .addition import addition_run
from .atari import atari_run
from .bbob import bbob_run
from .brax import brax_run
from .cifar import cifar_run
from .hpob import hpob_run
from .minatar import minatar_run
from .mnist_classify import mnist_classify_run
from .mnist_generate import mnist_generate_run
from .smnist import smnist_run
from .svhn import svhn_run

# Dictionary look up of training loops
neb_eval_loops = {
    "addition": addition_run,
    "atari": atari_run,
    "bbob": bbob_run,
    "brax": brax_run,
    "cifar": cifar_run,
    "hpob": hpob_run,
    "minatar": minatar_run,
    "mnist_classify": mnist_classify_run,
    "mnist_generate": mnist_generate_run,
    "smnist": smnist_run,
    "svhn": svhn_run,
}


__all__ = [
    "NeuroevolutionEvaluator",
    "addition_run",
    "atari_run",
    "bbob_run",
    "brax_run",
    "cifar_run",
    "hpob_run",
    "minatar_run",
    "mnist_classify_run",
    "mnist_generate_run",
    "smnist_run",
    "svhn_run",
    "neb_eval_loops",
]
