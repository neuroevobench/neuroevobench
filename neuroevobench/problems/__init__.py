from .neb_evaluator import NeuroevolutionEvaluator
from .addition import addition_run
from .atari import atari_run
from .brax import brax_run
from .cifar import cifar_run

neb_eval_loops = {
    "addition": addition_run,
    "atari": atari_run,
    "brax": brax_run,
    "cifar": cifar_run,
}


__all__ = [
    "NeuroevolutionEvaluator",
    "addition_run",
    "atari_run",
    "brax_run",
    "cifar_run",
    "neb_eval_loops",
]
