from .neb_evaluator import NeuroevolutionEvaluator
from .addition import addition_run
from .atari import atari_run
from .brax import brax_run

neb_eval_loops = {
    "addition": addition_run,
    "atari": atari_run,
    "brax": brax_run,
}


__all__ = [
    "NeuroevolutionEvaluator",
    "addition_run",
    "atari_run",
    "brax_run",
    "neb_eval_loops",
]
