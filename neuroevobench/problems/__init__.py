from .brax import brax_run

neb_eval_loops = {"brax": brax_run}


__all__ = ["brax_run", "neb_eval_loops"]
