from typing import Optional
import time
import jax
import jax.numpy as jnp
import envpool
import multiprocessing
from atari_policy import AtariPolicy


class AtariTask(object):
    def __init__(
        self,
        env_name: str = "Pong-v5",
        max_steps: int = 1000,
        num_envs: Optional[int] = None,
        popsize: int = 256,
        test: bool = False,
    ):
        self.env_name = env_name
        self.max_steps = max_steps
        if num_envs is None:
            self.num_envs = multiprocessing.cpu_count()
        else:
            self.num_envs = num_envs
        self.popsize = popsize
        self.envs_per_pop = int(self.num_envs / self.popsize)
        self.test = test

        # Setup envpool environment step collector
        self.env = envpool.make_gym("Pong-v5", num_envs=num_envs)
        self.handle, recv, send, self.step = self.env.xla()
        self.policy = AtariPolicy(
            hidden_dims=[32], num_actions=self.env.action_space.n
        )

    def evaluate(self, params: jnp.ndarray):
        """Evaluate multiple network params on multiple episodes."""
        # Rewards out - (popsize, envs_per_pop)
        all_rewards = self.evaluate_multi_params(params)
        return all_rewards

    def evaluate_single_params(self, params: jnp.ndarray) -> jnp.ndarray:
        def actor_step(iter, loop_var):
            handle0, (states, rewards, masks) = loop_var
            # Reshape = channels have to be last for flax conv2d
            states_re = jnp.moveaxis(states, 1, -1)
            action = self.policy.get_actions_single(params, states_re)
            # handle - stores state of environment
            handle1, (new_states, rew, term, trunc, info) = self.step(
                handle0, action
            )
            new_rewards = rewards + rew * masks
            new_masks = masks * (1 - term)
            return (handle1, (new_states, new_rewards, new_masks))

        @jax.jit
        def run_actor_loop(num_steps, init_var):
            return jax.lax.fori_loop(0, num_steps, actor_step, init_var)

        states, _ = self.env.reset()
        ep_masks, rewards = jnp.ones(self.num_envs), jnp.zeros(self.num_envs)
        out = run_actor_loop(
            self.max_steps, (self.handle, (states, rewards, ep_masks))
        )
        handle_out, carry_out = out[0], out[1]
        all_rewards, all_masks = carry_out[1], carry_out[2]
        return all_rewards

    def evaluate_multi_params(self, params: jnp.ndarray) -> jnp.ndarray:
        def actor_step(iter, loop_var):
            handle0, (states, rewards, masks) = loop_var
            # Reshape = channels have to be last for flax conv2d
            states_re = jnp.moveaxis(states, 1, -1)
            # Reshape obs = (popsize, env_per_pop, 84, 84, frames)
            states_batch = states_re.reshape(
                self.popsize, self.envs_per_pop, 84, 84, 4
            )
            action = self.policy.get_actions(params, states_batch)
            # Reshape action = (env_per_pop x popsize)
            action_re = action.reshape(self.envs_per_pop * self.popsize)
            # handle - stores state of environment
            handle1, (new_states, rew, term, trunc, info) = self.step(
                handle0, action_re
            )
            new_rewards = rewards + rew * masks
            new_masks = masks * (1 - term)
            return (handle1, (new_states, new_rewards, new_masks))

        @jax.jit
        def run_actor_loop(num_steps, init_var):
            return jax.lax.fori_loop(0, num_steps, actor_step, init_var)

        states, _ = self.env.reset()
        ep_masks, rewards = jnp.ones(self.num_envs), jnp.zeros(self.num_envs)
        out = run_actor_loop(
            self.max_steps, (self.handle, (states, rewards, ep_masks))
        )
        handle_out, carry_out = out[0], out[1]
        all_rewards, all_masks = carry_out[1], carry_out[2]
        return all_rewards.reshape(self.popsize, self.envs_per_pop)


if __name__ == "__main__":
    popsize = 256
    num_envs = 256 * 1
    task = AtariTask(
        "Pong-v5", max_steps=1000, num_envs=num_envs, popsize=popsize
    )
    params = jnp.zeros((popsize, task.policy.num_params))
    start_t = time.time()
    all_rewards = task.evaluate(params)
    print(time.time() - start_t)
    print(all_rewards.shape)
