import time
import jax
import jax.numpy as jnp

try:
    import envpool
except Exception:
    print("envpool not installed, Atari problems will not work.")


class AtariTask(object):
    def __init__(
        self,
        policy,
        env_name: str = "Pong-v5",
        popsize: int = 256,
        num_envs_per_member: int = 1,
        max_steps: int = 1000,
        test: bool = False,
    ):
        self.env_name = env_name
        self.popsize = popsize
        self.num_envs_per_member = num_envs_per_member
        self.num_total_envs = int(self.popsize * self.num_envs_per_member)
        self.max_steps = max_steps
        self.test = test

        # Setup envpool environment step collector
        self.env = envpool.make_gym(self.env_name, num_envs=self.num_total_envs)
        self.handle, recv, send, self.step = self.env.xla()
        self.policy = policy
        self.num_dims = self.policy.num_params

    def evaluate(self, params: jnp.ndarray):
        """Evaluate multiple network params on multiple episodes."""
        # Rewards out - (popsize, envs_per_pop)
        # TODO(RobertTLange): Track total effective steps per generation
        all_rewards = self.evaluate_multi_params(params)
        return all_rewards

    def evaluate_single_params(self, params: jnp.ndarray) -> jnp.ndarray:
        def actor_step(iter, loop_var):
            handle0, (states, rewards, masks) = loop_var
            # Reshape = channels have to be last for flax conv2d
            states_re = jnp.moveaxis(states, 1, -1) / 255.0
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
        ep_masks = jnp.ones(self.num_total_envs)
        rewards = jnp.zeros(self.num_total_envs)
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
            states_re = jnp.moveaxis(states, 1, -1) / 255.0
            # Reshape obs = (popsize, env_per_pop, 84, 84, frames)
            states_batch = states_re.reshape(
                self.popsize, self.num_envs_per_member, 84, 84, 4
            )
            action = self.policy.get_actions(params, states_batch)
            # Reshape action = (env_per_pop x popsize)
            action_re = action.reshape(self.popsize * self.num_envs_per_member)
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
        ep_masks = jnp.ones(self.num_total_envs)
        rewards = jnp.zeros(self.num_total_envs)
        out = run_actor_loop(
            self.max_steps, (self.handle, (states, rewards, ep_masks))
        )
        handle_out, carry_out = out[0], out[1]
        all_rewards, all_masks = carry_out[1], carry_out[2]
        return all_rewards.reshape(self.popsize, self.num_envs_per_member)


if __name__ == "__main__":
    from evosax_benchmark.atari import AtariPolicy

    env = envpool.make_gym("Pong-v5", num_envs=1)
    policy = AtariPolicy(hidden_dims=[32], num_actions=env.action_space.n)

    popsize = 256
    num_envs_per_member = 2
    train_task = AtariTask(
        policy=policy,
        env_name="Pong-v5",
        max_steps=500,
        num_envs_per_member=num_envs_per_member,
        popsize=popsize,
    )
    test_task = AtariTask(
        policy=policy,
        env_name="Pong-v5",
        max_steps=500,
        num_envs_per_member=20,
        popsize=2,
    )
    # params = jnp.zeros((popsize, task.policy.num_params))
    # start_t = time.time()
    # all_rewards = task.evaluate(params)
    # print(time.time() - start_t)
    # print(all_rewards.shape)

    from evosax import OpenES

    strategy = OpenES(
        popsize=popsize,
        num_dims=train_task.num_dims,
        opt_name="adam",
        centered_rank=True,
        maximize=True,
        lrate_init=0.003,
        sigma_init=0.01,
    )

    print(f"START EVOLVING {train_task.num_dims} PARAMETERS.")
    rng = jax.random.PRNGKey(0)
    es_state = strategy.initialize(rng)
    best_return = -100
    for g in range(2000):
        start_t = time.time()
        rng, rng_ask = jax.random.split(rng)
        params, es_state = strategy.ask(rng_ask, es_state)
        all_rewards = train_task.evaluate(params)
        fit_re = jnp.mean(all_rewards, axis=1)
        es_state = strategy.tell(params, fit_re, es_state)
        if g % 10 == 0:
            eval_params = jnp.stack(
                [
                    es_state.mean.reshape(-1, 1),
                    es_state.best_member.reshape(-1, 1),
                ]
            ).squeeze()
            all_rewards = test_task.evaluate(eval_params)
            print("Eval", all_rewards.mean(axis=1))
        improved = best_return < fit_re.max()
        best_return = best_return * (1 - improved) + fit_re.max() * improved
        print(
            g,
            time.time() - start_t,
            best_return,
            fit_re.mean(),
            fit_re.max(),
            fit_re.min(),
        )
    # env = envpool.make_gym("Pong-v5", num_envs=2)
    # states, _ = env.reset()
    # print(states)
