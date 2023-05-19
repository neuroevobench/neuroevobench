"""EvoJAX MNIST VAE compatible policies."""
from typing import Tuple
import jax
import jax.numpy as jnp
from flax import linen as nn
from evojax.policy.base import PolicyNetwork
from evojax.policy.base import PolicyState
from evojax.util import get_params_format_fn
from evojax.task.base import TaskState


class Encoder(nn.Module):
    num_hidden_units: int
    num_vae_latents: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.num_hidden_units, name="fc1")(x)
        x = nn.relu(x)
        mean_x = nn.Dense(self.num_vae_latents, name="fc2_mean")(x)
        logvar_x = nn.Dense(self.num_vae_latents, name="fc2_logvar")(x)
        return mean_x, logvar_x


class Decoder(nn.Module):
    num_hidden_units: int

    @nn.compact
    def __call__(self, z):
        z = nn.Dense(self.num_hidden_units, name="fc1")(z)
        z = nn.relu(z)
        z = nn.Dense(784, name="fc2")(z)
        return z


class VAE(nn.Module):
    num_hidden_units: int = 32
    num_vae_latents: int = 20

    def setup(self):
        self.encoder = Encoder(self.num_hidden_units, self.num_vae_latents)
        self.decoder = Decoder(self.num_hidden_units)

    def __call__(self, x, z_rng):
        mean, logvar = self.encoder(x)
        z = reparameterize(z_rng, mean, logvar)
        recon_x = self.decoder(z)
        return recon_x, mean, logvar

    def generate(self, z):
        return nn.sigmoid(self.decoder(z))


def reparameterize(rng, mean, logvar):
    std = jnp.exp(0.5 * logvar)
    eps = jax.random.normal(rng, logvar.shape)
    return mean + eps * std


class MNIST_Generate_Policy(PolicyNetwork):
    """A MLP neural network for the MNIST generation task."""

    def __init__(
        self,
        num_hidden_units: int,
        num_vae_latents: int,
    ):
        self.input_dim = [1, 784]
        self.model = VAE(num_hidden_units, num_vae_latents)
        self.pholder_params = self.model.init(
            jax.random.PRNGKey(0),
            jnp.zeros(self.input_dim),
            jax.random.PRNGKey(0),
        )
        self.num_params, format_params_fn = get_params_format_fn(
            self.pholder_params
        )
        self._format_params_fn = jax.vmap(format_params_fn)
        self._forward_fn = jax.vmap(self.model.apply)

    def get_actions(
        self, t_states: TaskState, params: jnp.ndarray, p_states: PolicyState
    ) -> Tuple[jnp.ndarray, PolicyState]:
        params = self._format_params_fn(params)
        return (
            self._forward_fn(
                params,
                t_states.obs.reshape(
                    t_states.obs.shape[0], t_states.obs.shape[1], 784
                ),
                t_states.rng,
            ),
            p_states,
        )
