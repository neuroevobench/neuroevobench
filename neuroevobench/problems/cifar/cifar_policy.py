from typing import Tuple
import jax
import jax.numpy as jnp
from flax import linen as nn
import chex


def conv_relu_block(
    x: chex.Array,
    features: int,
    kernel_size: Tuple[int, int],
    strides: Tuple[int, int],
    padding: str = "SAME",
) -> chex.Array:
    """Convolution layer + ReLU activation."""
    x = nn.Conv(
        features=features,
        kernel_size=kernel_size,
        strides=strides,
        use_bias=True,
        padding=padding,
    )(x)
    x = nn.relu(x)
    return x


class All_CNN_C(nn.Module):
    """All-CNN-inspired architecture as in Springenberg et al. (2015).
    Reference: https://arxiv.org/abs/1412.6806"""

    num_output_units: int = 10
    depth_1: int = 1
    depth_2: int = 1
    features_1: int = 8
    features_2: int = 16
    kernel_1: int = 5
    kernel_2: int = 5
    strides_1: int = 1
    strides_2: int = 1
    final_window: Tuple[int, int] = (32, 32)
    model_name: str = "All_CNN_C"

    @nn.compact
    def __call__(self, x: chex.Array) -> chex.Array:
        # Block In 1:
        for _ in range(self.depth_1):
            x = conv_relu_block(
                x,
                self.features_1,
                (self.kernel_1, self.kernel_1),
                (self.strides_1, self.strides_1),
            )

        # Block In 2:
        for _ in range(self.depth_2):
            x = conv_relu_block(
                x,
                self.features_2,
                (self.kernel_2, self.kernel_2),
                (self.strides_2, self.strides_2),
            )

        # Block Out: 1 × 1 conv. num_outputs-ReLu ×n
        x = nn.Conv(
            features=self.num_output_units,
            kernel_size=(1, 1),
            strides=(1, 1),
            use_bias=True,
            padding="SAME",
        )(x)

        # Global average pooling -> logits
        x = nn.avg_pool(
            x, window_shape=self.final_window, strides=None, padding="VALID"
        )
        x = jnp.squeeze(x, axis=(1, 2))

        return x


class CifarPolicy(object):
    """An LSTM for the Sequential MNIST classification task."""

    def __init__(self):
        self.input_dim = [1, 32, 32, 3]
        self.model = All_CNN_C()
        self.pholder_params = self.model.init(
            jax.random.PRNGKey(0), jnp.zeros(self.input_dim)
        )
