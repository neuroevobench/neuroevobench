from typing import Any
import chex
from functools import partial
from typing import Sequence
import jax
import jax.numpy as jnp
from flax import linen as nn

ModuleDef = Any


class ConvBlock(nn.Module):
    channels: int
    kernel_size: int
    stride: int = 1
    act: bool = True

    @nn.compact
    def __call__(self, x: chex.Array) -> chex.Array:
        x = nn.Conv(
            self.channels,
            (self.kernel_size, self.kernel_size),
            strides=self.stride,
            padding="SAME",
            use_bias=False,
            kernel_init=nn.initializers.kaiming_normal(),
        )(x)
        if self.act:
            x = nn.swish(x)
        return x


class ResidualBlock(nn.Module):
    channels: int
    conv_block: ModuleDef

    @nn.compact
    def __call__(self, x: chex.Array) -> chex.Array:
        channels = self.channels
        conv_block = self.conv_block

        shortcut = x

        residual = conv_block(channels, 3)(x)
        residual = conv_block(channels, 3, act=False)(residual)

        # Projection shortcut to make shapes work with residual addition
        if shortcut.shape != residual.shape:
            shortcut = conv_block(channels, 1, act=False)(shortcut)

        gamma = self.param("gamma", nn.initializers.zeros, 1, jnp.float32)
        out = shortcut + gamma * residual
        out = nn.swish(out)
        return out


class Stage(nn.Module):
    channels: int
    num_blocks: int
    stride: int
    block: ModuleDef

    @nn.compact
    def __call__(self, x: chex.Array) -> chex.Array:
        stride = self.stride
        if stride > 1:
            x = nn.max_pool(x, (stride, stride), strides=(stride, stride))
        for _ in range(self.num_blocks):
            x = self.block(self.channels)(x)
        return x


class Body(nn.Module):
    channel_list: Sequence[int]
    num_blocks_list: Sequence[int]
    strides: Sequence[int]
    stage: ModuleDef

    @nn.compact
    def __call__(self, x: chex.Array) -> chex.Array:
        for channels, num_blocks, stride in zip(
            self.channel_list, self.num_blocks_list, self.strides
        ):
            x = self.stage(channels, num_blocks, stride)(x)
        return x


class Stem(nn.Module):
    channel_list: Sequence[int]
    stride: int
    conv_block: ModuleDef

    @nn.compact
    def __call__(self, x: chex.Array) -> chex.Array:
        stride = self.stride
        for channels in self.channel_list:
            x = self.conv_block(channels, 3, stride=stride)(x)
            stride = 1
        return x


class Head(nn.Module):
    classes: int

    @nn.compact
    def __call__(self, x: chex.Array) -> chex.Array:
        x = jnp.mean(x, axis=(1, 2))
        x = nn.Dense(self.classes)(x)
        return x


class ResNet(nn.Module):
    classes: int
    channel_list: Sequence[int]
    num_blocks_list: Sequence[int]
    strides: Sequence[int]
    head_p_drop: float = 0.0

    @nn.compact
    def __call__(self, x: chex.Array):
        conv_block = partial(ConvBlock)
        residual_block = partial(ResidualBlock, conv_block=conv_block)
        stage = partial(Stage, block=residual_block)

        x = Stem([32, 32, 64], self.strides[0], conv_block)(x)
        x = Body(
            self.channel_list, self.num_blocks_list, self.strides[1:], stage
        )(x)
        x = Head(self.classes)(x)
        return x


class SVHNPolicy(object):
    """An LSTM for the Sequential MNIST classification task."""

    def __init__(self, resnet_number: int):
        STAGE_SIZES = {
            # Num conv blocks (2 conv layers + skip connection)
            # with feature channels [64, 128, 256, 512]
            9: [1, 1, 1, 1],
            18: [2, 2, 2, 2],
            34: [3, 4, 6, 3],
            50: [3, 4, 6, 3],
            101: [3, 4, 23, 3],
            152: [3, 8, 36, 3],
            200: [3, 24, 36, 3],
            269: [3, 30, 48, 8],
        }

        # WideResNet: channel_list=[128, 256, 512, 1024]

        self.model = ResNet(
            classes=10,
            channel_list=[64, 128, 256, 512],
            num_blocks_list=STAGE_SIZES[resnet_number],
            strides=[1, 1, 2, 2, 2],
            head_p_drop=0.3,
        )
        self.input_dim = [1, 32, 32, 3]
        self.pholder_params = self.model.init(
            jax.random.PRNGKey(0), jnp.zeros(self.input_dim)
        )
