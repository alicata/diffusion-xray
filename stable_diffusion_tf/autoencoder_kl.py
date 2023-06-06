import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa

from .layers import apply_seq, PaddedConv2D


class AttentionBlock(keras.layers.Layer):
    def __init__(self, channels):
        super().__init__()
        self.norm = tfa.layers.GroupNormalization(epsilon=1e-5)
        self.q = PaddedConv2D(channels, 1)
        self.k = PaddedConv2D(channels, 1)
        self.v = PaddedConv2D(channels, 1)
        self.proj_out = PaddedConv2D(channels, 1)

    def call(self, x):
        h_ = self.norm(x)
        q, k, v = self.q(h_), self.k(h_), self.v(h_)

        # Compute attention
        b, h, w, c = q.shape
        q = tf.reshape(q, (-1, h * w, c))  # b,hw,c
        k = keras.layers.Permute((3, 1, 2))(k)
        k = tf.reshape(k, (-1, c, h * w))  # b,c,hw
        w_ = q @ k
        w_ = w_ * (c ** (-0.5))
        w_ = keras.activations.softmax(w_)

        # Attend to values
        v = keras.layers.Permute((3, 1, 2))(v)
        v = tf.reshape(v, (-1, c, h * w))
        w_ = keras.layers.Permute((2, 1))(w_)
        h_ = v @ w_
        h_ = keras.layers.Permute((2, 1))(h_)
        h_ = tf.reshape(h_, (-1, h, w, c))
        return x + self.proj_out(h_)


class ResnetBlock(keras.layers.Layer):
    def __init__(self, in_channels, out_channels=None):
        super().__init__()
        if out_channels is None:
            out_channels = in_channels
        self.norm1 = tfa.layers.GroupNormalization(epsilon=1e-5)
        self.conv1 = PaddedConv2D(out_channels, 3, padding=1)
        self.norm2 = tfa.layers.GroupNormalization(epsilon=1e-5)
        self.conv2 = PaddedConv2D(out_channels, 3, padding=1)
        self.nin_shortcut = (
            PaddedConv2D(out_channels, 1)
            if in_channels != out_channels
            else lambda x: x
        )

    def call(self, x):
        h = self.conv1(keras.activations.swish(self.norm1(x)))
        h = self.conv2(keras.activations.swish(self.norm2(h)))
        return self.nin_shortcut(x) + h


class Decoder(keras.Sequential):
    def __init__(self):
        super().__init__(
            [
                keras.layers.Lambda(lambda x: 1 / 0.18215 * x),
                PaddedConv2D(4, 1),
                PaddedConv2D(512, 3, padding=1),
                ResnetBlock(512, 512),
                AttentionBlock(512),
                ResnetBlock(512, 512),
                ResnetBlock(512, 512),
                ResnetBlock(512, 512),
                ResnetBlock(512, 512),
                keras.layers.UpSampling2D(size=(2, 2)),
                PaddedConv2D(512, 3, padding=1),
                ResnetBlock(512, 512),
                ResnetBlock(512, 512),
                ResnetBlock(512, 512),
                keras.layers.UpSampling2D(size=(2, 2)),
                PaddedConv2D(512, 3, padding=1),
                ResnetBlock(512, 256),
                ResnetBlock(256, 256),
                ResnetBlock(256, 256),
                keras.layers.UpSampling2D(size=(2, 2)),
                PaddedConv2D(256, 3, padding=1),
                ResnetBlock(256, 128),
                ResnetBlock(128, 128),
                ResnetBlock(128, 128),
                tfa.layers.GroupNormalization(epsilon=1e-5),
                keras.layers.Activation("swish"),
                PaddedConv2D(3, 3, padding=1),
            ]
        )

class PaddedConv2DE(keras.layers.Layer):
    def __init__(self, filters, kernel_size, padding=0, strides=1, **kwargs):
        super().__init__(**kwargs)
        self.padding2d = keras.layers.ZeroPadding2D(padding)
        self.conv2d = keras.layers.Conv2D(filters, kernel_size, strides=strides)

    def call(self, inputs):
        x = self.padding2d(inputs)
        return self.conv2d(x)

class GroupNormalizationE(tf.keras.layers.Layer):
    """GroupNormalization layer.

    This layer is only here temporarily and will be removed
    as we introduce GroupNormalization in core Keras.
    """

    def __init__(
        self,
        groups=32,
        axis=-1,
        epsilon=1e-5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.groups = groups
        self.axis = axis
        self.epsilon = epsilon

    def build(self, input_shape):
        dim = input_shape[self.axis]
        self.gamma = self.add_weight(
            shape=(dim,),
            name="gamma",
            initializer="ones",
        )
        self.beta = self.add_weight(
            shape=(dim,),
            name="beta",
            initializer="zeros",
        )

    def call(self, inputs):
        input_shape = tf.shape(inputs)
        reshaped_inputs = self._reshape_into_groups(inputs, input_shape)
        normalized_inputs = self._apply_normalization(reshaped_inputs, input_shape)
        return tf.reshape(normalized_inputs, input_shape)

    def _reshape_into_groups(self, inputs, input_shape):
        group_shape = [input_shape[i] for i in range(inputs.shape.rank)]
        group_shape[self.axis] = input_shape[self.axis] // self.groups
        group_shape.insert(self.axis, self.groups)
        group_shape = tf.stack(group_shape)
        return tf.reshape(inputs, group_shape)

    def _apply_normalization(self, reshaped_inputs, input_shape):
        group_reduction_axes = list(range(1, reshaped_inputs.shape.rank))
        axis = -2 if self.axis == -1 else self.axis - 1
        group_reduction_axes.pop(axis)
        mean, variance = tf.nn.moments(
            reshaped_inputs, group_reduction_axes, keepdims=True
        )
        gamma, beta = self._get_reshaped_weights(input_shape)
        return tf.nn.batch_normalization(
            reshaped_inputs,
            mean=mean,
            variance=variance,
            scale=gamma,
            offset=beta,
            variance_epsilon=self.epsilon,
        )

    def _get_reshaped_weights(self, input_shape):
        broadcast_shape = self._create_broadcast_shape(input_shape)
        gamma = tf.reshape(self.gamma, broadcast_shape)
        beta = tf.reshape(self.beta, broadcast_shape)
        return gamma, beta

    def _create_broadcast_shape(self, input_shape):
        broadcast_shape = [1] * len(input_shape)
        broadcast_shape[self.axis] = input_shape[self.axis] // self.groups
        broadcast_shape.insert(self.axis, self.groups)
        return broadcast_shape

class ResnetBlockE(keras.layers.Layer):
    def __init__(self, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.output_dim = output_dim
        self.norm1 = GroupNormalizationE(epsilon=1e-5)
        self.conv1 = PaddedConv2DE(output_dim, 3, padding=1)
        self.norm2 = GroupNormalizationE(epsilon=1e-5)
        self.conv2 = PaddedConv2DE(output_dim, 3, padding=1)

    def build(self, input_shape):
        if input_shape[-1] != self.output_dim:
            self.residual_projection = PaddedConv2D(self.output_dim, 1)
        else:
            self.residual_projection = lambda x: x

    def call(self, inputs):
        x = self.conv1(keras.activations.swish(self.norm1(inputs)))
        x = self.conv2(keras.activations.swish(self.norm2(x)))
        return x + self.residual_projection(inputs)



class AttentionBlockE(keras.layers.Layer):
    def __init__(self, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.output_dim = output_dim
        self.norm = GroupNormalizationE(epsilon=1e-5)
        self.q = PaddedConv2D(output_dim, 1)
        self.k = PaddedConv2D(output_dim, 1)
        self.v = PaddedConv2D(output_dim, 1)
        self.proj_out = PaddedConv2D(output_dim, 1)

    def call(self, inputs):
        x = self.norm(inputs)
        q, k, v = self.q(x), self.k(x), self.v(x)

        # Compute attention
        _, h, w, c = q.shape
        q = tf.reshape(q, (-1, h * w, c))  # b, hw, c
        k = tf.transpose(k, (0, 3, 1, 2))
        k = tf.reshape(k, (-1, c, h * w))  # b, c, hw
        y = q @ k
        y = y * (c**-0.5)
        y = keras.activations.softmax(y)

        # Attend to values
        v = tf.transpose(v, (0, 3, 1, 2))
        v = tf.reshape(v, (-1, c, h * w))
        y = tf.transpose(y, (0, 2, 1))
        x = v @ y
        x = tf.transpose(x, (0, 2, 1))
        x = tf.reshape(x, (-1, h, w, c))
        return self.proj_out(x) + inputs

class Encoder(keras.Sequential):
    """VAE Encoder to convert image into latent embedding tensor """

    def __init__(self, img_height=512, img_width=512, download_weights=True):
        super().__init__(
            [
                keras.layers.Input((img_height, img_width, 3)),
                PaddedConv2DE(128, 3, padding=1),
                ResnetBlockE(128),
                ResnetBlockE(128),
                PaddedConv2DE(128, 3, padding=1, strides=2),
                ResnetBlockE(256),
                ResnetBlockE(256),
                PaddedConv2DE(256, 3, padding=1, strides=2),
                ResnetBlockE(512),
                ResnetBlockE(512),
                PaddedConv2DE(512, 3, padding=1, strides=2),
                ResnetBlockE(512),
                ResnetBlockE(512),
                ResnetBlockE(512),
                AttentionBlockE(512),
                ResnetBlockE(512),
                GroupNormalizationE(epsilon=1e-5),
                keras.layers.Activation("swish"),
                PaddedConv2DE(8, 3, padding=1),
                PaddedConv2DE(8, 1),
                # expand from 3 to 4 channels
                keras.layers.Lambda(lambda x: x[..., :4] * 0.18215),
            ]
        )

