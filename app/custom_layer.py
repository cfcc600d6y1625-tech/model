import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


@tf.keras.utils.register_keras_serializable(package="Synar")
class ChannelAttention(layers.Layer):
    def __init__(self, reduction_ratio=8, **kwargs):
        super().__init__(**kwargs)
        self.reduction_ratio = reduction_ratio
        self.global_avg_pool = layers.GlobalAveragePooling2D()
        self.global_max_pool = layers.GlobalMaxPooling2D()
        self.reshape = None
        self.shared_dense_1 = None
        self.shared_dense_2 = None

    def build(self, input_shape):
        channels = int(input_shape[-1])
        reduced_channels = max(channels // self.reduction_ratio, 1)

        self.reshape = layers.Reshape((1, 1, channels))
        self.shared_dense_1 = layers.Dense(
            reduced_channels,
            activation="relu",
            kernel_initializer="he_normal",
            use_bias=True
        )
        self.shared_dense_2 = layers.Dense(
            channels,
            kernel_initializer="he_normal",
            use_bias=True
        )

        super().build(input_shape)

    def call(self, inputs):
        avg_pool = self.global_avg_pool(inputs)
        max_pool = self.global_max_pool(inputs)

        avg_pool = self.reshape(avg_pool)
        max_pool = self.reshape(max_pool)

        avg_out = self.shared_dense_2(self.shared_dense_1(avg_pool))
        max_out = self.shared_dense_2(self.shared_dense_1(max_pool))

        attention = tf.nn.sigmoid(avg_out + max_out)

        return inputs * attention

    def get_config(self):
        config = super().get_config()
        config.update({
            "reduction_ratio": self.reduction_ratio
        })
        return config


@tf.keras.utils.register_keras_serializable(package="Synar")
class ColorFeatureStandardization(layers.Layer):
    def __init__(self, mean, std, **kwargs):
        super().__init__(**kwargs)
        self.mean = mean
        self.std = std

    def build(self, input_shape):
        self.mean_tensor = tf.constant(self.mean, dtype=tf.float32)
        self.std_tensor = tf.constant(self.std, dtype=tf.float32)
        super().build(input_shape)

    def call(self, inputs):
        return (tf.cast(inputs, tf.float32) - self.mean_tensor) / (self.std_tensor + 1e-8)

    def get_config(self):
        config = super().get_config()
        config.update({
            "mean": self.mean,
            "std": self.std
        })
        return config


def get_custom_layers():
    return {
        "ChannelAttention": ChannelAttention,
        "ColorFeatureStandardization": ColorFeatureStandardization,
        "Synar>ChannelAttention": ChannelAttention,
        "Synar>ColorFeatureStandardization": ColorFeatureStandardization
    }
