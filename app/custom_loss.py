import tensorflow as tf
from tensorflow import keras


@tf.keras.utils.register_keras_serializable(package="Synar")
class OrdinalMAE(tf.keras.metrics.Metric):
    def __init__(self, name="ordinal_mae", **kwargs):
        super().__init__(name=name, **kwargs)
        self.total = self.add_weight(name="total", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
        y_pred_label = tf.cast(tf.argmax(y_pred, axis=-1), tf.float32)

        error = tf.abs(y_pred_label - y_true)

        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, tf.float32)
            error = error * sample_weight
            batch_count = tf.reduce_sum(sample_weight)
        else:
            batch_count = tf.cast(tf.size(error), tf.float32)

        self.total.assign_add(tf.reduce_sum(error))
        self.count.assign_add(batch_count)

    def result(self):
        return tf.math.divide_no_nan(self.total, self.count)

    def reset_state(self):
        self.total.assign(0.0)
        self.count.assign(0.0)


@tf.keras.utils.register_keras_serializable(package="Synar")
class OrdinalFocalLoss(tf.keras.losses.Loss):
    def __init__(
        self,
        num_classes=6,
        gamma=2.0,
        ordinal_lambda=0.15,
        name="ordinal_focal_loss",
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.gamma = gamma
        self.ordinal_lambda = ordinal_lambda

    def call(self, y_true, y_pred):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)

        y_pred = tf.cast(y_pred, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)

        ce = keras.losses.sparse_categorical_crossentropy(y_true, y_pred)

        y_true_onehot = tf.one_hot(y_true, depth=self.num_classes)
        p_t = tf.reduce_sum(y_true_onehot * y_pred, axis=-1)

        focal_factor = tf.pow(1.0 - p_t, self.gamma)
        focal_loss = focal_factor * ce

        class_indices = tf.cast(tf.range(self.num_classes), tf.float32)
        expected_label = tf.reduce_sum(y_pred * class_indices, axis=-1)
        true_label = tf.cast(y_true, tf.float32)

        ordinal_penalty = tf.abs(expected_label - true_label) / float(self.num_classes - 1)

        return focal_loss + self.ordinal_lambda * ordinal_penalty

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_classes": self.num_classes,
            "gamma": self.gamma,
            "ordinal_lambda": self.ordinal_lambda
        })
        return config


def get_custom_objects():
    return {
        "OrdinalMAE": OrdinalMAE,
        "OrdinalFocalLoss": OrdinalFocalLoss,
        "Synar>OrdinalMAE": OrdinalMAE,
        "Synar>OrdinalFocalLoss": OrdinalFocalLoss
    }
