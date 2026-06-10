from pathlib import Path
from datetime import datetime

import tensorflow as tf
from tensorflow import keras


class LearningRateLogger(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        try:
            lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        except Exception:
            lr = None

        if lr is not None:
            logs["learning_rate"] = lr
            print(f"\nEpoch {epoch + 1}: learning_rate = {lr:.8f}")


def create_training_callbacks(
    output_dir,
    checkpoint_name="best_model.keras",
    log_subdir="tensorboard",
    monitor="val_ordinal_mae",
    mode="min",
    early_stop_patience=6,
    reduce_lr_patience=2,
    min_lr=1e-7
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = output_dir / checkpoint_name

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = output_dir / log_subdir / timestamp
    log_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor=monitor,
            mode=mode,
            save_best_only=True,
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor=monitor,
            mode=mode,
            patience=early_stop_patience,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor=monitor,
            mode=mode,
            factor=0.5,
            patience=reduce_lr_patience,
            min_lr=min_lr,
            verbose=1
        ),
        keras.callbacks.TensorBoard(
            log_dir=str(log_dir),
            histogram_freq=1,
            write_graph=True,
            write_images=False,
            update_freq="epoch"
        ),
        keras.callbacks.CSVLogger(
            filename=str(output_dir / "training_log.csv"),
            append=True
        ),
        LearningRateLogger()
    ]

    return callbacks, checkpoint_path, log_dir
