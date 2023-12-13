import os
import tensorflow as tf


def create_model_checkpoint(model_name, save_path="model_experiments"):
    """
    Used with google colab
    """
    return tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(save_path, model_name),
        monitor="val_loss",
        verbose=0,  # only output a limited amount of text
        save_best_only=True,
    )
