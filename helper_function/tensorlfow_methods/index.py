import os
import tensorflow as tf
import datetime


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


# Run: `tensorboard --logdir house_price_per_day` to see tensorboard graphs
def create_tensorboard_callback(dir_name, experiment_name):
    log_dir = dir_name + "/" + experiment_name + "/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
    print(f"Saving TensorBoard log files to: {log_dir}")
    return tensorboard_callback
