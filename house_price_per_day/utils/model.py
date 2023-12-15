import tensorflow as tf
from helper_function.tensorlfow_methods.index import create_model_checkpoint


# generic params used to compile model
def compile_model(model):
    model.compile(
        loss="mae",
        optimizer=tf.keras.optimizers.legacy.Adam(),
        metrics=["mae", "mse"]
    )


# generic params used to fit model
def fit_model(model, train_windows, train_labels, test_windows, test_labels):
    model.fit(
        x=train_windows,
        y=train_labels,
        epochs=100,
        verbose=1,
        batch_size=128,
        validation_data=(test_windows, test_labels),
        callbacks=[create_model_checkpoint(model_name=model.name)]
    )
