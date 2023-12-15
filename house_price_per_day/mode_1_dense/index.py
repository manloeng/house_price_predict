import tensorflow as tf
from tensorflow.keras import layers

from house_price_per_day.utils.model import compile_model, fit_model
from helper_function.data_prep.time_series_data_prep import train_test_data
from helper_function.evaluation.index import evaluate

HORIZON = 1  # predict next 1 day
WINDOW_SIZE = 7  # use the past week of Bitcoin data to make the prediction


def main():
    model_name = 'model_1_dense'
    train_windows, test_windows, train_labels, test_labels = train_test_data(WINDOW_SIZE, HORIZON)
    train_model_1(model_name, train_windows, test_windows, train_labels, test_labels)
    evaluate(f"model_experiments/{model_name}", test_windows, test_labels)
    print("complete!")


def train_model_1(model_name, train_windows, test_windows, train_labels, test_labels):
    # Set random seed for as reproducible results as possible
    tf.random.set_seed(42)

    # 1. Construct model
    model_1 = tf.keras.Sequential([
        layers.Dense(128, activation="relu"),
        layers.Dense(HORIZON, activation="linear")  # linear activation is the same as having no activation
    ], name=model_name)  # name our model so we can save it

    # 2. Compile
    compile_model(model_1)

    # 3. Fit the model
    fit_model(model_1, train_windows, train_labels, test_windows, test_labels)
    print(f"evaluate model: {model_1.evaluate(test_windows, test_labels)}")


main()
