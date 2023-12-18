import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers

from house_price_per_day.utils.index import get_manchester_house_price_prices, show_graph
from house_price_per_day.utils.model import compile_model, fit_model
from helper_function.data_prep.time_series_data_prep import train_test_data
from helper_function.evaluation.index import evaluate
from helper_function.evaluation.plot_graph import plot_time_series

HORIZON = 1  # predict next 1 day
WINDOW_SIZE = 7  # use the past week of Bitcoin data to make the prediction


def main():
    house_prices = get_manchester_house_price_prices()

    model_name = 'model_1_dense'
    train_windows, test_windows, train_labels, test_labels = train_test_data(house_prices, WINDOW_SIZE, HORIZON)
    # train_model_1(model_name, train_windows, test_windows, train_labels, test_labels)
    model_results, model_preds = evaluate(f"model_experiments/{model_name}", test_windows, test_labels)

    print("---------------------------")
    print(test_labels.shape, "label shape")
    print(model_preds.shape, "model shape")

    offset = 0
    timesteps = house_prices['Date'].to_numpy()
    fig, ax = plt.subplots(figsize=(10, 8))
    plot_time_series(fig=fig, ax=ax,
                     timesteps=timesteps[-len(test_windows):],
                     values=test_labels[:, 0],
                     start=offset,
                     label="Test Data")

    plot_time_series(fig=fig, ax=ax,
                     timesteps=timesteps[-len(test_windows):],
                     values=tf.expand_dims(model_preds, axis=1),
                     format="-",
                     label="Test Data")

    show_graph()
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
