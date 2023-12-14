import matplotlib.pyplot as plt

from house_price_per_day.utils.index import get_manchester_house_price_prices, show_graph
from helper_function.data_prep.split_data import split_data
from helper_function.evaluation.plot_graph import plot_time_series
from helper_function.evaluation.metrics import evaluate_preds

house_prices = get_manchester_house_price_prices()

timesteps = house_prices['Date'].to_numpy()
average_detach_prices = house_prices["Detached_Average_Price"].to_numpy()

# splitting dataset
X_train, X_test, y_train, y_test = split_data(timesteps, average_detach_prices)

# Create a naive fore cast
naive_forecast = y_test[:-1]
# Plot naive forecast
fig, ax = plt.subplots(figsize=(10, 8))
plot_time_series(fig=fig, ax=ax, timesteps=X_test, values=y_test,  format=".", label="Test data")
plot_time_series(fig=fig, ax=ax, timesteps=X_test[1:], values=naive_forecast,  format=".",
                 label="Naive Forecast")

naive_results = evaluate_preds(y_true=y_test[1:],
                               y_pred=naive_forecast)

show_graph()
# should append results onto csv, so it's easy to access
print(naive_results)
