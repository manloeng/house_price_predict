import matplotlib.pyplot as plt
import pandas as pd


def show_graph():
    plt.show()


def get_manchester_house_price_data():
    df = pd.read_csv(
        "/Users/andrewchung/Downloads/Average-prices-Property-Type-2022-08.csv")
    pd.to_datetime(df['Date'])
    return df.sort_values('Date')


def get_manchester_house_price_prices():
    """
    Filter data down to only greater manchester - for testing purpose
    """
    df = get_manchester_house_price_data()
    filter_df = df[['Date', 'Region_Name', 'Area_Code', 'Detached_Average_Price', 'Semi_Detached_Average_Price', 'Terraced_Average_Price', 'Flat_Average_Price']].copy()
    filter_df = filter_df.loc[df.Region_Name == "Greater Manchester"]
    filter_df.dropna()

    return filter_df


get_manchester_house_price_prices()
