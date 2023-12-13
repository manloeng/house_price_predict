def split_data(x, y):
    """
    Create train and test splits the right way for time series data

    Returns:
        X_train, X_test, y_train, y_test
    """

    # 80% train, 20% test - you can change these values as needed, e.g. 90/10
    split_size = int(0.8 * len(y))

    # Create train data splits (everything before the split)
    X_train, y_train = x[:split_size], y[:split_size]

    # Create test data splits (everything beyond the split)
    X_test, y_test = x[split_size:], y[split_size:]

    return X_train, X_test, y_train, y_test


def make_train_test_splits_time_series(windows, labels, test_split=0.2):
    """
    Splits matching pairs of windows and labels into train and test splits.
    """
    split_size = int(
        len(windows) * (1 - test_split)
    )  # this will default to 80% train/20% test
    train_windows = windows[:split_size]
    train_labels = labels[:split_size]
    test_windows = windows[split_size:]
    test_labels = labels[split_size:]
    return train_windows, test_windows, train_labels, test_labels
