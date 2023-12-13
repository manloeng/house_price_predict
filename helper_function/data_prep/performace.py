import tensorflow as tf

BATCH_SIZE = 1024


def combine_dataset(features: tf.data.Dataset, labels: tf.data.Dataset):
    """
    Combine labels and features by zipping together -> (features, labels)
    """
    return tf.data.Dataset.zip((features, labels))


# Batch and prefetch
def increase_dataset_performance(train_dataset: tf.data.Dataset.zip, test_dataset: tf.data.Dataset.zip):
    """
    For more detail: https://www.tensorflow.org/guide/data_performance
    """
    train_dataset = train_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    return train_dataset, test_dataset
