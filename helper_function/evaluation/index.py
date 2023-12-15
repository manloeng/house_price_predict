import tensorflow as tf

from helper_function.evaluation.metrics import make_preds, evaluate_preds


def evaluate(model_location, test_windows, test_labels):
    """
    Evaluates the model provided with the test data
    :param model_location: model_experiments/model_1_dense/
    """
    print("Begin Evaluating")
    model = tf.keras.models.load_model(model_location)
    print(f"evaluation: {model.evaluate(test_windows, test_labels)}")

    model_preds = make_preds(model, test_windows)
    print(f"length of preds: {len(model_preds)}")
    print(f"preds: {model_preds[:10]}")

    model_results = evaluate_preds(
        y_true=tf.squeeze(test_labels),
        y_pred=model_preds
    )
    print(f"results:{model_results}")
    return model_results
