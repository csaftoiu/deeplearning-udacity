from sklearn.linear_model import LogisticRegression


def flatten_image_arrays(imarrays):
    """Given a 3d array of images (array of 2-dim arrays), flatten it
    to a 2d array (array of 1-dim arrays)"""
    return imarrays.reshape(imarrays.shape[0], -1)


def fit_sklearn_logisic_regression(data):
    """Given `data` (a dict with `data` and `labels`), create and
     return a LogisticRegression trained on the data."""
    lr = LogisticRegression()

    print("Fitting regression on %d data points..." % len(data['data']))
    lr.fit(flatten_image_arrays(data['data']), data['labels'])

    return lr


def get_accuracy(lr, data):
    """Get the accuracy, as a float, of the LogisticRegression on the given data."""
    predictions = lr.predict(flatten_image_arrays(data['data']))

    correct = sum(1 for prediction, label in zip(predictions, data['labels'])
                  if prediction == label)
    return correct / float(len(data['labels']))
