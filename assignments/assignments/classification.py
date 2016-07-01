from sklearn.linear_model import LogisticRegression, SGDClassifier


def fit_sklearn_logisic_regression(data):
    """Given flattened `data` (a dict with `data` and `labels`), create and
     return a LogisticRegression trained on the data."""
    lr = LogisticRegression()

    print("Fitting regression on %d data points..." % len(data['data']))
    lr.fit(data['data'], data['labels'])

    return lr


def fit_sklearn_sgd(data):
    sgd = SGDClassifier()

    print("Fitting stochastic gradient descent on %d data points..." % len(data['data']))
    sgd.fit(data['data'], data['labels'])

    return sgd


def get_accuracy(lr, data):
    """Get the accuracy, as a float, of the LogisticRegression on the given data."""
    predictions = lr.predict(data['data'])

    correct = sum(1 for prediction, label in zip(predictions, data['labels'])
                  if prediction == label)
    return correct / float(len(data['labels']))
