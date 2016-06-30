from __future__ import print_function

import numpy as np

from assignment1.loading import load_datasets
from assignment1.dataset import get_training_data, sanitize_training_data, visually_check_data
from assignment1 import classification


def main():
    # initialize
    np.random.seed(133)

    train_datasets, test_datasets = load_datasets()

    for train_size in (50, 100, 1000, 5000, 200000):
        training_data = get_training_data(
            train_datasets, test_datasets,
            train_size=train_size, valid_size=train_size//20, test_size=10000)
        lr = classification.fit_sklearn_logisic_regression(training_data['train'])
        accuracy = classification.get_accuracy(lr, training_data['test'])

        print("Accuracy trained from %d samples on %d samples is %.2f%%" % (
            train_size, 10000, accuracy * 100,
        ))
    # sanitized_data = sanitize_training_data(training_data)


if __name__ == "__main__":
    main()

