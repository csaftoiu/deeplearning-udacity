from __future__ import print_function

import random

import matplotlib.pyplot as plt
import numpy as np

from assignments.loading import load_datasets, letter_for
from assignments import loading, dataset, classification


def main():
    # initialize
    np.random.seed(133)

    train_datasets, test_datasets = loading.load_datasets()
    training_data = dataset.get_training_data(
        train_datasets, test_datasets,
        train_size=200000, valid_size=10000, test_size=10000,
        store_pickle=True)

    training_data = dataset.flatten_training_data(training_data)

    for which, data in training_data.items():
        print(which, data['data'].shape)

    # training_data = sanitize_training_data(training_data)


if __name__ == "__main__":
    main()
