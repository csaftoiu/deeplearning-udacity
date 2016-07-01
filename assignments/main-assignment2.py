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
    training_sets = dataset.get_training_sets(
        train_datasets, test_datasets,
        train_size=200000, valid_size=10000, test_size=10000,
        store_pickle=True)

    training_sets = dataset.mapsets(dataset.flatten, training_sets)
    training_sets = dataset.mapsets(dataset.onehotify, training_sets)

    for which, data in training_sets.items():
        print(which, data['data'].shape)
        print(data['labels'][:50])


if __name__ == "__main__":
    main()
