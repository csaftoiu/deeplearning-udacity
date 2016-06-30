from __future__ import print_function

import pprint

import matplotlib.pyplot as plt
# from scipy import ndimage
# from sklearn.linear_model import LogisticRegression

import numpy as np

from assignment1.loading import load_datasets
from assignment1.datasets import get_training_data, measure_overlap, sanitize_training_data



def main():
    # initialize
    np.random.seed(133)

    train_datasets, test_datasets = load_datasets()
    training_data = get_training_data(
        train_datasets, test_datasets,
        train_size=200000, valid_size=10000, test_size=10000)

    sanitized_data = sanitize_training_data(training_data)


if __name__ == "__main__":
    main()

