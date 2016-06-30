from __future__ import print_function

import random

import matplotlib.pyplot as plt
import numpy as np

from assignments.loading import load_datasets, letter_for
from assignments.dataset import (
    get_training_data, sanitize_training_data, visually_check_data,
    measure_overlap, flatten_training_data
)
from assignments import classification


def main():
    # initialize
    np.random.seed(133)

    train_datasets, test_datasets = load_datasets()

    train_sizes = (
        # 50,
        # 100,
        1000,
        # 5000,
        # 10000, 50000, 200000
    )

    for train_size in train_sizes:
        training_data = get_training_data(
            train_datasets, test_datasets,
            train_size=train_size, valid_size=train_size, test_size=10000,
            store_pickle=True)
        import pprint; pprint.pprint(measure_overlap(training_data))
        training_data = sanitize_training_data(training_data)
        training_data = flatten_training_data(training_data)
        lr = classification.fit_sklearn_logisic_regression(training_data['train'])
        # lr = classification.fit_sklearn_sgd(training_data['train'])

        print("Accuracy trained from %d samples on %d 'valid' samples is %.2f%%" % (
            train_size, train_size // 20, classification.get_accuracy(lr, training_data['valid']) * 100,
        ))

        # # check some from test
        # test = training_data['test']
        # for _ in xrange(100):
        #     i = random.randint(0, len(test['data']) - 1)
        #     guess = lr.predict(classification.flatten_image_arrays(np.array([test['data'][i]])))
        #     if guess == test['labels'][i]:
        #         print("Correctly guessed %s!" % (letter_for(guess),))
        #     else:
        #         print("Incorrectly guessed %s for %s" % (
        #             letter_for(guess), letter_for(test['labels'][i])))
        #
        #     plt.imshow(test['data'][i])
        #     plt.show()




    # sanitized_data = sanitize_training_data(training_data)


if __name__ == "__main__":
    main()

    # predictions = lr.predict(flatten_image_arrays(data['data']))
