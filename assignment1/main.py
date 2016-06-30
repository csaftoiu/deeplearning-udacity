from __future__ import print_function

import random

import matplotlib.pyplot as plt
import numpy as np

from assignment1.loading import load_datasets, letter_for
from assignment1.dataset import get_training_data, sanitize_training_data, visually_check_data
from assignment1 import classification


def main():
    # initialize
    np.random.seed(133)

    train_datasets, test_datasets = load_datasets()

    for train_size in (1000,):  # 50, 100, 1000, 5000):
        training_data = get_training_data(
            train_datasets, test_datasets,
            train_size=train_size, valid_size=train_size//20, test_size=10000)
        lr = classification.fit_sklearn_logisic_regression(training_data['train'])

        print("Accuracy trained from %d samples on %d 'test' samples is %.2f%%" % (
            train_size, 10000, classification.get_accuracy(lr, training_data['test']) * 100,
        ))
        print("Accuracy trained from %d samples on %d 'valid' samples is %.2f%%" % (
            train_size, 10000, classification.get_accuracy(lr, training_data['valid']) * 100,
        ))

        # check some from validation
        valid = training_data['valid']
        for _ in xrange(100):
            i = random.randint(0, len(valid['data']) - 1)
            guess = lr.predict(classification.flatten_image_arrays(np.array([valid['data'][i]])))
            if guess == valid['labels'][i]:
                print("Correctly guessed %s!" % (letter_for(guess),))
            else:
                print("Incorrectly guessed %s for %s" % (
                    letter_for(guess), letter_for(valid['labels'][i])))

            plt.imshow(valid['data'][i])
            plt.show()




    # sanitized_data = sanitize_training_data(training_data)


if __name__ == "__main__":
    main()

    # predictions = lr.predict(flatten_image_arrays(data['data']))
