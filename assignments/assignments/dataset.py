from __future__ import print_function

from itertools import combinations
import os
import os.path as P

import numpy as np
from six.moves import cPickle as pickle

from .loading import image_size, data_dir


def make_arrays(nb_rows, img_size):
    if nb_rows:
        dataset = np.ndarray((nb_rows, img_size, img_size), dtype=np.float32)
        labels = np.ndarray(nb_rows, dtype=np.int32)
    else:
        dataset, labels = None, None
    return dataset, labels


def randomize(dataset, labels):
    if dataset is None or labels is None:
        return dataset, labels

    permutation = np.random.permutation(labels.shape[0])
    shuffled_dataset = dataset[permutation, :, :]
    shuffled_labels = labels[permutation]
    return shuffled_dataset, shuffled_labels


def merge_datasets(pickle_files, image_size, train_size, valid_size):
    """Merge datasets from the pickle files, picking `train_size` for training
    and `valid_size` for validation from each class.

    Return the randomized data sets."""
    num_classes = len(pickle_files)
    valid_dataset, valid_labels = make_arrays(valid_size, image_size)
    train_dataset, train_labels = make_arrays(train_size, image_size)
    vsize_per_class = valid_size // num_classes
    tsize_per_class = train_size // num_classes

    start_v, start_t = 0, 0
    end_v, end_t = vsize_per_class, tsize_per_class
    end_l = vsize_per_class + tsize_per_class
    for label, pickle_file in enumerate(pickle_files):
        try:
            with open(pickle_file, 'rb') as f:
                print("Processing %s..." % pickle_file)
                letter_set = pickle.load(f)
                # let's shuffle the letters to have random validation and training set
                np.random.shuffle(letter_set)
                if valid_dataset is not None:
                    valid_letter = letter_set[:vsize_per_class, :, :]
                    valid_dataset[start_v:end_v, :, :] = valid_letter
                    valid_labels[start_v:end_v] = label
                    start_v += vsize_per_class
                    end_v += vsize_per_class

                train_letter = letter_set[vsize_per_class:end_l, :, :]
                train_dataset[start_t:end_t, :, :] = train_letter
                train_labels[start_t:end_t] = label
                start_t += tsize_per_class
                end_t += tsize_per_class
        except Exception as e:
            print('Unable to process data from', pickle_file, ':', e)
            raise

    # i question the necessity of this as the sets were already shuffled ...
    valid_dataset, valid_labels = randomize(valid_dataset, valid_labels)
    train_dataset, train_labels = randomize(train_dataset, train_labels)
    return valid_dataset, valid_labels, train_dataset, train_labels


def get_training_data(train_datasets, test_datasets,
                      train_size, valid_size, test_size,
                      force_regen=False, store_pickle=True):
    """Create and return training, validation, and testing datasets.
    If force_regen is False, it will generate the dataset, even if a pickled
    version is available."""
    TRAINING_DATA_FILENAME = P.join(data_dir, 'training_data-%d-%d-%d.pickle' % (
        train_size, valid_size, test_size,
    ))
    if not force_regen and os.path.exists(TRAINING_DATA_FILENAME):
        print("Loading training data from pickle...")
        with open(TRAINING_DATA_FILENAME, 'rb') as f:
            return pickle.load(f)

    valid_dataset, valid_labels, train_dataset, train_labels = merge_datasets(
        train_datasets, image_size, train_size, valid_size)
    _, _, test_dataset, test_labels = merge_datasets(
        test_datasets, image_size, test_size, valid_size=0)

    print('Training:', train_dataset.shape, train_labels.shape)
    print('Validation:', valid_dataset.shape, valid_labels.shape)
    print('Testing:', test_dataset.shape, test_labels.shape)

    result = {
        'train': {
            'data': train_dataset,
            'labels': train_labels,
        },
        'valid': {
            'data': valid_dataset,
            'labels': valid_labels,
        },
        'test': {
            'data': test_dataset,
            'labels': test_labels,
        },
    }

    if store_pickle:
        with open(TRAINING_DATA_FILENAME, 'wb') as f:
            pickle.dump(result, f, pickle.HIGHEST_PROTOCOL)

    return result


def measure_overlap(training_data):
    """Measure the overlap in the training data sets.

    Uses a hash function, so it's not 100% exact, but close enough.

    Return a dict of (setname, sename) to number of overlapping letters."""
    contained_letters = {}
    for name, dataset in training_data.items():
        for letter in dataset['data']:
            letter.flags.writeable = False
            contained_letters.setdefault(name, set()).add(hash(letter.data))

    return {(a, b): len(contained_letters[a] & contained_letters[b])
            for a, b in combinations(contained_letters.keys(), 2)}


def shallow_copy_training_data(training_data):
    """Return shallow copy of training data, where the numpy arrays are not
    copied, but the dicts are deep-copied."""
    result = {}
    for name, d in training_data.items():
        result[name] = dict(d)
    return result


def remove_overlaps(training_data, keep_src, remove_src):
    """Given training data, return a new training data set where
    images from remove_src whose hash equals the hash of an image in keep_src
    have been removed."""
    assert keep_src in training_data
    assert remove_src in training_data

    contained_letters = {}
    hash_to_im = {}
    for name in [keep_src, remove_src]:
        for i, letter in enumerate(training_data[name]['data']):
            letter.flags.writeable = False
            h = hash(letter.data)
            contained_letters.setdefault(name, set()).add(h)
            hash_to_im.setdefault(name, {}).setdefault(h, []).append(i)

    common = contained_letters[keep_src] & contained_letters[remove_src]

    # calculate back the indices to remove
    rem_indices = [i for c in common for i in hash_to_im[remove_src][c]]

    # remove them from a sort-of shallow copy
    training_data = shallow_copy_training_data(training_data)
    training_data[remove_src]['data'] = np.delete(training_data[remove_src]['data'],
                                                  rem_indices, axis=0)
    training_data[remove_src]['labels'] = np.delete(training_data[remove_src]['labels'],
                                                  rem_indices, axis=0)
    return training_data


def sanitize_training_data(training_data):
    """Sanitize the training data **in place** by removing overlaps."""
    print("Removing overlaps between train and valid...")
    training_data = remove_overlaps(training_data, 'train', 'valid')
    print("Removing overlaps between train and test...")
    training_data = remove_overlaps(training_data, 'train', 'test')
    print("Removing overlaps between valid and test...")
    training_data = remove_overlaps(training_data, 'valid', 'test')

    # check no more overlaps
    assert all(overlaps == 0 for overlaps in measure_overlap(training_data).values())

    return training_data


def visually_check_data(training_data, n=5):
    """Use matplotlib to show the letter & corresponding label for `n` random instances,
    to verify the data still matches."""
    import matplotlib.pyplot as plt
    import random

    for name, data in training_data.items():
        print("Checking %s..." % name)
        for which in xrange(n):
            # show last image for first run, otherwise a random one
            i = random.randint(0, len(data['data']) - 1) if which > 0 else len(data['data']) - 1
            print("Label is %s" % ("ABCDEFGHIJ"[data['labels'][i]]))
            plt.imshow(data['data'][i])
            plt.show()


def flatten_image_arrays(imarrays):
    """Given a 3d array of images (array of 2-dim arrays), flatten it
    to a 2d array (array of 1-dim arrays)"""
    return imarrays.reshape(imarrays.shape[0], -1)


def flatten_training_data(training_data):
    """Flatten the training data."""
    result = shallow_copy_training_data(training_data)
    for which, data in result.items():
        data['data'] = flatten_image_arrays(data['data'])
    return result
