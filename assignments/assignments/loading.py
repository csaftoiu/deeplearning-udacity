import os
import os.path as P
import sys
import tarfile

import numpy as np
from scipy import ndimage
from six.moves import cPickle as pickle
from six.moves.urllib.request import urlretrieve


num_classes = 10
image_size = 28      # Pixel width and height.
data_dir = "data"


def letter_for(label):
    """Return the letter for a given label."""
    return "ABCDEFGHIJ"[label]


def sizeof_fmt(num, suffix='B'):
    for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)


def maybe_download(filename, expected_bytes, force=False):
    """Download a file if not present, and make sure it's the right size."""
    filepath = P.join(data_dir, filename)
    if force or not P.exists(filepath):
        print("Downloading %s, %s bytes..." % (filename, sizeof_fmt(expected_bytes)))
        url = 'http://commondatastorage.googleapis.com/books1000/'
        filename, _ = urlretrieve(url + filename, filepath)

    statinfo = os.stat(filepath)
    if statinfo.st_size == expected_bytes:
        print('Found and verified', filename)
    else:
        raise Exception('Failed to verify ' + filename + '. Can you get to it with a browser?')

    return filename


def maybe_extract(filename, force=False):
    filepath = P.join(data_dir, filename)
    root = P.splitext(P.splitext(filepath)[0])[0]  # remove .tar.gz
    if P.isdir(root) and not force:
        # You may override by setting force=True.
        print('%s already present - Skipping extraction of %s.' % (root, filename))
    else:
        print('Extracting data for %s. This may take a while. Please wait.' % root)
        tar = tarfile.open(filepath)
        sys.stdout.flush()
        tar.extractall()
        tar.close()
    data_folders = [
        P.join(root, d) for d in sorted(os.listdir(root))
        if P.isdir(P.join(root, d))]
    return data_folders


def load_letter(folder, min_num_images, image_size):
    """Load the data for a single letter label."""
    pixel_depth = 255.0

    image_files = os.listdir(folder)
    dataset = np.ndarray(shape=(len(image_files), image_size, image_size),
                         dtype=np.float32)
    image_index = 0
    print(folder)
    for image in os.listdir(folder):
        image_file = P.join(folder, image)
        try:
            image_data = (ndimage.imread(image_file).astype(float) -
                          pixel_depth / 2) / (pixel_depth / 2)
            if image_data.shape != (image_size, image_size):
                raise Exception('Unexpected image shape: %s' % str(image_data.shape))
            dataset[image_index, :, :] = image_data
            image_index += 1
        except IOError as e:
            print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')

    num_images = image_index
    dataset = dataset[0:num_images, :, :]
    if num_images < min_num_images:
        raise Exception('Many fewer images than expected: %d < %d' %
                        (num_images, min_num_images))

    print('Full dataset tensor:', dataset.shape)
    print('Mean:', np.mean(dataset))
    print('Standard deviation:', np.std(dataset))
    return dataset


def maybe_pickle(data_folders, min_num_images_per_class,
                 image_size, force=False):
    dataset_names = []
    for folder in data_folders:
        set_filename = folder + '.pickle'
        dataset_names.append(set_filename)
        if P.exists(set_filename) and not force:
            # You may override by setting force=True.
            print('%s already present - Skipping pickling.' % set_filename)
        else:
            print('Pickling %s.' % set_filename)
            dataset = load_letter(folder, min_num_images_per_class, image_size)
            try:
                with open(set_filename, 'wb') as f:
                    pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
            except Exception as e:
                print('Unable to save data to', set_filename, ':', e)

    return dataset_names


def load_datasets():
    """Download, extract, and pickle the notMNIST datasets."""
    train_filename = maybe_download('notMNIST_large.tar.gz', 247336696)
    test_filename = maybe_download('notMNIST_small.tar.gz', 8458043)

    train_folders = maybe_extract(train_filename)
    test_folders = maybe_extract(test_filename)
    if not (len(train_folders) == len(test_folders) == num_classes):
        raise Exception('Expected %d folders, one per class. Found %d and %d instead.' % (
                num_classes, len(train_folders), len(test_folders)))
    print("Dataset folders: %s, %s" % (train_folders, test_folders))

    # load datasets
    train_datasets = maybe_pickle(train_folders, 45000, image_size)
    test_datasets = maybe_pickle(test_folders, 1800, image_size)

    return train_datasets, test_datasets
