from __future__ import print_function

import random

import numpy as np
import tensorflow as tf

from assignments import loading, dataset, classification

from six.moves import range


def load_training_sets():
    """Get the training sets for use in tensorflow."""
    train_datasets, test_datasets = loading.load_datasets()
    training_sets = dataset.get_training_sets(
        train_datasets, test_datasets,
        train_size=200000, valid_size=10000, test_size=10000,
        store_pickle=True)

    training_sets = dataset.mapsets(dataset.flatten, training_sets)
    training_sets = dataset.mapsets(dataset.onehotify, training_sets)

    return training_sets


def tf_dataset(dataset):
    """Get the dataset as a tensorflow constant. Optionally get a
    subset of the dataset."""
    return {
        'data': tf.constant(dataset['data']),
        'labels': tf.constant(dataset['labels'])
    }


def accuracy(predictions, labels):
    """Given a prediction and the one-hot labels, return the accuracy."""
    # argmax of prediction == which label it thinks
    # argmax of label = which label
    # equate, sum = number of accurate predictions
    num_correct = np.sum(np.argmax(predictions, axis=1) == np.argmax(labels, axis=1))
    return num_correct / float(predictions.shape[0])


def main():
    # initialize
    np.random.seed(133)

    training_sets = load_training_sets()

    # linear regression with tensor flow
    train_subset_size = 10000
    train_subset = dataset.subset(training_sets['train'], train_subset_size)
    learning_rate = 0.5
    num_steps = 801

    graph = tf.Graph()
    with graph.as_default():
        train = tf_dataset(train_subset)
        valid = tf_dataset(training_sets['valid'])
        test = tf_dataset(training_sets['test'])

        # initialize training parameters
        weights = tf.Variable(
            tf.truncated_normal([dataset.image_size ** 2, dataset.num_classes])
        )
        biases = tf.Variable(tf.zeros([dataset.num_classes]))

        # training computation
        # Y = WX + b
        logits = tf.matmul(train['data'], weights) + biases
        # loss function is the mean of the cross-entropy of (the softmax of the
        # logits, the labels). This is built in exactly!
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits, train['labels'])
        )

        # optimizer - gradient descent, minimizing the loss function
        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

        # predictions, so we can compare output
        train_prediction = tf.nn.softmax(logits)
        valid_prediction = tf.nn.softmax(tf.matmul(valid['data'], weights) + biases)
        test_prediction = tf.nn.softmax(tf.matmul(test['data'], weights) + biases)

    # now that graph is created, run the session
    with tf.Session(graph=graph) as session:
        # initialize everything
        tf.initialize_all_variables().run()

        print("Initialized")

        for step in range(num_steps):
            _, loss_val, predictions = session.run([optimizer, loss, train_prediction])
            if step % 50 == 0:
                print("Loss function: %f" % loss_val)

                print("Accuracy on training data:   %.2f%%" % (
                    accuracy(predictions, train_subset['labels']) * 100.0,
                ))
                # evaluate predictions and see their accuracy
                print("Accuracy on validation data: %.2f%%" % (
                    accuracy(valid_prediction.eval(), training_sets['valid']['labels']) * 100.0,
                ))

        print('Test accuracy: %.2f%%' % (
            accuracy(test_prediction.eval(), training_sets['test']['labels'])*100.0,
        ))


if __name__ == "__main__":
    main()
