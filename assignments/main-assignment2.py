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
    """Given a prediction and the one-hot labels, return the accuracy as a percentage."""
    # argmax of prediction == which label it thinks
    # argmax of label = which label
    # equate, sum = number of accurate predictions
    num_correct = np.sum(np.argmax(predictions, axis=1) == np.argmax(labels, axis=1))
    return 100.0 * num_correct / predictions.shape[0]


def main_gradient_descent():
    # initialize
    np.random.seed(133)

    training_sets = load_training_sets()

    # linear regression with tensor flow
    train_subset_size = 10000
    train_subset = dataset.subset(training_sets['train'], train_subset_size)
    initial_learning_rate = 5.0
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

        # exponential decay learning rate
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(
            initial_learning_rate, global_step, 300, 0.96)

        # optimizer - gradient descent, minimizing the loss function
        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

        # predictions, so we can compare output
        train_prediction = tf.nn.softmax(logits)
        valid_prediction = tf.nn.softmax(tf.matmul(valid['data'], weights) + biases)
        test_prediction = tf.nn.softmax(tf.matmul(test['data'], weights) + biases)

    # now that graph is created, run the session
    with tf.Session(graph=graph) as session:
        # initialize everything
        tf.initialize_all_variables().run()

        print("Initialized")

        while True:  # for step in range(num_steps):
            try:
                step = global_step.eval()
                _, loss_val, predictions = session.run([optimizer, loss, train_prediction])
                if step % 50 == 0:
                    print("Global step: %d" % step)
                    print("Learning rate: %f" % learning_rate.eval())
                    print("Loss function: %f" % loss_val)

                    print("Accuracy on training data:   %.2f%%" % (
                        accuracy(predictions, train_subset['labels']),
                    ))
                    # evaluate predictions and see their accuracy
                    print("Accuracy on validation data: %.2f%%" % (
                        accuracy(valid_prediction.eval(), training_sets['valid']['labels']),
                    ))
            except KeyboardInterrupt:
                print("Stopping from keyboard interrupt.")
                break

        print('Test accuracy: %.2f%%' % (
            accuracy(test_prediction.eval(), training_sets['test']['labels']),
        ))


def main_sgd():
    # initialize
    np.random.seed(133)

    training_sets = load_training_sets()

    # stochastic gradient descent
    batch_size = 128
    initial_learning_rate = 5.0

    graph = tf.Graph()
    with graph.as_default():
        train = {
            'data': tf.placeholder(tf.float32, shape=(batch_size, dataset.image_size ** 2)),
            'labels': tf.placeholder(tf.float32, shape=(batch_size, dataset.num_classes)),
        }
        batch_offset = tf.random_uniform((1,), minval=0, maxval=len(training_sets['test']['labels']) - batch_size)

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

        # learning rate
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(
            initial_learning_rate, global_step, 300, 0.96)
        # learning_rate = tf.constant(initial_learning_rate)

        # optimizer - gradient descent, minimizing the loss function
        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

        # predictions, so we can compare output
        train_prediction = tf.nn.softmax(logits)
        valid_prediction = tf.nn.softmax(tf.matmul(valid['data'], weights) + biases)
        test_prediction = tf.nn.softmax(tf.matmul(test['data'], weights) + biases)

    # now that graph is created, run the session
    with tf.Session(graph=graph) as session:
        # initialize everything
        tf.initialize_all_variables().run()

        print("Initialized")

        while True:  # for step in range(num_steps):
            try:
                step = global_step.eval()
                offs = batch_offset.eval()[0]
                batch = {
                    'data': training_sets['train']['data'][offs:offs + batch_size, :],
                    'labels': training_sets['train']['labels'][offs:offs + batch_size],
                }

                _, loss_val, predictions = session.run([optimizer, loss, train_prediction], feed_dict={
                    train['data']: batch['data'],
                    train['labels']: batch['labels'],
                })

                if step % 200 == 0:
                    print("Global step: %d" % step)
                    print("Learning rate: %f" % learning_rate.eval())
                    print("Batch loss function: %f" % loss_val)

                    print("Accuracy on batch data:   %.2f%%" % (
                        accuracy(predictions, batch['labels']),
                    ))
                    # evaluate predictions and see their accuracy
                    print("Accuracy on validation data: %.2f%%" % (
                        accuracy(valid_prediction.eval(), training_sets['valid']['labels']),
                    ))
            except KeyboardInterrupt:
                print("Stopping from keyboard interrupt.")
                break

        print('Test accuracy: %.2f%%' % (
            accuracy(test_prediction.eval(), training_sets['test']['labels']),
        ))


if __name__ == "__main__":
    main_sgd()
