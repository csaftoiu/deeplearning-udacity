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


def main_relunet(summarize):
    # initialize
    training_sets = load_training_sets()

    # stochastic gradient descent
    batch_size = 128
    initial_learning_rate = 0.0075
    learning_rate_steps = 200
    relus = [dataset.image_size**2, dataset.image_size**2]

    graph = tf.Graph()
    with graph.as_default():
        with tf.name_scope("training_data"):
            train = {
                'data': tf.placeholder(tf.float32, shape=(batch_size, dataset.image_size ** 2)),
                'labels': tf.placeholder(tf.float32, shape=(batch_size, dataset.num_classes)),
            }
            batch_offset = tf.random_uniform(
                (1,), dtype=tf.int32,
                minval=0, maxval=len(training_sets['test']['labels']) - batch_size)

        with tf.name_scope("validation_data"):
            valid = tf_dataset(training_sets['valid'])

        with tf.name_scope("testing_data"):
            test = tf_dataset(training_sets['test'])

        # create & initialize training parameters
        def make_weight(from_, to):
            return tf.Variable(tf.truncated_normal([from_, to], stddev=0.5))

        def make_bias(to):
            return tf.Variable(tf.truncated_normal([to], stddev=0.5))

        layer_sizes = [dataset.image_size**2] + relus + [dataset.num_classes]
        with tf.name_scope("parameters"):
            with tf.name_scope("weights"):
                weights = [make_weight(layer_sizes[i], layer_sizes[i+1])
                           for i in xrange(len(layer_sizes) - 1)]
            with tf.name_scope("biases"):
                biases = [make_bias(layer_sizes[i + 1])
                          for i in xrange(len(layer_sizes) - 1)]

        # pipeline to get a logit
        def build_logit_pipeline(data):
            # X --> *W1 --> +b1 --> relu --> *W2 --> +b2 ... --> softmax etc...
            pipeline = data

            for i in xrange(len(layer_sizes) - 1):
                with tf.name_scope("linear%d" % i):
                    pipeline = tf.matmul(pipeline, weights[i])
                    pipeline = pipeline + biases[i]

                if i != len(layer_sizes) - 2:
                    with tf.name_scope("relu%d" % i):
                        pipeline = tf.nn.relu(pipeline)

            return pipeline

        with tf.name_scope("training_pipeline"):
            train_logits = build_logit_pipeline(train['data'])
            train_prediction = tf.nn.softmax(train_logits)

        with tf.name_scope("validation_pipeline"):
            valid_logits = build_logit_pipeline(valid['data'])
            valid_prediction = tf.nn.softmax(valid_logits)

        with tf.name_scope("testing_pipeline"):
            test_logits = build_logit_pipeline(test['data'])
            test_prediction = tf.nn.softmax(test_logits)

        # the optimization
        # loss function is the mean of the cross-entropy of (the softmax of the
        # logits, the labels). This is built in exactly!
        with tf.name_scope("loss"):
            loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(train_logits, train['labels'])
            )
            tf.scalar_summary('loss', loss)

        # learning rate
        with tf.name_scope("global_step"):
            global_step = tf.Variable(0, trainable=False)

        with tf.name_scope("learning_rate"):
            learning_rate = tf.train.exponential_decay(
                initial_learning_rate, global_step, learning_rate_steps, 0.96)
            # learning_rate = tf.constant(initial_learning_rate)
            tf.scalar_summary('learning_rate', learning_rate)

        # optimizer - gradient descent, minimizing the loss function
        with tf.name_scope("optimizer"):
            optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

        summaries = tf.merge_all_summaries()
        writer = tf.train.SummaryWriter('logs/', graph=graph)

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

                summary, _, loss_val, predictions = session.run(
                    [summaries, optimizer, loss, train_prediction],
                    feed_dict={
                        train['data']: batch['data'],
                        train['labels']: batch['labels'],
                    },
                )

                if summarize:
                    writer.add_summary(summary, step)

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
    main_relunet(summarize=True)
