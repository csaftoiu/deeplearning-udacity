"""Usage:
    run.py [-c <config>]

Options:
    -c --config <config>    Which configuration to run. [default: main]
"""

from __future__ import print_function

import math
import os
import time

from docopt import docopt
import numpy as np
import tensorflow as tf

from assignments import loading, dataset, classification


def load_training_sets(train_size, valid_size, test_size):
    """Get the training sets for use in tensorflow."""
    train_datasets, test_datasets = loading.load_datasets()
    training_sets = dataset.get_training_sets(
        train_datasets, test_datasets,
        train_size=train_size, valid_size=valid_size, test_size=test_size,
        store_pickle=True)

    training_sets = dataset.mapsets(dataset.flatten, training_sets)
    training_sets = dataset.mapsets(dataset.onehotify, training_sets)

    return training_sets


def tf_dataset(dataset, prefix=None):
    """Get the dataset as a tensorflow constant. Optionally get a
    subset of the dataset."""
    return {
        'data': tf.constant(dataset['data'], name=('%s_data' % prefix) if prefix else None),
        'labels': tf.constant(dataset['labels'], name=('%s_labels' % prefix) if prefix else None)
    }


def accuracy(predictions, labels):
    """Given a prediction and the one-hot labels, return the accuracy as a percentage."""
    # argmax of prediction == which label it thinks
    # argmax of label = which label
    # equate, sum = number of accurate predictions
    num_correct = np.sum(np.argmax(predictions, axis=1) == np.argmax(labels, axis=1))
    return 100.0 * num_correct / predictions.shape[0]


def flatten_variable(v):
    from operator import mul
    return tf.reshape(v, (int(reduce(mul, v.get_shape())),))


# store parameters
parameters = {
    'defaults': {
        # training sets sizes
        'train_size': 200000,
        'valid_size': 10000,
        'test_size': 10000,
        # batch size for SGD
        'batch_size': 128,
        # learning rate
        'initial_learning_rate': 0.05,
        'l2_loss_weight_scale': 0.01,
        'learning_rate_steps': 50.0,
        # feedback
        'step_print': 300,
        'step_eval': 300,
    },
    '84%_onelayer': {
        'l2_loss_weight_scale': 0.0,
        'relus': [dataset.image_size**2],
        'use_dropout': False,
    },
    '85.18%': {
        'relus': [dataset.image_size**2],
        'use_dropout': False,
    },
    '89.91%': {
        'relus': [dataset.image_size**2],
        'use_dropout': True,
    },
    '91.06%': {
        'relus': [dataset.image_size**2 * 4],
        'use_dropout': True,
    },
    'tiny_nodropout': {
        'train_size': 256,
        'relus': [dataset.image_size**2],
        'use_dropout': False,
    },
    'tiny_yesdropout': {
        'train_size': 256,
        'relus': [dataset.image_size**2],
        'use_dropout': True,
    },
    'main': {
        'relus': [dataset.image_size**2 * 4],
        'use_dropout': True,
    },
}


def main_relunet(summarize=True, **kwargs):
    def arg(name, _missing=object()):
        res = kwargs.get(name, parameters['defaults'].get(name, _missing))
        if res is _missing:
            raise ValueError("Parameter '%s' is required, has no default" % (name,))
        return res

    training_sets = load_training_sets(
        train_size=arg('train_size'),
        valid_size=arg('valid_size'),
        test_size=arg('test_size'),
    )

    graph = tf.Graph()
    with graph.as_default():
        # learning rate tweaking
        initial_learning_rate = tf.constant(arg('initial_learning_rate'), name='initial_learning_rate')
        learning_rate_steps = tf.constant(arg('learning_rate_steps'), name='learning_rate_steps')

        # loss penalization params
        l2_loss_weight = tf.constant(arg('l2_loss_weight_scale'), name='loss_weight_scale')

        with tf.name_scope("training_data"):
            train = {
                'data': tf.placeholder(tf.float32, shape=(arg('batch_size'), dataset.image_size ** 2),
                                       name='batch_input'),
                'labels': tf.placeholder(tf.float32, shape=(arg('batch_size'), dataset.num_classes),
                                         name='batch_labels'),
            }
            batch_offset = tf.random_uniform(
                (1,), dtype=tf.int32,
                minval=0, maxval=len(training_sets['train']['labels']) - arg('batch_size'),
                name='batch_offset')

        with tf.name_scope("validation_data"):
            valid = tf_dataset(training_sets['valid'], 'valid')

        with tf.name_scope("testing_data"):
            test = tf_dataset(training_sets['test'], 'test')

        # create & initialize training parameters
        def make_weight(from_, to, name=None):
            return tf.Variable(tf.truncated_normal([from_, to], stddev=0.5), name=name)

        def make_bias(to, name=None):
            return tf.Variable(tf.truncated_normal([to], stddev=0.5), name=name)

        layer_sizes = [dataset.image_size**2] + arg('relus') + [dataset.num_classes]
        with tf.name_scope("parameters"):
            with tf.name_scope("weights"):
                weights = [make_weight(layer_sizes[i], layer_sizes[i+1], name="weights_%d" % i)
                           for i in xrange(len(layer_sizes) - 1)]
                # for i, w in enumerate(weights):
                #     tf.histogram_summary('weights_%d' % i, w)

            with tf.name_scope("biases"):
                biases = [make_bias(layer_sizes[i + 1], name="biases_%d" % i)
                          for i in xrange(len(layer_sizes) - 1)]
                # for i, b in enumerate(biases):
                #     tf.histogram_summary('biases_%d' % i, b)

        # pipeline to get a logit
        def build_logit_pipeline(data, include_dropout):
            # X --> *W1 --> +b1 --> relu --> *W2 --> +b2 ... --> softmax etc...
            pipeline = data

            for i in xrange(len(layer_sizes) - 1):
                last = i == len(layer_sizes) - 2
                with tf.name_scope("linear%d" % i):
                    pipeline = tf.matmul(pipeline, weights[i])
                    pipeline = tf.add(pipeline, biases[i])

                if not last:
                    # insert relu after every one before the last
                    with tf.name_scope("relu%d" % i):
                        pipeline = tf.nn.relu(pipeline)
                        if include_dropout and arg('use_dropout'):
                            pipeline = tf.nn.dropout(pipeline, 0.5, name='dropout')

            return pipeline

        with tf.name_scope("training_pipeline"):
            train_logits = build_logit_pipeline(train['data'], include_dropout=True)
            train_prediction = tf.nn.softmax(train_logits, name='train_predictions')

        with tf.name_scope("validation_pipeline"):
            valid_logits = build_logit_pipeline(valid['data'], include_dropout=False)
            valid_prediction = tf.nn.softmax(valid_logits, name='valid_predictions')

        with tf.name_scope("testing_pipeline"):
            test_logits = build_logit_pipeline(test['data'], include_dropout=False)
            test_prediction = tf.nn.softmax(test_logits, name='test_predictions')

        with tf.name_scope("accuracy_variables"):
            # inserted via python code
            batch_accuracy = tf.Variable(0.0, trainable=False, name='batch_accuracy')
            valid_accuracy = tf.Variable(0.0, trainable=False, name='valid_accuracy')
            tf.scalar_summary('accuracy/batch', batch_accuracy)
            tf.scalar_summary('accuracy/valid', valid_accuracy)

        # the optimization
        # loss function is the mean of the cross-entropy of (the softmax of the
        # logits, the labels). This is built in exactly!
        with tf.name_scope("loss"):
            with tf.name_scope("loss_main"):
                loss_main = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits(train_logits, train['labels']),
                    name='loss_main',
                )
                tf.scalar_summary('loss/main', loss_main)

            with tf.name_scope("loss_weights"):
                # calculate l2 loss as the sum of losses of all weights and biases
                l2_loss_unweighted = tf.add_n([
                    tf.nn.l2_loss(w) for w in weights
                ] + [
                    tf.nn.l2_loss(b) for b in biases
                ])
                # l2_loss_unweighted = tf.nn.l2_loss(tf.concat(
                #     0,
                #     map(flatten_variable, weights) + map(flatten_variable, biases)
                # ), name='loss_weights')
                tf.scalar_summary('loss/weights_unscaled', l2_loss_unweighted)
                l2_loss = l2_loss_weight * l2_loss_unweighted
                tf.scalar_summary('loss/weights_scaled', l2_loss)

            loss = tf.add(loss_main, l2_loss, name='loss')
            tf.scalar_summary('loss/total', loss)

        # learning rate
        with tf.name_scope("global_step"):
            global_step = tf.Variable(0, trainable=False, name='global_step')

        learning_rate = tf.train.exponential_decay(
            initial_learning_rate, global_step, learning_rate_steps, 0.96,
            name='learning_rate')
        # learning_rate = tf.constant(initial_learning_rate, name='learning_rate')
        tf.scalar_summary('learning_rate', learning_rate)

        # optimizer - gradient descent, minimizing the loss function
        with tf.name_scope("optimizer"):
            optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

        summaries = tf.merge_all_summaries()
        writer = tf.train.SummaryWriter(os.path.join('logs', str(time.time())), graph=graph)

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
                    'data': training_sets['train']['data'][offs:offs + arg('batch_size'), :],
                    'labels': training_sets['train']['labels'][offs:offs + arg('batch_size')],
                }

                summary, _, loss_val, predictions = session.run(
                    [summaries, optimizer, loss, train_prediction],
                    feed_dict={
                        train['data']: batch['data'],
                        train['labels']: batch['labels'],
                    },
                )

                if step % arg('step_print') == 0:
                    _batch_accuracy = accuracy(predictions, batch['labels'])
                    batch_accuracy.assign(_batch_accuracy).op.run()

                    print("-----")
                    print("Global step: %d" % step)
                    print("log(Learning rate): %f" % math.log(learning_rate.eval()))
                    print("Batch loss function: %f" % loss_val)

                    print("Accuracy on batch data:   %.2f%%" % (
                        _batch_accuracy,
                    ))

                if step % arg('step_eval') == 0:
                    # evaluate predictions and see their accuracy
                    _valid_accuracy = accuracy(valid_prediction.eval(), training_sets['valid']['labels'])
                    valid_accuracy.assign(_valid_accuracy).op.run()

                    print("Accuracy on validation data: %.2f%%" % (
                        _valid_accuracy
                    ))

                if summarize:
                    writer.add_summary(summary, step)

            except KeyboardInterrupt:
                print("Stopping from keyboard interrupt.")
                break

        print('Test accuracy: %.2f%%' % (
            accuracy(test_prediction.eval(), training_sets['test']['labels']),
        ))


if __name__ == "__main__":
    args = docopt(__doc__)
    main_relunet(summarize=True, **parameters[args['--config']])
