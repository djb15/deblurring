import tensorflow as tf
import numpy as np

def build_network(input_layer):
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=128,
        kernel=[19, 19],
        padding='same',
        activation=tf.nn.relu
    )

    conv2 = tf.layers.conv2d(
        inputs=conv1,
        filters=320,
        kernel=[1, 1],
        padding='same',
        activation=tf.nn.relu
    )

    conv3 = tf.layers.conv2d(
        inputs=conv2,
        filters=320,
        kernel=[1, 1],
        padding='same',
        activation=tf.nn.relu
    )

    conv4 = tf.layers.conv2d(
        inputs=conv3,
        filters=320,
        kernel=[1, 1],
        padding='same',
        activation=tf.nn.relu
    )

    conv5 = tf.layers.conv2d(
        inputs=conv4,
        filters=128,
        kernel=[1, 1],
        padding='same',
        activation=tf.nn.relu
    )

    conv6 = tf.layers.conv2d(
        inputs=conv5,
        filters=128,
        kernel=[3, 3],
        padding='same',
        activation=tf.nn.relu
    )

    conv7 = tf.layers.conv2d(
        inputs=conv6,
        filters=512,
        kernel=[1, 1],
        padding='same',
        activation=tf.nn.relu
    )

    conv8 = tf.layers.conv2d(
        inputs=conv7,
        filters=128,
        kernel=[5, 5],
        padding='same',
        activation=tf.nn.relu
    )

    conv9 = tf.layers.conv2d(
        inputs=conv8,
        filters=128,
        kernel=[5, 5],
        padding='same',
        activation=tf.nn.relu
    )

    conv10 = tf.layers.conv2d(
        inputs=conv9,
        filters=128,
        kernel=[3, 3],
        padding='same',
        activation=tf.nn.relu
    )

    conv11 = tf.layers.conv2d(
        inputs=conv10,
        filters=128,
        kernel=[5, 5],
        padding='same',
        activation=tf.nn.relu
    )

    conv12 = tf.layers.conv2d(
        inputs=conv11,
        filters=128,
        kernel=[5, 5],
        padding='same',
        activation=tf.nn.relu
    )

    conv13 = tf.layers.conv2d(
        inputs=conv12,
        filters=256,
        kernel=[1, 1],
        padding='same',
        activation=tf.nn.relu
    )

    conv14 = tf.layers.conv2d(
        inputs=conv13,
        filters=64,
        kernel=[7, 7],
        padding='same',
        activation=tf.nn.relu
    )

    conv15 = tf.layers.conv2d(
        inputs=conv14,
        filters=3,
        kernel=[7, 7],
        padding='same',
        activation=tf.nn.relu #Think this should actually be linear
    )
